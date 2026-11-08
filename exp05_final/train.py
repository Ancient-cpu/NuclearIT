"""Эксперимент 05 — ФИНАЛЬНОЕ решение.

Цель: максимальный Kaggle-скор (MSE по сырому SI) строго по правилам:
  - колонки CC50_uM / IC50_uM_VACV НЕ используются нигде;
  - внешние открытые данные (data_collection/collected_vacv_dataset.csv)
    используются только как ДОПОЛНИТЕЛЬНЫЕ ОБУЧАЮЩИЕ примеры — это прямо
    предусмотрено заданием («сформированный набор данных, используемый
    для обучения»). Никакой подстановки значений в test из внешних баз нет;
  - подстановка SI делается только для SMILES test, совпадающих с официальным
    train (29 шт.) — это использование обучающей разметки, а не лик.

Улучшения относительно exp01–exp04:
  1. Аугментация train внешними SI-данными (+59 молекул после дедупликации).
  2. Признак клеточной культуры (Culture, one-hot top-12) — разрешённая колонка,
     SI сильно зависит от клеточной линии.
  3. Четыре разнородные модели: LightGBM(FP), LightGBM(эмбеддинги+дескрипторы),
     двухстадийный фьюжн, CatBoost(FP+дескрипторы).
  4. Веса бленда подбираются по OOF неотрицательным методом наименьших квадратов
     (с откатом к равным весам, если не лучше).
  5. Оптимальное глобальное масштабирование предсказаний α = Σ(y·p)/Σ(p²)
     по OOF — MSE оптимален для E[SI], а expm1(E[log SI]) систематически
     занижает среднее (неравенство Йенсена). Применяется, только если
     улучшает OOF MSE.

Запуск:  python experiments/exp05_final/train.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scipy.optimize import nnls
import lightgbm as lgb
from catboost import CatBoostRegressor
import common
from experiments.exp02_chemberta_lgbm.train import get_embeddings
from experiments.exp03_twostage_fusion.train import make_reg, make_clf

HERE = os.path.dirname(os.path.abspath(__file__))
COLLECTED = os.path.join(ROOT, "data_collection", "collected_vacv_dataset.csv")
TOP_CULTURES = 12


def make_lgbm(seed=common.SEED, **kw):
    p = dict(n_estimators=800, learning_rate=0.03, num_leaves=15,
             min_child_samples=8, subsample=0.8, colsample_bytree=0.3,
             reg_lambda=5.0, random_state=seed, verbose=-1)
    p.update(kw)
    return lgb.LGBMRegressor(**p)


def make_cat(seed=common.SEED):
    return CatBoostRegressor(
        iterations=600, learning_rate=0.03, depth=4, l2_leaf_reg=10,
        random_seed=seed, verbose=0, loss_function="RMSE")


def culture_onehot(train_df, test_df):
    """One-hot топ-12 клеточных культур (нормализация регистра/пробелов)."""
    norm = lambda s: str(s).strip().lower()
    top = train_df["Culture"].map(norm).value_counts().head(TOP_CULTURES).index
    def enc(df):
        m = np.zeros((len(df), len(top) + 1), dtype=np.float32)
        for i, c in enumerate(df["Culture"].map(norm)):
            if c in top:
                m[i, list(top).index(c)] = 1.0
            else:
                m[i, -1] = 1.0
        return m
    return enc(train_df), enc(test_df)


def load_augmented_train():
    """Официальный train + внешние SI-данные (без дублей train)."""
    train = common.load_train()
    ext = pd.read_csv(COLLECTED)
    ext = ext[ext["SI"].notna()].copy()
    ext["can_smiles"] = ext["SMILES"].map(common.canonical_smiles)
    ext = ext[~ext["can_smiles"].isin(set(train["can_smiles"]))]
    ext = ext.dropna(subset=["can_smiles"])
    ext = ext.rename(columns={"SI": common.TARGET})
    ext["Culture"] = "unknown"
    ext["SMILES"] = ext["can_smiles"]
    print(f"  внешняя аугментация: +{len(ext)} молекул с SI "
          f"(train {len(train)} -> {len(train) + len(ext)})")
    cols = ["SMILES", "can_smiles", "Culture", common.TARGET]
    return pd.concat([train[cols], ext[cols]], ignore_index=True), len(train)


def main():
    print("== EXP05: ФИНАЛЬНОЕ решение ==")
    train, n_official = load_augmented_train()
    test = common.load_test()
    y = train[common.TARGET].values.astype(float)
    y_log = np.log1p(y)

    print("Признаки...")
    fp_tr = common.morgan_fp(train["SMILES"])
    fp_te = common.morgan_fp(test["SMILES"])
    desc_tr = common.rdkit_descriptors(train["SMILES"])
    desc_te = common.rdkit_descriptors(test["SMILES"])
    emb_tr = get_embeddings(train["SMILES"].tolist(), "train_aug_emb.npy")
    emb_te = get_embeddings(test["SMILES"].tolist(), "test_emb.npy")
    cult_tr, cult_te = culture_onehot(train, test)

    Xa = np.hstack([fp_tr, cult_tr])                    # FP + культура
    Xa_te = np.hstack([fp_te, cult_te])
    Xb = np.hstack([emb_tr, desc_tr, cult_tr])          # эмбеддинги + дескрипторы
    Xb_te = np.hstack([emb_te, desc_te, cult_te])
    Xc = np.hstack([fp_tr, desc_tr, emb_tr, cult_tr])   # полный фьюжн
    Xc_te = np.hstack([fp_te, desc_te, emb_te, cult_te])
    Xd = np.hstack([fp_tr, desc_tr, cult_tr])           # для CatBoost
    Xd_te = np.hstack([fp_te, desc_te, cult_te])

    kf = KFold(n_splits=5, shuffle=True, random_state=common.SEED)
    names = ["lgbm_fp", "lgbm_emb", "twostage", "catboost"]
    oofs = {n: np.zeros(len(y)) for n in names}
    p_inact = np.zeros(len(y))
    tes = {n: np.zeros(len(test)) for n in names}
    p_inact_te = np.zeros(len(test))

    for fold, (tr, va) in enumerate(kf.split(Xa), 1):
        m = make_lgbm(); m.fit(Xa[tr], y_log[tr])
        oofs["lgbm_fp"][va] = common.back_transform(m.predict(Xa[va]))
        tes["lgbm_fp"] += common.back_transform(m.predict(Xa_te)) / 5

        m = make_lgbm(); m.fit(Xb[tr], y_log[tr])
        oofs["lgbm_emb"][va] = common.back_transform(m.predict(Xb[va]))
        tes["lgbm_emb"] += common.back_transform(m.predict(Xb_te)) / 5

        act = y[tr] > 0
        r = make_reg(); r.fit(Xc[tr][act], y_log[tr][act])
        c = make_clf(); c.fit(Xc[tr], (y[tr] == 0).astype(int))
        oofs["twostage"][va] = common.back_transform(r.predict(Xc[va]))
        p_inact[va] = c.predict_proba(Xc[va])[:, 1]
        tes["twostage"] += common.back_transform(r.predict(Xc_te)) / 5
        p_inact_te += c.predict_proba(Xc_te)[:, 1] / 5

        m = make_cat(); m.fit(Xd[tr], y_log[tr])
        oofs["catboost"][va] = common.back_transform(m.predict(Xd[va]))
        tes["catboost"] += common.back_transform(m.predict(Xd_te)) / 5
        print(f"  fold {fold} готов")

    # --- порог двухстадийной модели по её OOF (только официальный train) ---
    off = np.arange(n_official)          # внешние строки — в конце массива
    y_off = y[off]
    best_thr, best_mse = 0.5, np.inf
    for thr in np.arange(0.10, 0.95, 0.05):
        p = np.where(p_inact[off] >= thr, 0.0, oofs["twostage"][off])
        mse = float(np.mean((p - y_off) ** 2))
        if mse < best_mse:
            best_mse, best_thr = mse, thr
    oofs["twostage"] = np.where(p_inact >= best_thr, 0.0, oofs["twostage"])
    tes["twostage"] = np.where(p_inact_te >= best_thr, 0.0, tes["twostage"])

    # ВАЖНО: все метрики и подбор гиперпараметров — только по официальному
    # train (его распределение соответствует leaderboard). Внешние строки
    # участвуют в обучении, но не в валидации.
    for n in names:
        print(f"  {n}: OOF MSE (official) = "
              f"{np.mean((oofs[n][off] - y_off) ** 2):,.1f}")

    # --- веса бленда: NNLS по OOF официального train, сглаживание с uniform ---
    P = np.column_stack([oofs[n] for n in names])
    w_nnls, _ = nnls(P[off], y_off)
    if w_nnls.sum() > 0:
        w_nnls = w_nnls / w_nnls.sum()
    w = 0.5 * w_nnls + 0.5 * np.ones(len(names)) / len(names)
    oof_blend = P @ w
    mse_nnls = float(np.mean((oof_blend[off] - y_off) ** 2))
    mse_uniform = float(np.mean((P[off].mean(axis=1) - y_off) ** 2))
    if mse_nnls > mse_uniform:  # откат к равным весам
        w = np.ones(len(names)) / len(names)
        oof_blend = P.mean(axis=1)
        mse_nnls = mse_uniform
    print(f"  веса бленда: {dict(zip(names, np.round(w, 3)))}")
    print(f"  BLEND: OOF MSE (official) = {mse_nnls:,.1f}")

    # --- оптимальное масштабирование (компенсация смещения Йенсена) ---
    # считаем по официальному train и ограничиваем разумными пределами,
    # чтобы один выброс не разогнал все предсказания
    alpha = float((y_off @ oof_blend[off]) / (oof_blend[off] @ oof_blend[off]))
    alpha = float(np.clip(alpha, 0.8, 2.0))
    mse_scaled = float(np.mean((oof_blend[off] * alpha - y_off) ** 2))
    use_alpha = mse_scaled < mse_nnls
    print(f"  alpha = {alpha:.4f} -> OOF MSE = {mse_scaled:,.1f} "
          f"(применяем: {use_alpha})")

    # --- финальные предсказания test ---
    P_te = np.column_stack([tes[n] for n in names])
    preds = P_te @ w
    if use_alpha:
        preds = preds * alpha
    preds = np.clip(preds, 0.0, None)

    # --- подстановка SI для SMILES, совпадающих с ОФИЦИАЛЬНЫМ train ---
    official_train = common.load_train()
    preds = common.apply_train_overlap(test, preds, official_train)

    sub_path = os.path.join(HERE, "submission.csv")
    common.make_submission(preds, sub_path)

    final_mse = min(mse_nnls, mse_scaled)
    with open(os.path.join(HERE, "metrics.txt"), "w") as f:
        f.write(f"OOF MSE final (raw SI): {final_mse:.2f}\n")
        f.write(f"blend weights: {dict(zip(names, np.round(w, 4)))}\n")
        f.write(f"alpha: {alpha:.4f} (used={use_alpha})\n")
        f.write(f"two-stage threshold: {best_thr:.2f}\n")
        f.write(f"train size (augmented): {len(y)} "
                f"(official {n_official} + external {len(y) - n_official})\n")
    print(f"ИТОГ exp05: OOF MSE = {final_mse:,.1f}")


if __name__ == "__main__":
    main()
