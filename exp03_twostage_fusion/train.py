"""Эксперимент 03 — двухстадийная модель + фьюжн признаков.

Идея: 26% строк train имеют SI = 0 (вещество неактивно, IC50 не измеряли).
Стадия 1: классификатор P(SI == 0) — если неактивно, предсказываем ровно 0.
Стадия 2: регрессор log1p(SI) только по активным веществам.

Признаки: Morgan FP (2048) + RDKit дескрипторы (~210) + ChemBERTa CLS (384).
Порог классификатора подбирается по OOF, минимизируя MSE на исходной шкале.
Дополнительно: поправка Дуана (smearing) на смещение expm1-преобразования
(MSE оптимален для E[SI], а не exp(E[log SI])).

Запуск:  python experiments/exp03_twostage_fusion/train.py
(требует кэш эмбеддингов из exp02 — сначала запустите его,
 либо скрипт скачает ChemBERTa сам)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
import common

HERE = os.path.dirname(os.path.abspath(__file__))
EXP02_CACHE = os.path.join(HERE, "..", "exp02_chemberta_lgbm", "cache")


def build_features(df, emb_file):
    X_fp = common.morgan_fp(df["SMILES"])
    X_desc = common.rdkit_descriptors(df["SMILES"])
    emb_path = os.path.join(EXP02_CACHE, emb_file)
    if os.path.exists(emb_path):
        X_emb = np.load(emb_path)
    else:  # считаем сами, если exp02 не запускался
        sys.path.insert(0, os.path.join(HERE, "..", "exp02_chemberta_lgbm"))
        from train import get_embeddings
        X_emb = get_embeddings(df["SMILES"].tolist(), emb_file)
    return np.hstack([X_fp, X_desc, X_emb])


def make_reg():
    return lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.03, num_leaves=15,
        min_child_samples=8, subsample=0.8, colsample_bytree=0.3,
        reg_lambda=5.0, random_state=common.SEED, verbose=-1,
    )


def make_clf():
    return lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.03, num_leaves=15,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.3,
        reg_lambda=5.0, random_state=common.SEED, verbose=-1,
    )


def main():
    print("== EXP03: two-stage (классификатор активности + регрессор) + фьюжн признаков ==")
    train = common.load_train()
    test = common.load_test()

    print("Признаки train...")
    X = build_features(train, "train_emb.npy")
    print("Признаки test...")
    X_test = build_features(test, "test_emb.npy")

    y = train[common.TARGET].values.astype(float)
    y_bin = (y == 0).astype(int)          # 1 = неактивно
    y_log = np.log1p(y)

    kf = KFold(n_splits=5, shuffle=True, random_state=common.SEED)
    oof_reg = np.zeros(len(y))            # предсказание регрессора (raw) для всех
    oof_p_inactive = np.zeros(len(y))     # вероятность неактивности
    regs, clfs = [], []

    for fold, (tr, va) in enumerate(kf.split(X), 1):
        # --- регрессор: только активные вещества ---
        act = y[tr] > 0
        reg = make_reg()
        reg.fit(X[tr][act], y_log[tr][act])
        oof_reg[va] = common.back_transform(reg.predict(X[va]))
        # --- классификатор неактивности ---
        clf = make_clf()
        clf.fit(X[tr], y_bin[tr])
        oof_p_inactive[va] = clf.predict_proba(X[va])[:, 1]
        regs.append(reg)
        clfs.append(clf)
        print(f"  fold {fold} готов")

    # --- подбор порога по OOF ---
    best_thr, best_mse = 0.5, np.inf
    for thr in np.arange(0.10, 0.95, 0.05):
        pred = np.where(oof_p_inactive >= thr, 0.0, oof_reg)
        mse = float(np.mean((pred - y) ** 2))
        if mse < best_mse:
            best_mse, best_thr = mse, thr
    print(f"  лучший порог P(inactive) = {best_thr:.2f}, OOF MSE = {best_mse:,.1f}")

    # --- smearing-поправка (фактор на OOF активных) ---
    act_mask = oof_p_inactive < best_thr
    resid = y[act_mask] / np.maximum(oof_reg[act_mask], 1e-6)
    smear = float(np.mean(resid))
    pred_smeared = np.where(oof_p_inactive >= best_thr, 0.0, oof_reg * smear)
    mse_smeared = float(np.mean((pred_smeared - y) ** 2))
    print(f"  smearing-фактор = {smear:.3f}, OOF MSE (smearing) = {mse_smeared:,.1f}")
    use_smear = mse_smeared < best_mse
    print(f"  применяем smearing: {use_smear}")

    # --- финальное предсказание test ---
    pred_test_reg = np.mean(
        [common.back_transform(r.predict(X_test)) for r in regs], axis=0)
    p_inactive_test = np.mean(
        [c.predict_proba(X_test)[:, 1] for c in clfs], axis=0)
    preds = np.where(p_inactive_test >= best_thr, 0.0, pred_test_reg)
    if use_smear:
        preds = np.where(p_inactive_test >= best_thr, 0.0, pred_test_reg * smear)

    preds = common.apply_train_overlap(test, preds, train)
    sub_path = os.path.join(HERE, "submission.csv")
    common.make_submission(preds, sub_path)

    final_mse = min(best_mse, mse_smeared)
    with open(os.path.join(HERE, "metrics.txt"), "w") as f:
        f.write(f"OOF MSE (raw SI): {final_mse:.2f}\n")
        f.write(f"threshold: {best_thr:.2f}\nsmearing: {smear:.3f} (used={use_smear})\n")
    print(f"ИТОГ exp03: OOF MSE = {final_mse:,.1f}")


if __name__ == "__main__":
    main()
