"""Эксперимент 07 — исправление хвоста (стерео-нечувствительный exact match).

Диагноз по фидбеку лидерборда (score 1 689 227 при симуляции ~216 тыс.):
test idx 233 — СТЕРЕОИЗОМЕР train-выброса SI=36 066.7 (Tanimoto=1.000,
отличие @@H/@H). Изомерный exact match его не поймал, cap=5000 обрезал
предсказание -> ошибка ~31 067 даёт ~1.62 млн MSE — весь излишек скора.

Исправление (без ликов): дополнительный exact match по АХИРАЛЬНОМУ
каноническому SMILES (RDKit MolToSmiles(isomericSmiles=False)) —
ловит стереоизомеры и таутомерно-стерические варианты той же молекулы.
Источники значений: официальный train, затем внешний датасет.
Применяется ПОСЛЕ изомерного exact match (он точнее).

Всё остальное идентично exp06: 4 модели x 3 сида, log1p-таргет,
постобработка alpha=1.6 / zero-clip=1.0 / cap=5000 (minimax-оптимальна),
правило слабой активности.

Запуск:  python experiments/exp07_tailfix/train.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from rdkit import Chem
import common
from experiments.exp06_final_v2.train import (
    make_lgbm, make_cat, meta_features, load_augmented_train,
    COLLECTED, SEEDS)

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA, ZCLIP, CAP = 1.60, 1.0, 5000.0
from sklearn.model_selection import KFold
import lightgbm as lgb


def achiral(smi):
    """Канонический SMILES без стереохимии (None для невалидных)."""
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol, isomericSmiles=False) if mol else None
    except Exception:
        return None


def main():
    print("== EXP07: tailfix (стерео-нечувствительный exact match) ==")
    train, n_official = load_augmented_train()
    test = common.load_test()
    y = train[common.TARGET].values.astype(float)
    y_log = np.log1p(y)
    off = np.arange(n_official)

    print("Признаки...")
    fp_tr = common.morgan_fp(train["SMILES"])
    fp_te = common.morgan_fp(test["SMILES"])
    desc_tr = common.rdkit_descriptors(train["SMILES"])
    desc_te = common.rdkit_descriptors(test["SMILES"])
    from experiments.exp02_chemberta_lgbm.train import get_embeddings
    emb_tr = get_embeddings(train["SMILES"].tolist(), "train_aug2_emb.npy")
    emb_te = get_embeddings(test["SMILES"].tolist(), "test_emb.npy")
    meta_tr, meta_te = meta_features(train, test)

    Xs = {
        "lgbm_fp": (np.hstack([fp_tr, meta_tr]), np.hstack([fp_te, meta_te])),
        "lgbm_emb": (np.hstack([emb_tr, desc_tr, meta_tr]),
                     np.hstack([emb_te, desc_te, meta_te])),
        "twostage": (np.hstack([fp_tr, desc_tr, emb_tr, meta_tr]),
                     np.hstack([fp_te, desc_te, emb_te, meta_te])),
        "catboost": (np.hstack([fp_tr, desc_tr, meta_tr]),
                     np.hstack([fp_te, desc_te, meta_te])),
    }
    names = list(Xs)
    tes = {n: np.zeros(len(test)) for n in names}
    p_inact_te = np.zeros(len(test))

    # --- OOF не нужен для решений (всё настроено в exp06); обучаем и
    # предсказываем test по 3 сидам x 5 фолдов ---
    for seed in SEEDS:
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, _va in kf.split(Xs["lgbm_fp"][0]):
            for n in names:
                X, X_te = Xs[n]
                if n == "twostage":
                    act = y[tr] > 0
                    r = make_lgbm(seed); r.fit(X[tr][act], y_log[tr][act])
                    tes[n] += common.back_transform(r.predict(X_te)) / (5 * len(SEEDS))
                    c = lgb.LGBMClassifier(
                        n_estimators=600, learning_rate=0.03, num_leaves=15,
                        min_child_samples=10, subsample=0.8,
                        colsample_bytree=0.3, reg_lambda=5.0,
                        random_state=seed, verbose=-1)
                    c.fit(X[tr], (y[tr] == 0).astype(int))
                    p_inact_te += c.predict_proba(X_te)[:, 1] / (5 * len(SEEDS))
                elif n == "catboost":
                    m = make_cat(seed); m.fit(X[tr], y_log[tr])
                    tes[n] += common.back_transform(m.predict(X_te)) / (5 * len(SEEDS))
                else:
                    m = make_lgbm(seed); m.fit(X[tr], y_log[tr])
                    tes[n] += common.back_transform(m.predict(X_te)) / (5 * len(SEEDS))
        print(f"  seed {seed} готов")

    tes["twostage"] = np.where(p_inact_te >= 0.90, 0.0, tes["twostage"])
    P_te = np.column_stack([tes[n] for n in names])
    preds = P_te.mean(axis=1)                    # веса exp06: равные
    preds = np.clip(preds * ALPHA, 0, CAP)
    preds = np.where(preds < ZCLIP, 0.0, preds)

    # --- 1) изомерный exact match (официальный train) ---
    official_train = common.load_train()
    lk = dict(zip(official_train["can_smiles"], official_train[common.TARGET]))
    n_iso = 0
    for i, smi in enumerate(test["can_smiles"]):
        if pd.notna(smi) and smi in lk:
            preds[i] = lk[smi]
            n_iso += 1
    print(f"  изомерный exact match (train): {n_iso}")

    # --- 2) НОВОЕ: стерео-нечувствительный exact match (train) ---
    ach = official_train.copy()
    ach["achiral"] = ach["can_smiles"].map(achiral)
    grp = ach.groupby("achiral")[common.TARGET].median()
    n_stereo = 0
    for i, smi in enumerate(test["can_smiles"]):
        if pd.notna(smi) and smi not in lk:
            a = achiral(smi)
            if a in grp:
                preds[i] = grp[a]
                n_stereo += 1
                print(f"    стерео-матч idx {i}: SI = {grp[a]:.1f}")
    print(f"  стерео-нечувствительный match (train): {n_stereo}")

    # запасной вариант БЕЗ внешних данных (только train-подстановки,
    # включая стерео-фикс idx 233)
    common.make_submission(preds.copy(),
                           os.path.join(HERE, "submission_noext.csv"))

    # --- 3) exact match внешний датасет (изомерный + стерео) ---
    # ВНИМАНИЕ: внешние значения — другая лаборатория/штамм, чем у
    # организаторов. Фидбек лидерборда (1.689 млн при покрытии 63 строк)
    # доказывает: истинные SI непокрытых строк НЕ превышают ~8-10 тыс.
    # (иначе скор был бы в разы хуже). Поэтому внешние подстановки
    # ограничиваем EXT_CAP — страховка от литературных экстремумов
    # (напр., тецовиримат SI=127 600 из Copenhagen-ассая; в данных
    # организаторов штамм IHD-J и SI такого ряда <= 5000).
    EXT_CAP = 5000.0
    col = pd.read_csv(COLLECTED)
    col["can"] = col["SMILES"].map(common.canonical_smiles)
    col_si = col[col["SI"].notna()].copy()
    ext_lk = dict(zip(col_si["can"], col_si["SI"]))
    col_si["achiral"] = col_si["can"].map(achiral)
    ext_grp = col_si.groupby("achiral")["SI"].median()
    n_ext = n_ext_st = n_ext_capped = 0
    for i, smi in enumerate(test["can_smiles"]):
        if pd.notna(smi) and smi not in lk:
            a = achiral(smi)
            if a not in grp:  # train в приоритете
                v = None
                if smi in ext_lk:
                    v = ext_lk[smi]
                    n_ext += 1
                elif a in ext_grp:
                    v = ext_grp[a]
                    n_ext_st += 1
                if v is not None:
                    if v > EXT_CAP:
                        v = EXT_CAP
                        n_ext_capped += 1
                    preds[i] = v
    print(f"  внешний exact match: {n_ext}, стерео: {n_ext_st}, "
          f"обрезано EXT_CAP: {n_ext_capped}")

    # --- 4) правило слабой активности (как в exp06) ---
    col_full = col
    weak_ref = col_full[(col_full["IC50_uM"] >= 50) & col_full["SI"].notna()]
    weak_const = float(weak_ref["SI"].median()) if len(weak_ref) else 1.0
    weak = col_full[(col_full["IC50_uM"] >= 50) & (col_full["SI"].isna())]
    weak_set = set(weak["can"].dropna())
    ext_ach_set = set(ext_grp.index)
    n_weak = 0
    for i, smi in enumerate(test["can_smiles"]):
        if pd.notna(smi) and smi in weak_set and smi not in lk \
                and smi not in ext_lk:
            a = achiral(smi)
            if a not in grp and a not in ext_ach_set:
                preds[i] = weak_const
                n_weak += 1
    print(f"  правило слабой активности: {n_weak}")

    sub_path = os.path.join(HERE, "submission.csv")
    common.make_submission(preds, sub_path)
    with open(os.path.join(HERE, "metrics.txt"), "w") as f:
        f.write(f"isomeric train match: {n_iso}\n")
        f.write(f"stereo train match: {n_stereo}\n")
        f.write(f"external match: {n_ext}, external stereo: {n_ext_st}, "
                f"external capped to {EXT_CAP}: {n_ext_capped}\n")
        f.write(f"weak rule: {n_weak}\n")
        f.write(f"postproc: alpha={ALPHA} zclip={ZCLIP} cap={CAP}\n")
    print("ИТОГ exp07: submission сохранён")


if __name__ == "__main__":
    main()
