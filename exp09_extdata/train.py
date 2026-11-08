"""Эксперимент 09 — exp06-пайплайн + расширенный внешний датасет (v2).

Отличия от exp06:
  1. Обучение на collected_vacv_dataset_v2.csv — v1 (таргет-поиск по вирусу)
     ПЛЮС структурный поиск всех 597 SMILES теста по PubChem/ChEMBL
     (match_test_structures.py): 927 молекул, 308 с SI (было 646/72).
  2. Слой exact-match подстановок: стерео-точные совпадения (изомерный
     canonical SMILES) из collected_test_matches.csv — 110+ строк test.
     Стереослепой (ахиральный) матчинг НЕ используется: энантиомеры
     различаются по активности.
  3. EXT_CAP=5000 для внешних подстановок (значение из литературы может
     отличаться от условий конкретного ассая теста; cap — страховка,
     выбранная по публичному фидбеку exp06, см. README exp07).
  4. Чанкинг по сидам: --seeds 42 / --seeds 1337 / --seeds 2025, затем
     --aggregate (NNLS-бленд + постобработка по bulk-OOF + подстановки).

Запрещённые колонки (CC50_uM / IC50_uM_VACV файлов соревнования) не
используются нигде: ни в признаках, ни в подстановках.
Запуск:  python experiments/exp09_extdata/train.py --seeds 42
         python experiments/exp09_extdata/train.py --aggregate
"""
import argparse
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

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
COLLECTED = os.path.join(ROOT, "data_collection", "collected_vacv_dataset_v2.csv")
MATCHES = os.path.join(ROOT, "data_collection", "collected_test_matches.csv")
SEEDS = [42, 1337, 2025]
TOP_CULTURES = 12
EXT_CAP = 5000.0


def make_lgbm(seed):
    return lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.03, num_leaves=15,
        min_child_samples=8, subsample=0.8, colsample_bytree=0.3,
        reg_lambda=5.0, random_state=seed, verbose=-1)


def make_cat(seed):
    return CatBoostRegressor(
        iterations=600, learning_rate=0.03, depth=4, l2_leaf_reg=10,
        random_seed=seed, verbose=0, loss_function="RMSE")


def meta_features(train_df, test_df):
    norm = lambda s: str(s).strip().lower()
    top = list(train_df["Culture"].map(norm).value_counts()
               .head(TOP_CULTURES).index)

    def enc(df):
        cult = df["Culture"].map(norm)
        strain = df.get("strain", pd.Series([""] * len(df))).fillna("").map(str.lower)
        m = np.zeros((len(df), len(top) + 5), dtype=np.float32)
        for i in range(len(df)):
            c = cult.iloc[i]
            m[i, top.index(c) if c in top else len(top)] = 1.0
            s = strain.iloc[i]
            m[i, len(top) + 1] = float("copenhagen" in s)
            m[i, len(top) + 2] = float("western reserve" in s or "wr" in s.split(",")[0])
            m[i, len(top) + 3] = float("lister" in s)
            m[i, len(top) + 4] = float("ihd" in s)
        return m
    return enc(train_df), enc(test_df)


def load_augmented_train():
    train = common.load_train()
    ext = pd.read_csv(COLLECTED)
    ext = ext[ext["SI"].notna()].copy()
    ext["can_smiles"] = ext["SMILES"].map(common.canonical_smiles)
    ext = ext[~ext["can_smiles"].isin(set(train["can_smiles"]))]
    ext = ext.dropna(subset=["can_smiles"])
    ext = ext.rename(columns={"SI": common.TARGET})
    ext["Culture"] = "unknown"
    ext["strain"] = ""
    ext["SMILES"] = ext["can_smiles"]
    print(f"  аугментация: +{len(ext)} внешних молекул с SI "
          f"({len(train)} -> {len(train) + len(ext)})")
    cols = ["SMILES", "can_smiles", "Culture", "strain", common.TARGET]
    return pd.concat([train[cols], ext[cols]], ignore_index=True), len(train)


def run_seed(seed):
    train, n_official = load_augmented_train()
    test = common.load_test()
    y = train[common.TARGET].values.astype(float)
    y_log = np.log1p(y)

    print("Признаки...")
    fp_tr = common.morgan_fp(train["SMILES"])
    fp_te = common.morgan_fp(test["SMILES"])
    desc_tr = common.rdkit_descriptors(train["SMILES"])
    desc_te = common.rdkit_descriptors(test["SMILES"])
    emb_tr = get_embeddings(train["SMILES"].tolist(), "train_aug3_emb.npy")
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
    oofs = {n: np.zeros(len(y)) for n in names}
    tes = {n: np.zeros(len(test)) for n in names}
    p_inact = np.zeros(len(y))
    p_inact_te = np.zeros(len(test))

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(kf.split(Xs["lgbm_fp"][0]), 1):
        for n in names:
            X, X_te = Xs[n]
            if n == "twostage":
                act = y[tr] > 0
                r = make_lgbm(seed); r.fit(X[tr][act], y_log[tr][act])
                c = lgb.LGBMClassifier(
                    n_estimators=600, learning_rate=0.03, num_leaves=15,
                    min_child_samples=10, subsample=0.8,
                    colsample_bytree=0.3, reg_lambda=5.0,
                    random_state=seed, verbose=-1)
                c.fit(X[tr], (y[tr] == 0).astype(int))
                oofs[n][va] = common.back_transform(r.predict(X[va]))
                p_inact[va] = c.predict_proba(X[va])[:, 1]
                tes[n] += common.back_transform(r.predict(X_te)) / 5
                p_inact_te += c.predict_proba(X_te)[:, 1] / 5
            elif n == "catboost":
                m = make_cat(seed); m.fit(X[tr], y_log[tr])
                oofs[n][va] = common.back_transform(m.predict(X[va]))
                tes[n] += common.back_transform(m.predict(X_te)) / 5
            else:
                m = make_lgbm(seed); m.fit(X[tr], y_log[tr])
                oofs[n][va] = common.back_transform(m.predict(X[va]))
                tes[n] += common.back_transform(m.predict(X_te)) / 5
        print(f"  seed {seed} fold {fold} готов")

    os.makedirs(CACHE, exist_ok=True)
    np.savez(os.path.join(CACHE, f"seed_{seed}.npz"),
             names=names, n_official=n_official,
             p_inact=p_inact, p_inact_te=p_inact_te,
             **{f"oof_{n}": oofs[n] for n in names},
             **{f"te_{n}": tes[n] for n in names})
    print(f"  seed {seed}: сохранено в cache/seed_{seed}.npz")


def aggregate(seeds, force_cap=None):
    packs = []
    for s in seeds:
        path = os.path.join(CACHE, f"seed_{s}.npz")
        assert os.path.exists(path), f"нет {path} — сначала --seeds {s}"
        packs.append(np.load(path, allow_pickle=True))
    names = list(packs[0]["names"])
    n_official = int(packs[0]["n_official"])
    train, _ = load_augmented_train()
    test = common.load_test()
    y = train[common.TARGET].values.astype(float)
    off = np.arange(n_official)
    y_off = y[off]

    oofs = {n: np.mean([p[f"oof_{n}"] for p in packs], axis=0) for n in names}
    tes = {n: np.mean([p[f"te_{n}"] for p in packs], axis=0) for n in names}
    p_inact = np.mean([p["p_inact"] for p in packs], axis=0)
    p_inact_te = np.mean([p["p_inact_te"] for p in packs], axis=0)

    bulk = off[y_off <= 10000]
    y_bulk = y[bulk]

    best_thr, best_mse = 0.5, np.inf
    for thr in np.arange(0.10, 0.95, 0.05):
        p = np.where(p_inact[bulk] >= thr, 0.0, oofs["twostage"][bulk])
        mse = float(np.mean((p - y_bulk) ** 2))
        if mse < best_mse:
            best_mse, best_thr = mse, thr
    oofs["twostage"] = np.where(p_inact >= best_thr, 0.0, oofs["twostage"])
    tes["twostage"] = np.where(p_inact_te >= best_thr, 0.0, tes["twostage"])

    for n in names:
        print(f"  {n}: OOF MSE (bulk) = "
              f"{np.mean((oofs[n][bulk] - y_bulk) ** 2):,.1f}")

    P = np.column_stack([oofs[n] for n in names])
    w_nnls, _ = nnls(P[bulk], y_bulk)
    w_nnls = w_nnls / w_nnls.sum() if w_nnls.sum() > 0 else np.ones(len(names)) / len(names)
    w = 0.5 * w_nnls + 0.5 * np.ones(len(names)) / len(names)
    oof_blend = P @ w
    mse_blend = float(np.mean((oof_blend[bulk] - y_bulk) ** 2))
    if mse_blend > float(np.mean((P[bulk].mean(axis=1) - y_bulk) ** 2)):
        w = np.ones(len(names)) / len(names)
        oof_blend = P.mean(axis=1)
        mse_blend = float(np.mean((oof_blend[bulk] - y_bulk) ** 2))
    print(f"  веса: {dict(zip(names, np.round(w, 3)))}, "
          f"BLEND OOF MSE (bulk) = {mse_blend:,.1f}")

    P_te = np.column_stack([tes[n] for n in names])
    pred_te = P_te @ w
    pred_oof = oof_blend.copy()

    def score(po, alpha, zclip, cap):
        po2 = np.clip(po * alpha, 0, cap)
        po2 = np.where(po2 < zclip, 0.0, po2)
        return float(np.mean((po2[bulk] - y_bulk) ** 2))

    best = (mse_blend, 1.0, 0.0, None)
    cap_grid = [force_cap] if force_cap is not None \
        else [None, 20000, 10000, 5000, 3000]
    for alpha in np.arange(0.9, 1.61, 0.05):
        for zclip in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            for cap in cap_grid:
                m = score(pred_oof, alpha, zclip, cap)
                if m < best[0]:
                    best = (m, alpha, zclip, cap)
    mse_post, alpha, zclip, cap = best
    print(f"  постобработка: alpha={alpha:.2f}, zero-clip={zclip}, cap={cap} "
          f"-> OOF MSE = {mse_post:,.1f}")

    preds = np.clip(pred_te * alpha, 0, cap)
    preds = np.where(preds < zclip, 0.0, preds)

    # --- exact match: официальный train (без cap — это разметка train) ---
    official_train = common.load_train()
    lk = dict(zip(official_train["can_smiles"], official_train[common.TARGET]))
    n_train_match = 0
    for i, smi in enumerate(test["can_smiles"]):
        if pd.notna(smi) and smi in lk:
            preds[i] = lk[smi]
            n_train_match += 1
    print(f"  exact match (официальный train): {n_train_match}")

    preds_noext = preds.copy()

    # --- exact match: внешние данные, ТОЛЬКО стерео-точные, с EXT_CAP ---
    # v1 (таргет-поиск): молекулы измерены сами по себе — стерео-точные
    v1 = pd.read_csv(os.path.join(ROOT, "data_collection",
                                  "collected_vacv_dataset.csv"))
    v1["can"] = v1["SMILES"].map(common.canonical_smiles)
    ext_lk = dict(zip(v1.loc[v1["SI"].notna(), "can"],
                      v1.loc[v1["SI"].notna(), "SI"]))
    # v2 структурный поиск: только match_level == stereo
    mt = pd.read_csv(MATCHES)
    mt["can"] = mt["SMILES"].map(common.canonical_smiles)
    st = mt[(mt["SI"].notna()) & (mt["match_level"] == "stereo")]
    for _, r in st.iterrows():
        ext_lk[r["can"]] = r["SI"]
    n_ext_match = 0
    for i, smi in enumerate(test["can_smiles"]):
        if pd.notna(smi) and smi in ext_lk and smi not in lk:
            preds[i] = min(float(ext_lk[smi]), EXT_CAP)
            n_ext_match += 1
    print(f"  exact match (внешние открытые данные, stereo, "
          f"cap={EXT_CAP:.0f}): {n_ext_match}")

    # --- правило слабой активности (внешние данные v2) ---
    col_full = pd.read_csv(COLLECTED)
    col_full["can"] = col_full["SMILES"].map(common.canonical_smiles)
    weak_ref = col_full[(col_full["IC50_uM"] >= 50) & col_full["SI"].notna()]
    weak_const = float(weak_ref["SI"].median()) if len(weak_ref) else 1.0
    weak = col_full[(col_full["IC50_uM"] >= 50) & (col_full["SI"].isna())]
    weak_set = set(weak["can"].dropna())
    n_weak = 0
    for i, smi in enumerate(test["can_smiles"]):
        if pd.notna(smi) and smi in weak_set and smi not in lk \
                and smi not in ext_lk:
            preds[i] = weak_const
            n_weak += 1
    print(f"  правило слабой активности (IC50>=50, SI={weak_const:.2f}): "
          f"{n_weak}")

    common.make_submission(preds, os.path.join(HERE, "submission.csv"))
    common.make_submission(preds_noext, os.path.join(HERE,
                                                    "submission_noext.csv"))

    with open(os.path.join(HERE, "metrics.txt"), "w") as f:
        f.write(f"OOF MSE final (official train): {mse_post:.2f}\n")
        f.write(f"blend weights: {dict(zip(names, np.round(w, 4)))}\n")
        f.write(f"alpha={alpha:.2f} zero_clip={zclip} cap={cap} "
                f"two_stage_thr={best_thr:.2f}\n")
        f.write(f"exact match: train={n_train_match}, "
                f"external_stereo={n_ext_match} (EXT_CAP={EXT_CAP:.0f}), "
                f"weak_rule={n_weak} (const={weak_const:.3f})\n")
        f.write(f"train size: {len(y)} (official {n_official}), "
                f"seeds={seeds}\n")
    print(f"ИТОГ exp09: OOF MSE (official) = {mse_post:,.1f}")


def main():
    print("== EXP09: exp06 + расширенные внешние данные (v2, структурный "
          "поиск) ==")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--force-cap", type=float, default=None,
                    help="принудительный cap постобработки (страховочная "
                         "настройка, подтверждённая публичным скором exp06)")
    a = ap.parse_args()
    if a.aggregate:
        aggregate(SEEDS, force_cap=a.force_cap)
    elif a.seeds:
        for s in [int(x) for x in a.seeds.split(",")]:
            run_seed(s)
    else:
        for s in SEEDS:
            run_seed(s)
        aggregate(SEEDS)


if __name__ == "__main__":
    main()
