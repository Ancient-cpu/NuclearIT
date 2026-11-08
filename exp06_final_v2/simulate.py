"""Симуляция лидерборда: честная оценка ожидаемого скора exp06.

Идея: 5 раз откладываем 20% официального train как "псевдо-test",
обучаем тот же конвейер (4 модели, log1p-таргет) на оставшихся 80%
+ внешней аугментации, применяем ту же постобработку (alpha=1.6,
zero-clip=1.0, cap=5000) и ту же логику exact match
(псевдо-train + внешний датасет) — и считаем MSE на псевдо-test.

Это воспроизводит условия соревнования: exact-match покрытие в симуляции
естественно ниже, чем на настоящем test (меньше обучающая выборка),
поэтому оценка консервативна (реальный скор, скорее всего, лучше).

Никаких запрещённых колонок, никакого доступа к разметке test.

Запуск:  python experiments/exp06_final_v2/simulate.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import lightgbm as lgb
from catboost import CatBoostRegressor
import common
from experiments.exp06_final_v2.train import (
    meta_features, load_augmented_train, COLLECTED)

ALPHA, ZCLIP, CAP = 1.60, 1.0, 5000.0   # параметры, настроенные в exp06
N_SPLITS = 5


def make_lgbm(seed):
    return lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.03, num_leaves=15,
        min_child_samples=8, subsample=0.8, colsample_bytree=0.3,
        reg_lambda=5.0, random_state=seed, verbose=-1)


def make_cat(seed):
    return CatBoostRegressor(iterations=600, learning_rate=0.03, depth=4,
                             l2_leaf_reg=10, random_seed=seed, verbose=0)


def main():
    use_orthopox = "--orthopox" in sys.argv
    print(f"== СИМУЛЯЦИЯ ЛИДЕРБОРДА (5 сплитов 80/20) "
          f"orthopox-аугментация: {use_orthopox} ==")
    train, n_official = load_augmented_train()
    if use_orthopox:
        op = pd.read_csv(os.path.join(ROOT, "data_collection",
                                      "collected_pubchem_orthopox.csv"))
        op["can_smiles"] = op["SMILES"].map(common.canonical_smiles)
        op = op.dropna(subset=["can_smiles"])
        op = op[~op["can_smiles"].isin(set(train["can_smiles"]))]
        op = op.rename(columns={"SI": common.TARGET})
        op["Culture"] = "unknown"
        op["strain"] = ""
        op["SMILES"] = op["can_smiles"]
        cols = ["SMILES", "can_smiles", "Culture", "strain", common.TARGET]
        train = pd.concat([train[cols], op[cols]], ignore_index=True)
        print(f"  ортопокс-аугментация: +{len(op)} молекул")
    y_all = train[common.TARGET].values.astype(float)

    print("Признаки (эмбеддинги из кэша exp06)...")
    fp = common.morgan_fp(train["SMILES"])
    desc = common.rdkit_descriptors(train["SMILES"])
    from experiments.exp02_chemberta_lgbm.train import get_embeddings
    emb_cache = "train_aug2_op_emb.npy" if use_orthopox else "train_aug2_emb.npy"
    emb = get_embeddings(train["SMILES"].tolist(), emb_cache)
    meta, _ = meta_features(train, train)   # test-часть не нужна

    Xs = {
        "lgbm_fp": np.hstack([fp, meta]),
        "lgbm_emb": np.hstack([emb, desc, meta]),
        "twostage": np.hstack([fp, desc, emb, meta]),
        "catboost": np.hstack([fp, desc, meta]),
    }

    # exact-match словари: внешний датасет + официальный train (по сплиту)
    col = pd.read_csv(COLLECTED)
    col["can"] = col["SMILES"].map(common.canonical_smiles)
    ext_lk = dict(zip(col.loc[col["SI"].notna(), "can"],
                      col.loc[col["SI"].notna(), "SI"]))
    can = train["can_smiles"].values

    off_idx = np.arange(n_official)
    split_data = []   # (y_true, raw_preds, match_values)
    for rep in range(N_SPLITS):
        tr_idx, te_idx = train_test_split(
            off_idx, test_size=0.2, random_state=100 + rep)
        train_lk = {can[i]: y_all[i] for i in tr_idx}

        # обучение: псевдо-train (80% официального) + ВСЯ внешняя аугментация
        fit_idx = np.concatenate([tr_idx, np.arange(n_official, len(y_all))])
        y_fit = np.log1p(y_all[fit_idx])
        act = y_all[fit_idx] > 0

        preds = np.zeros(len(te_idx))
        for n, X in Xs.items():
            if n == "twostage":
                r = make_lgbm(42); r.fit(X[fit_idx][act], y_fit[act])
                preds += common.back_transform(r.predict(X[te_idx])) / len(Xs)
            elif n == "catboost":
                m = make_cat(42); m.fit(X[fit_idx], y_fit)
                preds += common.back_transform(m.predict(X[te_idx])) / len(Xs)
            else:
                m = make_lgbm(42); m.fit(X[fit_idx], y_fit)
                preds += common.back_transform(m.predict(X[te_idx])) / len(Xs)

        # exact match: псевдо-train + внешний датасет
        match_vals = np.full(len(te_idx), np.nan)
        for j, i in enumerate(te_idx):
            smi = can[i]
            if smi in train_lk:
                match_vals[j] = train_lk[smi]
            elif smi in ext_lk:
                match_vals[j] = ext_lk[smi]
        split_data.append((y_all[te_idx], preds, match_vals))
        print(f"  сплит {rep + 1} обучен "
              f"(покрытие {np.isfinite(match_vals).sum()}/{len(te_idx)})")

    def eval_cfg(alpha, zclip, cap):
        mses = []
        for y_true, raw, mv in split_data:
            p = np.clip(raw * alpha, 0, cap)
            p = np.where(p < zclip, 0.0, p)
            m = np.isfinite(mv)
            p[m] = mv[m]
            mses.append(float(np.mean((p - y_true) ** 2)))
        return mses

    if "--grid" in sys.argv:
        print("\n== подбор постобработки по худшему сплиту (minimax) ==")
        results = []
        for alpha in [1.0, 1.2, 1.4, 1.6]:
            for zclip in [0.0, 1.0, 2.0]:
                for cap in [3000.0, 5000.0, 8000.0, None]:
                    ms = eval_cfg(alpha, zclip, cap)
                    results.append((max(ms), np.mean(ms), alpha, zclip, cap))
        results.sort(key=lambda r: (r[0], r[1]))
        for worst, mean, a, z, c in results[:8]:
            print(f"  worst={worst:,.0f} mean={mean:,.0f} "
                  f"alpha={a} zclip={z} cap={c}")
        worst, mean, a, z, c = results[0]
        print(f"ЛУЧШАЯ КОНФИГУРАЦИЯ: alpha={a} zclip={z} cap={c} "
              f"(worst={worst:,.0f}, mean={mean:,.0f})")
        return

    mses = eval_cfg(ALPHA, ZCLIP, CAP)
    coverages = [float(np.isfinite(mv).mean()) for _, _, mv in split_data]
    for rep, m in enumerate(mses, 1):
        print(f"  сплит {rep}: MSE = {m:,.0f}")
    print(f"\nОжидаемый leaderboard MSE: среднее {np.mean(mses):,.0f}, "
          f"медиана {np.median(mses):,.0f}, "
          f"min {np.min(mses):,.0f}, max {np.max(mses):,.0f}")
    print(f"Среднее exact-match покрытие: {np.mean(coverages):.1%} "
          f"(на реальном test оно выше: 63/597 = 10.6%)")
    print(f"Целевой скор 807 115: "
          f"{'ВЕРОЯТНО ПОБИТ' if np.mean(mses) < 807115 else 'ПОД РИСКОМ'}")


if __name__ == "__main__":
    main()
