"""Эксперимент 04 — ансамбль (блендинг) exp01 + exp02 + exp03.

Три разнообразных конвейера усредняются по предсказаниям на исходной шкале:
  1) Morgan FP + LightGBM                      (exp01)
  2) ChemBERTa CLS + LightGBM                  (exp02)
  3) two-stage фьюжн (FP + дескрипторы + CLS)  (exp03)
Блендинг разнородных представлений обычно снижает дисперсию ошибки.

Запуск:  python experiments/exp04_ensemble/train.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
import numpy as np
from sklearn.model_selection import KFold
import common
from experiments.exp01_morgan_lgbm.train import make_model as make_morgan
from experiments.exp02_chemberta_lgbm.train import make_model as make_chemberta, get_embeddings
from experiments.exp03_twostage_fusion.train import build_features, make_reg, make_clf

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    print("== EXP04: ансамбль exp01 + exp02 + exp03 ==")
    train = common.load_train()
    test = common.load_test()
    y = train[common.TARGET].values.astype(float)
    y_log = np.log1p(y)

    print("Признаки...")
    X_fp = common.morgan_fp(train["SMILES"])
    X_fp_te = common.morgan_fp(test["SMILES"])
    X_emb = get_embeddings(train["SMILES"].tolist(), "train_emb.npy")
    X_emb_te = get_embeddings(test["SMILES"].tolist(), "test_emb.npy")
    X_fus = build_features(train, "train_emb.npy")
    X_fus_te = build_features(test, "test_emb.npy")

    kf = KFold(n_splits=5, shuffle=True, random_state=common.SEED)
    oof1 = np.zeros(len(y)); oof2 = np.zeros(len(y))
    oof3r = np.zeros(len(y)); oof3p = np.zeros(len(y))
    te1 = np.zeros(len(test)); te2 = np.zeros(len(test))
    te3r = np.zeros(len(test)); te3p = np.zeros(len(test))

    for fold, (tr, va) in enumerate(kf.split(X_fp), 1):
        m1 = make_morgan(); m1.fit(X_fp[tr], y_log[tr])
        oof1[va] = common.back_transform(m1.predict(X_fp[va]))
        te1 += common.back_transform(m1.predict(X_fp_te)) / 5

        m2 = make_chemberta(); m2.fit(X_emb[tr], y_log[tr])
        oof2[va] = common.back_transform(m2.predict(X_emb[va]))
        te2 += common.back_transform(m2.predict(X_emb_te)) / 5

        act = y[tr] > 0
        r3 = make_reg(); r3.fit(X_fus[tr][act], y_log[tr][act])
        c3 = make_clf(); c3.fit(X_fus[tr], (y[tr] == 0).astype(int))
        oof3r[va] = common.back_transform(r3.predict(X_fus[va]))
        oof3p[va] = c3.predict_proba(X_fus[va])[:, 1]
        te3r += common.back_transform(r3.predict(X_fus_te)) / 5
        te3p += c3.predict_proba(X_fus_te)[:, 1] / 5
        print(f"  fold {fold} готов")

    # порог для двухстадийной части — по её собственному OOF
    best_thr, best_mse3 = 0.5, np.inf
    for thr in np.arange(0.10, 0.95, 0.05):
        p = np.where(oof3p >= thr, 0.0, oof3r)
        mse3 = float(np.mean((p - y) ** 2))
        if mse3 < best_mse3:
            best_mse3, best_thr = mse3, thr
    oof3 = np.where(oof3p >= best_thr, 0.0, oof3r)
    te3 = np.where(te3p >= best_thr, 0.0, te3r)

    for name, oof in [("exp01", oof1), ("exp02", oof2), ("exp03", oof3)]:
        print(f"  {name}: OOF MSE = {np.mean((oof - y) ** 2):,.1f}")

    oof_blend = (oof1 + oof2 + oof3) / 3
    mse_blend = float(np.mean((oof_blend - y) ** 2))
    print(f"  BLEND: OOF MSE = {mse_blend:,.1f}")

    preds = (te1 + te2 + te3) / 3
    preds = common.apply_train_overlap(test, preds, train)
    sub_path = os.path.join(HERE, "submission.csv")
    common.make_submission(preds, sub_path)

    with open(os.path.join(HERE, "metrics.txt"), "w") as f:
        f.write(f"OOF MSE blend (raw SI): {mse_blend:.2f}\n")
        f.write(f"two-stage threshold: {best_thr:.2f}\n")
    print(f"ИТОГ exp04: OOF MSE = {mse_blend:,.1f}")


if __name__ == "__main__":
    main()
