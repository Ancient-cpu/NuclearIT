"""Эксперимент 01 — бейзлайн: Morgan fingerprints (ECFP4, 2048 бит) + LightGBM.

По мотивам бенчмарка arXiv:2508.06199: на малых датасетах биоактивности
классические ECFP-фингерпринты сопоставимы или лучше предобученных эмбеддингов.

Таргет: log1p(SI) при обучении; метрика MSE считается на исходной шкале (как на Kaggle).
Запуск:  python experiments/exp01_morgan_lgbm/train.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np
import lightgbm as lgb
import common

HERE = os.path.dirname(os.path.abspath(__file__))


def make_model():
    return lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=15,           # маленькие деревья — данных всего 320 строк
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.4,
        reg_lambda=5.0,
        random_state=common.SEED,
        verbose=-1,
    )


def main():
    print("== EXP01: Morgan FP + LightGBM ==")
    train = common.load_train()
    test = common.load_test()

    print("Фингерпринты train/test...")
    X = common.morgan_fp(train["SMILES"])
    X_test = common.morgan_fp(test["SMILES"])
    y = train[common.TARGET]

    print("Кросс-валидация (5-fold):")
    oof, mse, models = common.run_cv(make_model, X, y)

    print("Предсказание test...")
    preds = common.predict_test(models, X_test)
    preds = common.apply_train_overlap(test, preds, train)

    sub_path = os.path.join(HERE, "submission.csv")
    common.make_submission(preds, sub_path)

    with open(os.path.join(HERE, "metrics.txt"), "w") as f:
        f.write(f"OOF MSE (raw SI): {mse:.2f}\n")
    print(f"ИТОГ exp01: OOF MSE = {mse:,.1f}")


if __name__ == "__main__":
    main()
