"""Эксперимент 02 — предобученные эмбеддинги ChemBERTa-2 (77M-MTR) + LightGBM.

Эмбеддинги: CLS-токен из DeepChem/ChemBERTa-77M-MTR (Hugging Face),
модель предобучена мультизадачной регрессией на ~200 свойствах RDKit
для 77 млн молекул PubChem (Ahmad et al., 2022, arXiv:2209.01712).

Эмбеддинги кэшируются в experiments/exp02_chemberta_lgbm/cache/,
чтобы не пересчитывать при повторных запусках.
Запуск:  python experiments/exp02_chemberta_lgbm/train.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np
import lightgbm as lgb
import common

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"


def get_embeddings(smiles_list, cache_file):
    """CLS-эмбеддинги ChemBERTa; кэширует результат в .npy."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, cache_file)
    if os.path.exists(path):
        print(f"  эмбеддинги из кэша: {path}")
        return np.load(path)

    import torch
    from transformers import AutoTokenizer, AutoModel

    print(f"  загружаю {MODEL_NAME} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    embs = np.zeros((len(smiles_list), model.config.hidden_size), dtype=np.float32)
    bs = 32
    with torch.no_grad():
        for start in range(0, len(smiles_list), bs):
            batch = [str(s) for s in smiles_list[start:start + bs]]
            enc = tok(batch, padding=True, truncation=True, max_length=256,
                      return_tensors="pt")
            out = model(**enc).last_hidden_state  # (B, T, H)
            # mean-pooling по маске устойчивее CLS на очень коротких SMILES,
            # но для сравнения с литературой берём CLS:
            embs[start:start + len(batch)] = out[:, 0, :].cpu().numpy()
    np.save(path, embs)
    return embs


def make_model():
    return lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.4,
        reg_lambda=5.0,
        random_state=common.SEED,
        verbose=-1,
    )


def main():
    print("== EXP02: ChemBERTa-2 (77M-MTR) embeddings + LightGBM ==")
    train = common.load_train()
    test = common.load_test()

    print("Эмбеддинги train...")
    X = get_embeddings(train["SMILES"].tolist(), "train_emb.npy")
    print("Эмбеддинги test...")
    X_test = get_embeddings(test["SMILES"].tolist(), "test_emb.npy")
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
    print(f"ИТОГ exp02: OOF MSE = {mse:,.1f}")


if __name__ == "__main__":
    main()
