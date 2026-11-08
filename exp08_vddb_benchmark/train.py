"""Эксперимент 08 — воспроизведение бенчмарка VDDB (Tao et al., 2023,
MedComm – Future Medicine, DOI:10.1002/mef2.32) на нашей задаче.

В статье VDDB сравнивались 6 методов для предсказания противовирусной
активности (у них — классификация active/inactive, у нас — регрессия SI):
  1. RF      — Random Forest на ECFP4 (Morgan fingerprint)
  2. XGBoost — градиентный бустинг на ECFP4
  3. DNN     — полносвязная сеть на ECFP4
  4. GCN     — графовая свёрточная сеть (Duvino et al., DeepChem-стиль)
  5. GAT     — графовая attention-сеть
  6. FP-GNN  — гибрид графовой сети и фингерпринтов (idrugLab)

Адаптация под нашу задачу: регрессия log1p(SI), метрика MSE на исходной
шкале (как на Kaggle). Гиперпараметры — из диапазонов статьи (раздел 4.5).
Графовые модели реализованы на чистом PyTorch (RDKit-графы, батчинг
блочно-диагональной матрицей смежности) — без зависимости от DeepChem.

Валидация: 5-fold CV, обучение на аугментированном train (381 пример),
метрика на официальном train (320 строк): полная и "bulk" (без выброса
SI>10 000) — как в exp05/06 для сопоставимости.

Запуск:  python experiments/exp08_vddb_benchmark/train.py
"""
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import torch
import torch.nn as nn
from rdkit import Chem
import common

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cpu"

# ----------------------------- признаки ---------------------------------

def atom_features(mol):
    feats = []
    for a in mol.GetAtoms():
        feats.append([
            a.GetAtomicNum() / 54.0,
            a.GetTotalDegree() / 6.0,
            a.GetFormalCharge(),
            a.GetTotalNumHs() / 4.0,
            float(a.GetHybridization()) / 8.0,
            float(a.GetIsAromatic()),
            float(a.IsInRing()),
            float(int(a.GetChiralTag())),
        ])
    return np.array(feats, dtype=np.float32)


def build_graphs(smiles_list):
    """Список (X, A) для каждой молекулы."""
    graphs = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None or mol.GetNumAtoms() == 0:
            graphs.append((np.zeros((1, 8), np.float32),
                           np.zeros((1, 1), np.float32)))
            continue
        X = atom_features(mol)
        A = np.zeros((len(X), len(X)), dtype=np.float32)
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            A[i, j] = A[j, i] = 1.0
        graphs.append((X, A))
    return graphs


def collate(graphs, idx):
    """Блочно-диагональный батч графов."""
    xs, sizes = [], []
    N = sum(graphs[i][0].shape[0] for i in idx)
    A = np.zeros((N, N), dtype=np.float32)
    off = 0
    for gi, i in enumerate(idx):
        X, Ai = graphs[i]
        n = X.shape[0]
        xs.append(X)
        A[off:off + n, off:off + n] = Ai
        sizes.extend([gi] * n)
        off += n
    return (torch.tensor(np.vstack(xs)), torch.tensor(A),
            torch.tensor(sizes))


# ----------------------------- DL-модели --------------------------------

class MLP(nn.Module):
    """DNN статьи: полносвязная сеть на ECFP4."""
    def __init__(self, in_dim=2048, hidden=(512, 256), dropout=0.2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class GCNLayer(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.lin = nn.Linear(din, dout)

    def forward(self, X, A):
        N = A.shape[0]
        Ah = A + torch.eye(N)
        D = Ah.sum(1).clamp(min=1).pow(-0.5)
        An = Ah * D[:, None] * D[None, :]
        return torch.relu(An @ self.lin(X))


class GCN(nn.Module):
    def __init__(self, in_dim=8, conv=(128, 128), dense=128):
        super().__init__()
        self.convs = nn.ModuleList(
            [GCNLayer(in_dim, conv[0])] +
            [GCNLayer(conv[i - 1], conv[i]) for i in range(1, len(conv))])
        self.head = nn.Sequential(nn.Linear(conv[-1], dense), nn.ReLU(),
                                  nn.Linear(dense, 1))

    def forward(self, X, A, gidx, n_graphs):
        H = X
        for c in self.convs:
            H = c(H, A)
        out = torch.zeros(n_graphs, H.shape[1])
        out.index_add_(0, gidx, H)
        cnt = torch.zeros(n_graphs).index_add_(
            0, gidx, torch.ones_like(gidx, dtype=torch.float)).clamp(min=1)
        return self.head(out / cnt[:, None]).squeeze(-1)


class GATLayer(nn.Module):
    def __init__(self, din, dout, heads=8, dropout=0.1):
        super().__init__()
        self.h = heads
        self.dk = dout // heads
        self.W = nn.Linear(din, dout, bias=False)
        self.att_l = nn.Linear(self.dk, 1, bias=False)
        self.att_r = nn.Linear(self.dk, 1, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, X, A):
        N = X.shape[0]
        H = self.W(X).view(N, self.h, self.dk)
        # аддитивное внимание: e_ij = a_l^T h_i + a_r^T h_j (без тензора N^2·dk)
        el = self.att_l(H).squeeze(-1)               # (N,h)
        er = self.att_r(H).squeeze(-1)               # (N,h)
        e = el[:, None, :] + er[None, :, :]          # (N,N,h)
        mask = (A + torch.eye(N)) > 0                # (N,N)
        e = torch.where(mask[:, :, None], e,
                        torch.tensor(-1e9))
        alpha = torch.softmax(e, dim=1)
        out = torch.einsum("ijh,jhd->ihd", alpha, H)
        return torch.relu(self.drop(out.reshape(N, -1)))


class GAT(nn.Module):
    def __init__(self, in_dim=8, hidden=64, heads=8, dropout=0.1):
        super().__init__()
        self.g1 = GATLayer(in_dim, hidden, heads, dropout)
        self.g2 = GATLayer(hidden, hidden, heads, dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(),
                                  nn.Linear(128, 1))

    def encode(self, X, A):
        return self.g2(self.g1(X, A), A)

    def forward(self, X, A, gidx, n_graphs):
        H = self.encode(X, A)
        out = torch.zeros(n_graphs, H.shape[1])
        out.index_add_(0, gidx, H)
        cnt = torch.zeros(n_graphs).index_add_(
            0, gidx, torch.ones_like(gidx, dtype=torch.float)).clamp(min=1)
        return self.head(out / cnt[:, None]).squeeze(-1)


class FPGNN(nn.Module):
    """FP-GNN статьи: GAT-граф + фингерпринт-MLP, фьюжн."""
    def __init__(self, in_dim=8, hidden=64, heads=8, dropout=0.1):
        super().__init__()
        self.gnn = GAT(in_dim, hidden, heads, dropout)
        self.fpn = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(512, 256),
                                 nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden + 256, 128), nn.ReLU(),
                                  nn.Linear(128, 1))

    def forward(self, X, A, gidx, n_graphs, fp):
        H = self.gnn.encode(X, A)
        g = torch.zeros(n_graphs, H.shape[1])
        g.index_add_(0, gidx, H)
        cnt = torch.zeros(n_graphs).index_add_(
            0, gidx, torch.ones_like(gidx, dtype=torch.float)).clamp(min=1)
        g = g / cnt[:, None]
        return self.head(torch.cat([g, self.fpn(fp)], 1)).squeeze(-1)


def train_dl(model, graphs, fp, y_log, tr_idx, epochs=120, lr=1e-3,
             wd=1e-4, bs=32, is_fp=False, is_mlp=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.MSELoss()
    yt = torch.tensor(y_log, dtype=torch.float32)
    model.train()
    for ep in range(epochs):
        perm = np.random.permutation(len(tr_idx))
        for s in range(0, len(perm), bs):
            bidx = [tr_idx[p] for p in perm[s:s + bs]]
            opt.zero_grad()
            if is_mlp:
                out = model(torch.tensor(fp[bidx]))
            else:
                X, A, gidx = collate(graphs, bidx)
                if is_fp:
                    out = model(X, A, gidx, len(bidx),
                                torch.tensor(fp[bidx]))
                else:
                    out = model(X, A, gidx, len(bidx))
            loss = lossf(out, yt[bidx])
            loss.backward()
            opt.step()
    return model


def predict_dl(model, graphs, fp, idx, bs=16, is_fp=False, is_mlp=False):
    model.eval()
    preds = np.zeros(len(idx), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            bidx = idx[s:s + bs]
            if is_mlp:
                out = model(torch.tensor(fp[bidx]))
            else:
                X, A, gidx = collate(graphs, bidx)
                if is_fp:
                    out = model(X, A, gidx, len(bidx),
                                torch.tensor(fp[bidx]))
                else:
                    out = model(X, A, gidx, len(bidx))
            preds[s:s + len(bidx)] = out.numpy()
    return preds


METHODS = ["RF", "XGBoost", "DNN", "GCN", "GAT", "FP-GNN"]
CACHE = os.path.join(HERE, "cache")


def run_methods(methods):
    """Прогон указанных методов по 5 фолдам; артефакты -> cache/."""
    os.makedirs(CACHE, exist_ok=True)
    train, n_official = load_data()
    test = common.load_test()
    y = train[common.TARGET].values.astype(float)
    y_log = np.log1p(y)

    print("ECFP4 (Morgan fingerprint)...")
    fp_tr = common.morgan_fp(train["SMILES"])
    fp_te = common.morgan_fp(test["SMILES"])
    print("Молекулярные графы (RDKit)...")
    graphs = build_graphs(train["SMILES"])
    graphs_te = build_graphs(test["SMILES"])

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oofs = {m: np.zeros(len(y)) for m in methods}
    tes = {m: np.zeros(len(test)) for m in methods}
    timings = {m: 0.0 for m in methods}

    for fold, (tr, va) in enumerate(kf.split(fp_tr), 1):
        tr_l, va_l = list(tr), list(va)
        for m in methods:
            t0 = time.time()
            if m == "RF":
                # диапазоны статьи: n_estimators до 500, max_depth до 15
                md = RandomForestRegressor(
                    n_estimators=500, max_depth=15, min_samples_leaf=1,
                    max_features="sqrt", n_jobs=-1, random_state=SEED)
                md.fit(fp_tr[tr], y_log[tr])
                p_va, p_te = md.predict(fp_tr[va]), md.predict(fp_te)
            elif m == "XGBoost":
                # диапазоны статьи: lr 0.01-0.1, depth 3-5, n_est 50-100
                md = XGBRegressor(
                    n_estimators=100, learning_rate=0.05, max_depth=5,
                    min_child_weight=2, subsample=0.9, colsample_bytree=0.9,
                    gamma=0.05, random_state=SEED, n_jobs=-1)
                md.fit(fp_tr[tr], y_log[tr])
                p_va, p_te = md.predict(fp_tr[va]), md.predict(fp_te)
            elif m == "DNN":
                md = MLP(2048, (512, 256), 0.2)
                train_dl(md, graphs, fp_tr, y_log, tr_l, is_mlp=True)
                p_va = predict_dl(md, graphs, fp_tr, va_l, is_mlp=True)
                p_te = predict_dl(md, graphs_te, fp_te,
                                  list(range(len(test))), is_mlp=True)
            elif m == "GCN":
                md = GCN(8, (128, 128), 128)
                train_dl(md, graphs, fp_tr, y_log, tr_l)
                p_va = predict_dl(md, graphs, fp_tr, va_l)
                p_te = predict_dl(md, graphs_te, fp_te,
                                  list(range(len(test))))
            elif m == "GAT":
                md = GAT(8, 64, 8, 0.1)
                train_dl(md, graphs, fp_tr, y_log, tr_l, epochs=60, bs=8)
                p_va = predict_dl(md, graphs, fp_tr, va_l)
                p_te = predict_dl(md, graphs_te, fp_te,
                                  list(range(len(test))))
            else:  # FP-GNN
                md = FPGNN(8, 64, 8, 0.1)
                train_dl(md, graphs, fp_tr, y_log, tr_l, epochs=60, bs=8,
                         is_fp=True)
                p_va = predict_dl(md, graphs, fp_tr, va_l, is_fp=True)
                p_te = predict_dl(md, graphs_te, fp_te,
                                  list(range(len(test))), is_fp=True)
            oofs[m][va] = common.back_transform(p_va)
            tes[m] += common.back_transform(p_te) / 5
            timings[m] += time.time() - t0
            print(f"  fold {fold}: {m} готов ({timings[m]:.0f} с)")
        np.save(os.path.join(CACHE, "folds_done.txt"), np.array([fold]))

    for m in methods:
        np.save(os.path.join(CACHE, f"oof_{m}.npy"), oofs[m])
        np.save(os.path.join(CACHE, f"test_{m}.npy"), tes[m])
        with open(os.path.join(CACHE, f"time_{m}.txt"), "w") as f:
            f.write(f"{timings[m]:.1f}")


def aggregate():
    """Сводная таблица (как Table 1 в статье) + сырой бленд-сабмит."""
    train, n_official = load_data()
    test = common.load_test()
    y = train[common.TARGET].values.astype(float)
    off = np.arange(n_official)
    bulk = off[y[off] <= 10000]

    methods = [m for m in METHODS
               if os.path.exists(os.path.join(CACHE, f"oof_{m}.npy"))]
    print("\n== Результаты (OOF, официальный train) ==")
    print(f"{'Метод':<10} {'MSE полный':>15} {'MSE bulk':>12} {'время, с':>9}")
    rows, tes = [], {}
    for m in methods:
        oof = np.load(os.path.join(CACHE, f"oof_{m}.npy"))
        tes[m] = np.load(os.path.join(CACHE, f"test_{m}.npy"))
        with open(os.path.join(CACHE, f"time_{m}.txt")) as f:
            t = float(f.read())
        mse_full = float(np.mean((oof[off] - y[off]) ** 2))
        mse_bulk = float(np.mean((oof[bulk] - y[bulk]) ** 2))
        rows.append((m, mse_full, mse_bulk, t))
        print(f"{m:<10} {mse_full:>15,.0f} {mse_bulk:>12,.0f} {t:>9.1f}")

    with open(os.path.join(HERE, "metrics.txt"), "w") as f:
        f.write("method,MSE_full,MSE_bulk,time_s\n")
        for m, mf, mb, t in rows:
            f.write(f"{m},{mf:.1f},{mb:.1f},{t:.1f}\n")

    best = min(rows, key=lambda r: r[2])[0]
    print(f"\nЛучший VDDB-метод по bulk: {best}")
    blend_te = np.mean([tes[m] for m in methods], axis=0)
    common.make_submission(blend_te, os.path.join(
        HERE, "submission_vddb_blend_raw.csv"))
    print("Сырой бленд VDDB-методов (без постобработки и подстановок) "
          "сохранён: submission_vddb_blend_raw.csv")


def main():
    print("== EXP08: бенчмарк методов VDDB (Tao et al., 2023) ==")
    args = sys.argv[1:]
    if args and args[0] == "--aggregate":
        aggregate()
        return
    if args and args[0] == "--methods":
        methods = args[1].split(",")
    else:
        methods = METHODS
    bad = set(methods) - set(METHODS)
    assert not bad, f"неизвестные методы: {bad}"
    run_methods(methods)


def load_data():
    """Та же аугментированная обучающая выборка, что в exp05–07."""
    from experiments.exp06_final_v2.train import load_augmented_train
    return load_augmented_train()


if __name__ == "__main__":
    main()
