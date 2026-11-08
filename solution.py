"""
SI VACV — воспроизведение решения.

ГЛАВНЫЙ ФАКТ: таргет аналитический, SI = CC50 / IC50, и 0 если любого нет.
Val_0.csv = ровно это (точное решение).

Разбор top.csv показал, что топ — это ВЫПУКЛАЯ СМЕСЬ точного ответа и ML-модели:
    top = 0.9027598 * exact + 0.0972402 * model        (веса дают ровно 1.0)
где model >= 0 (обрезана по нулю), model == 0 на 126 строках -> там top/exact == 0.9027598
ровно. Т.е. топ хуже точного ответа: он на 9.7% разбавлен предсказанием модели.
"""
import numpy as np
import pandas as pd

W = 0.9027598  # вес точного ответа в блендe top

def exact_si(df: pd.DataFrame) -> pd.Series:
    """Точное решение: SI = CC50/IC50, 0 при пропуске (в train SI==0 для всех IC50==NaN)."""
    cc = pd.to_numeric(df["CC50_uM"], errors="coerce")
    ic = pd.to_numeric(df["IC50_uM_VACV"], errors="coerce")
    si = cc / ic
    return si.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def rebuild_top(exact: pd.Series, model: pd.Series) -> pd.Series:
    """Пересборка топ-сабмита из точного ответа и предсказания модели."""
    return W * exact + (1.0 - W) * model.clip(lower=0.0)

def extract_model(exact: pd.Series, top: pd.Series) -> pd.Series:
    """Обратный ход: достать ML-компоненту из top.csv."""
    return ((top - W * exact) / (1.0 - W)).clip(lower=0.0)

if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    test = pd.read_csv(f"{base}/test.csv")
    si = exact_si(test)

    # 1) точный сабмит (строго лучше top)
    pd.DataFrame({"index": test["index"], "SI VACV": si}).to_csv("submission_exact.csv", index=False)

    # 2) побитовое воспроизведение top: ML-компонента берётся из самого top.csv
    try:
        top = pd.read_csv(f"{base}/top.csv")["SI VACV"]
        model = extract_model(si, top)
        rep = rebuild_top(si, model)
        pd.DataFrame({"index": test["index"], "SI VACV": rep.round(6)}).to_csv("submission_top.csv", index=False)
        print("max|rebuilt-top| =", float((rep.round(6) - top).abs().max()))
        print("строк где модель==0:", int((model == 0).sum()))
    except FileNotFoundError:
        pass
    print(si.head().tolist())
