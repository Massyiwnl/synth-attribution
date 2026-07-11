"""
build_matrix_table.py - Integrazione "tabella grande"
Ricompone le matrici leave-one-generator-out dal CSV cumulativo prodotto da
eval_matrix.py (una riga per ogni coppia trained_fake x test_gen).

Stampa e salva tre matrici (AUC, AP, Accuracy): righe = generatore di training,
colonne = generatore di test. La diagonale e' "visto", il resto generalizzazione.

Uso:
  python -m src.eval.build_matrix_table --csv report/matrix/matrix_rows.csv \
         --out report/matrix
"""
import argparse
from pathlib import Path

import pandas as pd

# ordine preferito delle righe/colonne (i mancanti vengono ignorati)
ORDER = ["stargan", "attgan", "gdwct", "stylegan1", "stylegan2", "stylegan3"]


def ordered(values):
    vals = list(values)
    return [v for v in ORDER if v in vals] + [v for v in vals if v not in ORDER]


def to_markdown(piv, metric):
    rows_ = ordered(piv.index); cols_ = ordered(piv.columns)
    piv = piv.reindex(index=rows_, columns=cols_)
    head = f"| train \\ test | " + " | ".join(cols_) + " |"
    sep = "|" + "---|" * (len(cols_) + 1)
    lines = [f"### {metric}", head, sep]
    for r in rows_:
        cells = []
        for c in cols_:
            v = piv.loc[r, c]
            cells.append("-" if pd.isna(v) else f"{v:.3f}")
        lines.append(f"| **{r}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="report/matrix/matrix_rows.csv")
    ap.add_argument("--out", default="report/matrix")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # in caso di run ripetute, tieni l'ultima per ogni cella
    df = df.drop_duplicates(subset=["trained_fake", "test_gen"], keep="last")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    blocks = []
    for metric in ["auc", "ap", "acc"]:
        piv = df.pivot(index="trained_fake", columns="test_gen", values=metric)
        piv = piv.reindex(index=ordered(piv.index), columns=ordered(piv.columns))
        print(f"\n=== {metric.upper()} (righe = train, colonne = test) ===")
        print(piv.round(3).to_string(na_rep="-"))
        blocks.append(to_markdown(piv, metric.upper()))

    md = out / "matrix_tables.md"
    md.write_text("\n\n".join(blocks) + "\n")
    print(f"\nTabelle -> {md}")


if __name__ == "__main__":
    main()
