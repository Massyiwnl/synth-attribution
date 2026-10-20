"""
build_binary_table.py - riassume i report di eval_binary.py in una tabella markdown.

Per ogni run confronta tre metodi sullo stesso embedding e sullo stesso test:
  - centroidi, protocollo originale (centroidi e soglia calcolati sul test)
  - centroidi, pulito (centroidi sul train, soglia sul val)
  - binario (testa lineare Linear(128, 2), soglia sul val)
e l'accuratezza per generatore con il binario.

Uso:
  python -m src.eval.build_binary_table --out report/binary_summary.md
"""
import argparse
import json
from pathlib import Path

GENERATORS = ["stargan", "attgan", "gdwct", "stylegan2", "stylegan3",
              "real_celeba", "real_ffhq"]


def regime(r, name):
    """Etichetta leggibile del regime, dedotta da manifest e patch del checkpoint."""
    man = r["manifest"]
    if "realonly" in man:
        base = "real-only"
    elif "pm128" in man:
        base = "normalized"
    else:
        base = "original dataset"
    p = r["patch"]
    geo = "whole image 256" if not p["size"] else f"patch {p['size']} px, grid"
    hp = " + high-pass residual" if name.endswith("_hp") else ""
    return f"{base}, {geo}{hp}"


def pct(v):
    return "n/a" if v is None else f"{v * 100:.1f}%"


def num(v):
    return "n/a" if v is None else f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="report")
    ap.add_argument("--out", default="report/binary_summary.md")
    args = ap.parse_args()

    runs = []
    for path in sorted(Path(args.reports).glob("binary_*/report.json")):
        name = path.parent.name.replace("binary_", "", 1)
        if name.startswith("gradcam_") or name.startswith("robustness_"):
            continue  # report di altri script, non di eval_binary.py
        r = json.loads(path.read_text())
        if "methods" not in r:
            continue
        runs.append((name, r))

    lines = ["# Binary head on the frozen encoder: comparison with centroids", ""]
    lines += ["Test set sampled as in `eval_lineage.py`. Threshold: EER on the validation set (clean) "
              "or on the test set (original protocol, for comparison).", ""]
    lines += ["## Methods compared", "",
              "| run | regime | AUC unseen: original | AUC unseen: clean | AUC unseen: binary "
              "| acc. unseen: original | acc. unseen: clean | acc. unseen: binary "
              "| FAR binary | FRR binary | tau binary |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in runs:
        m = r["methods"]
        o, c, b = (m["centroid_original_protocol"], m["centroid_clean"], m["binary_clean"])
        lines.append(
            f"| {name} | {regime(r, name)} | {num(o['auc_unseen'])} | {num(c['auc_unseen'])} "
            f"| {num(b['auc_unseen'])} | {pct(o['acc_unseen'])} | {pct(c['acc_unseen'])} "
            f"| {pct(b['acc_unseen'])} | {pct(b['far_unseen'])} | {pct(b['frr_unseen'])} "
            f"| {b['tau']:+.4f} |")

    lines += ["", "## Per-generator accuracy with the binary head", "",
              "`*` = generator unseen in that run.", "",
              "| run | " + " | ".join(GENERATORS) + " |",
              "|---|" + "---|" * len(GENERATORS)]
    for name, r in runs:
        pc = r["per_class"]
        seen_archs = set(r["seen_architectures"]) | {"real"}
        cells = []
        for g in GENERATORS:
            if g not in pc:
                cells.append("n/a")
                continue
            star = "" if g in seen_archs or g.startswith("real") else " *"
            cells.append(pct(pc[g]["correct_rate"]) + star)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
