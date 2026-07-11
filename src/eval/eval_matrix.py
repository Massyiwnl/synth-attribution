"""
eval_matrix.py - Integrazione "tabella grande" (leave-one-generator-out)

Dato UN modello (allenato su reali + un solo generatore G), calcola la sua RIGA
della matrice: quanto separa i reali da OGNI generatore di test Y.

Score di attribuzione (coerente col metric learning): distanza dell'embedding di
un'immagine dal CENTROIDE dei reali. Reale -> vicino; fake -> lontano. Per ogni
generatore Y (nella sua stessa lineage, usando i reali di quella lineage come
negativi) si misura:
  - AUC  (roc_auc, reale-vs-Y)
  - AP   (average precision, reale-vs-Y)
  - Accuracy a una soglia tau, dove tau e' fissata sulla DIAGONALE (reale-vs-G,
    il generatore visto) al punto di EER; poi applicata a tutte le colonne.
La diagonale (Y == G) e' "visto"; il resto e' generalizzazione.

Ogni run appende le sue righe (formato lungo) a un CSV cumulativo, da cui
build_matrix_table.py ricompone le matrici.

Uso:
  python -m src.eval.eval_matrix --checkpoint runs/ffhq__stylegan2/best.pt \
         --trained-fake stylegan2 --out report/matrix/ffhq__stylegan2
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from src.data.siamese_dataset import SingleImageDataset
from src.models.siamese import SiameseEncoder


@torch.no_grad()
def embed_test(checkpoint, manifest, size, device, batch_size, num_workers):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    mcfg = ckpt["cfg"]["model"]
    model = SiameseEncoder(mcfg["backbone"], mcfg["pretrained"], mcfg["embedding_dim"],
                           front_end=mcfg.get("front_end", "none")).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ds = SingleImageDataset(manifest, "test", size, lineage_filter=None)  # tutte le lineage
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device == "cuda"))
    Z = []
    for x, _ in loader:
        Z.append(model.forward_one(x.to(device)).cpu().numpy())
    Z = np.concatenate(Z).astype(np.float32)
    arch = ds.df["architecture"].to_numpy()
    src = ds.df["source_dataset"].to_numpy()
    return Z, arch, src, ckpt.get("epoch")


def subsample(idx, n, rng):
    return rng.choice(idx, n, replace=False) if len(idx) > n else idx


def eer_threshold(scores, y):
    """Soglia al punto di Equal Error Rate (score alto = fake)."""
    fpr, tpr, thr = roc_curve(y, scores)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float(thr[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--trained-fake", required=True,
                    help="il generatore su cui il modello e' stato allenato (diagonale + soglia)")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--n-per-class", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="report/matrix/run")
    ap.add_argument("--matrix-csv", default="report/matrix/matrix_rows.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    Z, arch, src, epoch = embed_test(args.checkpoint, args.manifest, args.image_size,
                                     device, args.batch_size, args.num_workers)
    G = args.trained_fake
    fakes = sorted([a for a in np.unique(arch) if a != "real"])
    # lineage di ciascun generatore (dal manifest)
    gen_lineage = {a: src[arch == a][0] for a in fakes}
    if G not in fakes:
        raise SystemExit(f"'{G}' non e' fra i generatori del test: {fakes}")
    trained_lineage = gen_lineage[G]
    print(f"Modello: allenato su {trained_lineage}+{G} | epoca {epoch} | device {device}")
    print(f"Generatori di test: {fakes}")

    # centroide dei reali per lineage (dai reali di test, bilanciati)
    def real_scores_centroid(lineage):
        ridx = np.where((arch == "real") & (src == lineage))[0]
        ridx = subsample(ridx, args.n_per_class, rng)
        rZ = Z[ridx]
        c = rZ.mean(0)
        d_real = np.linalg.norm(rZ - c, axis=1)
        return c, d_real

    centroids, real_d = {}, {}
    for lin in sorted(set(gen_lineage.values())):
        centroids[lin], real_d[lin] = real_scores_centroid(lin)

    # distanze fake per ciascun generatore
    fake_d = {}
    for Y in fakes:
        lin = gen_lineage[Y]
        fidx = np.where(arch == Y)[0]
        fidx = subsample(fidx, args.n_per_class, rng)
        fake_d[Y] = np.linalg.norm(Z[fidx] - centroids[lin], axis=1)

    # soglia tau dalla diagonale (reale_lineage(G) vs G) al punto di EER
    linG = trained_lineage
    y_diag = np.r_[np.ones(len(fake_d[G])), np.zeros(len(real_d[linG]))]
    s_diag = np.r_[fake_d[G], real_d[linG]]
    tau = eer_threshold(s_diag, y_diag)

    # metriche per ogni generatore Y
    rows, per_gen = [], {}
    for Y in fakes:
        lin = gen_lineage[Y]
        s = np.r_[fake_d[Y], real_d[lin]]
        y = np.r_[np.ones(len(fake_d[Y])), np.zeros(len(real_d[lin]))]
        auc = float(roc_auc_score(y, s))
        aprec = float(average_precision_score(y, s))
        acc = float((((s > tau).astype(int)) == y).mean())
        per_gen[Y] = dict(auc=auc, ap=aprec, acc=acc, seen=(Y == G),
                          test_lineage=lin, n_fake=int(len(fake_d[Y])),
                          n_real=int(len(real_d[lin])),
                          mean_d_fake=float(fake_d[Y].mean()),
                          mean_d_real=float(real_d[lin].mean()))
        rows.append(dict(trained_lineage=trained_lineage, trained_fake=G,
                         test_gen=Y, test_lineage=lin, seen=int(Y == G),
                         auc=round(auc, 4), ap=round(aprec, 4), acc=round(acc, 4)))

    # sweep di soglia sulla diagonale (per la domanda "al variare della soglia")
    lo, hi = float(s_diag.min()), float(s_diag.max())
    sweep = []
    for t in np.linspace(lo, hi, 11):
        pred = (s_diag > t).astype(int)
        tp = int(((pred == 1) & (y_diag == 1)).sum()); fp = int(((pred == 1) & (y_diag == 0)).sum())
        tn = int(((pred == 0) & (y_diag == 0)).sum()); fn = int(((pred == 0) & (y_diag == 1)).sum())
        sweep.append(dict(thr=round(float(t), 4),
                          acc=round((tp + tn) / len(y_diag), 4),
                          tpr=round(tp / max(tp + fn, 1), 4),
                          fpr=round(fp / max(fp + tn, 1), 4)))

    # stampa leggibile della riga
    print(f"\nSoglia tau (EER sulla diagonale {G}): {tau:.4f}")
    print(f"{'test_gen':>12s} | {'lineage':>7s} | {'AUC':>6s} {'AP':>6s} {'Acc':>6s} | visto")
    for Y in fakes:
        g = per_gen[Y]
        star = "  <-- diagonale" if g["seen"] else ""
        print(f"{Y:>12s} | {g['test_lineage']:>7s} | {g['auc']:.4f} {g['ap']:.4f} "
              f"{g['acc']:.4f} | {'si' if g['seen'] else 'no'}{star}")

    # output
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    report = dict(checkpoint=args.checkpoint, epoch=epoch, trained_lineage=trained_lineage,
                  trained_fake=G, tau_eer=tau, n_per_class=args.n_per_class,
                  per_generator=per_gen, threshold_sweep_diagonal=sweep)
    (out / "report.json").write_text(json.dumps(report, indent=2))

    mcsv = Path(args.matrix_csv); mcsv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not mcsv.exists()
    with open(mcsv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trained_lineage", "trained_fake", "test_gen",
                                          "test_lineage", "seen", "auc", "ap", "acc"])
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"\nReport -> {out/'report.json'}   |   righe matrice -> {mcsv}")


if __name__ == "__main__":
    main()
