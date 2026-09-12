"""
class_signatures.py - Fase 8
Firme statistiche per classe (frequenza + colore) e, soprattutto, il PAVIMENTO
HANDCRAFTED del problema di lineage.

Serve a due scopi.

(A) EVIDENZA COMPLEMENTARE. Produce, per ogni classe, descrittori interpretabili:
      - profilo di potenza azimutale (FFT radiale, 128-d)
      - mappa media log|FFT 2D| e log|DCT 2D|
      - istogrammi di colore RGB e HSV
    con le relative matrici di distanza fra classi. Risponde alla domanda "che
    cosa accomuna CelebA e StarGAN?" in termini leggibili, non solo con un
    embedding opaco.

(B) PAVIMENTO / CONTROLLO DEL CONFOUND — il motivo principale.
    Per ogni famiglia di descrittori si allena una semplice regressione logistica
    a predire la LINEAGE usando SOLO le architetture viste in training, e la si
    testa sulle architetture HELD-OUT. E' lo stesso protocollo del modello deep.

    La lettura e' il punto cruciale dell'articolo:
      - se un descrittore banale (es. il solo profilo radiale, che e' in pratica
        una firma di ricampionamento) raggiunge gia' AUC ~0.95+ sui mai visti,
        allora il risultato del VLM NON dimostra l'attribuzione dei dati di
        addestramento: dimostra che le due lineage hanno subito resize diversi;
      - il contributo del deep e' il DELTA rispetto a questo pavimento, e va
        riportato esplicitamente.

    Va eseguito sia sul dataset originale sia su quello normalizzato con
    scripts/normalize_pipeline.py: il confronto fra i due pavimenti quantifica
    quanta parte del segnale era ricampionamento.

Uso:
  python -m src.eval.class_signatures --manifest data/manifest_lineage.csv \
      --out report/signatures_raw
  python -m src.eval.class_signatures --manifest data/manifest_lineage_pm128.csv \
      --out report/signatures_pm128
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

try:
    from scipy.fft import dctn
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

_R = {}


def radial_index(size):
    if size not in _R:
        c = size // 2
        y, x = np.ogrid[:size, :size]
        _R[size] = np.sqrt((y - c) ** 2 + (x - c) ** 2).astype(int)
    return _R[size]


def descriptors(path, size, patcher=None, patch_idx=0):
    """Tutte le famiglie di descrittori per una immagine (o per una sua patch)."""
    im = Image.open(path).convert("RGB")
    if patcher is not None:
        # patch a risoluzione NATIVA: nessun resize, quindi nessuna
        # interpolazione aggiunta sopra quella che stiamo gia' controllando
        im = patcher(im, patch_idx)
        size = im.size[0]
    elif im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    rgb = np.asarray(im, dtype=np.float64)
    g = np.asarray(im.convert("L"), dtype=np.float64)

    # --- frequenza ---
    F = np.fft.fftshift(np.fft.fft2(g))
    P = F.real ** 2 + F.imag ** 2
    logfft = np.log1p(P)
    r = radial_index(size)
    nb = size // 2
    tot = np.bincount(r.ravel(), weights=P.ravel())[:nb]
    cnt = np.bincount(r.ravel())[:nb]
    radial = np.log1p(tot / np.maximum(cnt, 1))

    if _HAS_SCIPY:
        d = np.log(np.abs(dctn(g, norm="ortho")) + 1e-8)
        k16 = 16
        H = (d.shape[0] // k16) * k16
        dct_map = d[:H, :H].reshape(k16, H // k16, k16, H // k16).mean(axis=(1, 3))
    else:
        k16, H = 16, 256
        dct_map = np.zeros((16, 16))

    # --- residuo high-pass: sopprime il CONTENUTO semantico e tiene la texture ---
    # Serve a separare due spiegazioni concorrenti della separabilita' fra lineage:
    #   (a) contenuto: CelebA e FFHQ hanno inquadratura, allineamento, illuminazione
    #       e demografia diverse -> distinguibili A OCCHIO, quindi non e' forense;
    #   (b) traccia sub-percettiva legata ai dati di addestramento -> e' cio' che
    #       l'articolo vuole dimostrare.
    # Se i descrittori sul RESIDUO continuano a separare le lineage, (a) da sola non
    # basta a spiegare il risultato.
    k = np.ones((3, 3)) / 9.0
    lowp = np.zeros_like(g)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            lowp += np.roll(np.roll(g, dy, 0), dx, 1) * k[dy + 1, dx + 1]
    res = g - lowp
    Fr = np.fft.fftshift(np.fft.fft2(res))
    Pr = Fr.real ** 2 + Fr.imag ** 2
    totr = np.bincount(r.ravel(), weights=Pr.ravel())[:nb]
    radial_res = np.log1p(totr / np.maximum(cnt, 1))
    if _HAS_SCIPY:
        dr = np.log(np.abs(dctn(res, norm="ortho")) + 1e-8)
        dct_res = dr[:H, :H].reshape(k16, H // k16, k16, H // k16).mean(axis=(1, 3)).ravel()
    else:
        dct_res = np.zeros(256)
    residual = np.concatenate([radial_res, dct_res])

    # --- colore ---
    hsv = np.asarray(im.convert("HSV"), dtype=np.float64)
    hist = []
    for arr in (rgb, hsv):
        for ch in range(3):
            h, _ = np.histogram(arr[:, :, ch], bins=32, range=(0, 255), density=True)
            hist.append(h)
    color = np.concatenate(hist)                       # 192-d

    return dict(radial=radial, dct=dct_map.ravel(), color=color, residual=residual,
                logfft_map=logfft, dct_map=dct_map)


def dist_matrix(means, classes):
    M = np.zeros((len(classes), len(classes)))
    for i, a in enumerate(classes):
        for j, b in enumerate(classes):
            M[i, j] = float(np.linalg.norm(means[a] - means[b]))
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest_lineage.csv")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--n-per-class", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patch-size", type=int, default=None,
                    help="se impostato, i descrittori sono calcolati su patch native")
    ap.add_argument("--patches-per-image", type=int, default=1)
    ap.add_argument("--patch-policy", default="flat")
    ap.add_argument("--out", default="report/signatures")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if not _HAS_SCIPY:
        print("scipy assente: la famiglia 'dct' sara' nulla.")

    df = pd.read_csv(args.manifest, dtype={"source_id": str}, keep_default_na=False)
    df["cls"] = np.where(df.label == "real", "real_" + df.source_dataset, df.architecture)
    seen = set(df[df.split == "train"].architecture.unique())
    print(f"Architetture viste in training: {sorted(seen)}")

    from src.data.siamese_dataset import PatchExtractor, expand_patches
    patcher = (PatchExtractor(args.patch_size, args.patch_policy, seed=args.seed)
               if args.patch_size else None)
    if patcher:
        print(f"Descrittori su PATCH {args.patch_size}x{args.patch_size} "
              f"({args.patch_policy}), {args.patches_per_image} per immagine")

    rng = np.random.default_rng(args.seed)
    rows = []
    for c, sub in df.groupby("cls"):
        idx = sub.index.to_numpy()
        if len(idx) > args.n_per_class:
            idx = rng.choice(idx, args.n_per_class, replace=False)
        rows.append(df.loc[idx])
    sample = pd.concat(rows).reset_index(drop=True)
    if patcher:
        sample = expand_patches(sample, args.patches_per_image)
    else:
        sample["patch_idx"] = 0
    print(f"Campione: {len(sample)} campioni su {sample.cls.nunique()} classi")

    feats = {k: [] for k in ("radial", "dct", "color", "residual")}
    fft_sum, dct_sum, n_by_cls = {}, {}, {}
    for _, r in tqdm(sample.iterrows(), total=len(sample), desc="descrittori"):
        d = descriptors(r.path, args.image_size, patcher, int(r.patch_idx))
        for k in feats:
            feats[k].append(d[k])
        fft_sum[r.cls] = fft_sum.get(r.cls, 0) + d["logfft_map"]
        dct_sum[r.cls] = dct_sum.get(r.cls, 0) + d["dct_map"]
        n_by_cls[r.cls] = n_by_cls.get(r.cls, 0) + 1
    X = {k: np.stack(v) for k, v in feats.items()}
    X["all"] = np.concatenate([X["radial"], X["dct"], X["color"]], axis=1)

    cls = sample.cls.to_numpy()
    lin = sample.source_dataset.to_numpy()
    arch = sample.architecture.to_numpy()
    classes = sorted(np.unique(cls).tolist())
    seen_mask = np.array([(a == "real") or (a in seen) for a in arch])

    # ---------- (A) firme e distanze ----------
    report = dict(manifest=args.manifest, n_per_class=args.n_per_class,
                  patch_size=args.patch_size,
                  classes=classes, seen_architectures=sorted(seen),
                  mean_radial={}, distance_matrices={}, floor={})
    for c in classes:
        report["mean_radial"][c] = X["radial"][cls == c].mean(0).tolist()

    print("\n== (A) Distanze fra firme medie di classe ==")
    for fam in ("radial", "dct", "color", "residual"):
        means = {c: X[fam][cls == c].mean(0) for c in classes}
        M = dist_matrix(means, classes)
        report["distance_matrices"][fam] = M.tolist()
        print(f"\n-- {fam} --")
        print("             " + "".join(f"{c[:10]:>11s}" for c in classes))
        for i, c in enumerate(classes):
            print(f"{c[:11]:>11s}  " + "".join(f"{M[i, j]:11.3f}" for j in range(len(classes))))

    # ---------- (B) pavimento handcrafted ----------
    print("\n== (B) PAVIMENTO handcrafted: predire la LINEAGE con descrittori banali ==")
    print("   (allenato SOLO sulle architetture viste, testato sui MAI VISTI)")
    y = (lin == "ffhq").astype(int)
    print(f"\n{'famiglia':>10s} {'dim':>5s} {'AUC visti':>10s} {'AUC mai visti':>14s}")
    for fam in ("radial", "dct", "color", "residual", "all"):
        Xf = X[fam]
        if Xf.shape[1] == 0 or np.allclose(Xf, 0):
            continue
        sc = StandardScaler().fit(Xf[seen_mask])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.transform(Xf[seen_mask]), y[seen_mask])
        s_seen = clf.decision_function(sc.transform(Xf[seen_mask]))
        auc_seen = float(roc_auc_score(y[seen_mask], s_seen))
        if len(np.unique(y[~seen_mask])) > 1:
            s_un = clf.decision_function(sc.transform(Xf[~seen_mask]))
            auc_un = float(roc_auc_score(y[~seen_mask], s_un))
        else:
            auc_un = None
        # quota di immagini assegnate alla lineage CORRETTA, per classe: e' la
        # colonna direttamente confrontabile con la tabella del modello deep
        pred = clf.predict(sc.transform(Xf))
        per_cls = {c: float((pred[cls == c] == y[cls == c]).mean())
                   for c in classes}
        report["floor"][fam] = dict(dim=int(Xf.shape[1]), auc_seen=auc_seen,
                                    auc_unseen=auc_un, per_class_correct=per_cls)
        print(f"{fam:>10s} {Xf.shape[1]:5d} {auc_seen:10.4f} "
              f"{'n/d' if auc_un is None else f'{auc_un:14.4f}'}")

    # breakdown per classe della famiglia migliore: quanto ci arriva gia'
    # un descrittore banale, generatore per generatore
    bestfam = max(report["floor"].items(), key=lambda kv: (kv[1]["auc_unseen"] or 0))
    print(f"\nQuota assegnata alla lineage corretta - famiglia '{bestfam[0]}':")
    for c in classes:
        a = arch[cls == c][0]
        mark = "si" if (a == "real" or a in seen) else "NO"
        print(f"  {c:>14s}  visto={mark:>2s}  {bestfam[1]['per_class_correct'][c]*100:6.1f}%")

    best = max((v["auc_unseen"] or 0) for v in report["floor"].values())
    print(f"\n>>> PAVIMENTO sui generatori mai visti: AUC {best:.4f}")
    print("    Qualsiasi risultato del modello deep va letto come DELTA rispetto a")
    print("    questo valore. Se il deep non lo supera in modo netto, non sta")
    print("    dimostrando l'attribuzione dei dati di addestramento.")

    save_figs(report, fft_sum, dct_sum, n_by_cls, classes, out)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nReport -> {out/'report.json'}")


def save_figs(report, fft_sum, dct_sum, n_by_cls, classes, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  (matplotlib assente: salto le figure)")
        return

    # spettro radiale medio per classe
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in classes:
        ax.plot(report["mean_radial"][c], label=c, lw=1.4)
    ax.set_xlabel("frequenza radiale (bin; 0 = DC .. Nyquist)")
    ax.set_ylabel("log-potenza media")
    ax.set_title("Spettro di potenza azimutale medio per classe")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "radial_spectrum.png", dpi=140); plt.close(fig)

    # zoom sulle alte frequenze: e' li' che vive la firma di ricampionamento
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in classes:
        v = report["mean_radial"][c]
        ax.plot(range(80, len(v)), v[80:], label=c, lw=1.4)
    ax.set_xlabel("frequenza radiale (bin, zoom alte frequenze)")
    ax.set_ylabel("log-potenza media")
    ax.set_title("Alte frequenze: qui vive la firma di ricampionamento")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "radial_spectrum_hf.png", dpi=140); plt.close(fig)

    # mappe medie log|FFT| per classe
    n = len(classes)
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.6))
    axes = np.atleast_2d(axes)
    for j, c in enumerate(classes):
        m = fft_sum[c] / n_by_cls[c]
        axes[0, j].imshow(m, cmap="viridis"); axes[0, j].set_title(c, fontsize=8)
        axes[0, j].axis("off")
        d = dct_sum[c] / n_by_cls[c]
        axes[1, j].imshow(d, cmap="magma"); axes[1, j].axis("off")
    axes[0, 0].set_ylabel("log|FFT|"); axes[1, 0].set_ylabel("log|DCT|")
    fig.suptitle("Mappe spettrali medie per classe (sopra log|FFT 2D|, sotto log|DCT 2D|)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(out / "spectral_maps.png", dpi=140); plt.close(fig)
    print(f"  figure -> {out}/radial_spectrum.png , radial_spectrum_hf.png , spectral_maps.png")


if __name__ == "__main__":
    main()
