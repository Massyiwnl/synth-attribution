"""
handcrafted_baseline.py - Fase 6
Baseline a feature spettrali (Durall: spettro di potenza azimutale FFT;
Frank: spettro 2D della DCT) + classificatore classico, per confrontare il
paradigma handcrafted del lavoro precedente con l'embedding deep SULLA STESSA
prova di generalizzazione (leave-one-architecture-out).

Feature per immagine (--features radial|dct|both):
  - radial: spettro di potenza azimutale (FFT 2D -> media radiale, log)  [128-d a 256px]
  - dct:    log|DCT 2D| ridotto a una griglia gridxgrid (default 16 -> 256-d)
Classificatori: RandomForest e SVM (RBF).

Autosufficiente: split deterministico interno su real+train-fake; TUTTE le immagini
di <holdout-fake> come test di generalizzazione. Nessun rebuild del manifest.

Metrica chiave di generalizzazione: gen_auc = AUC con cui il detector (allenato
real-vs-train-fake) separa l'holdout-fake DAI REALI. ~0.5 = l'holdout passa per reale.

Uso:
  python -m src.eval.handcrafted_baseline --manifest data/manifest.csv --lineage ffhq \
      --train-fake stylegan2 --holdout-fake stylegan3 --features both \
      --out report/handcrafted_ffhq_sg3
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score

try:
    from scipy.fft import dctn
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def read_manifest(path, lineage):
    df = pd.read_csv(path, dtype={"source_id": str}, keep_default_na=False)
    if lineage:
        df = df[df.source_dataset == lineage]
    return df.reset_index(drop=True)


_R_CACHE = {}
def radial_index(size):
    if size not in _R_CACHE:
        c = size // 2
        y, x = np.ogrid[:size, :size]
        _R_CACHE[size] = np.sqrt((y - c) ** 2 + (x - c) ** 2).astype(int)
    return _R_CACHE[size]


def radial_psd(g, size):
    F = np.fft.fftshift(np.fft.fft2(g))
    P = F.real ** 2 + F.imag ** 2
    r = radial_index(size); nb = size // 2
    tbin = np.bincount(r.ravel(), weights=P.ravel())
    cnt = np.bincount(r.ravel())
    return np.log1p(tbin[:nb] / np.maximum(cnt[:nb], 1))


def dct_pooled(g, grid=16):
    d = dctn(g, norm="ortho")
    ld = np.log(np.abs(d) + 1e-8)
    H = (ld.shape[0] // grid) * grid
    W = (ld.shape[1] // grid) * grid
    ld = ld[:H, :W]
    return ld.reshape(grid, H // grid, grid, W // grid).mean(axis=(1, 3)).ravel()


def _load_gray(path, size, jpeg=None):
    im = Image.open(path).convert("L").resize((size, size))
    if jpeg is not None:
        import io
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=int(jpeg)); buf.seek(0)
        im = Image.open(buf).convert("L")
    return np.asarray(im, dtype=np.float64)


def feats_of(path, size, use_radial, use_dct, jpeg=None):
    g = _load_gray(path, size, jpeg)
    parts = []
    if use_radial:
        parts.append(radial_psd(g, size))
    if use_dct and _HAS_SCIPY:
        parts.append(dct_pooled(g))
    return np.concatenate(parts)


def extract(paths, size, use_radial, use_dct, jpeg=None):
    return np.stack([feats_of(p, size, use_radial, use_dct, jpeg)
                     for p in tqdm(paths, desc="feat", leave=False)])


def split_tt(d, frac=0.8, seed=0):
    idx = d.index.to_numpy().copy()
    np.random.default_rng(seed).shuffle(idx)
    k = int(len(idx) * frac)
    return d.loc[idx[:k]], d.loc[idx[k:]]


def balance(X, arch, n, seed=0):
    rng = np.random.default_rng(seed); keep = []
    for c in np.unique(arch):
        ci = np.where(arch == c)[0]
        if len(ci) > n:
            ci = rng.choice(ci, n, replace=False)
        keep.append(ci)
    keep = np.concatenate(keep)
    return X[keep], arch[keep]


def pdist(a, b):
    aa = (a * a).sum(1)[:, None]; bb = (b * b).sum(1)[None, :]
    return np.sqrt(np.maximum(aa + bb - 2 * a @ b.T, 0.0))


def class_dist(X, arch, classes):
    M = np.zeros((len(classes), len(classes)))
    for i, ca in enumerate(classes):
        A = X[arch == ca]
        for j, cb in enumerate(classes):
            B = X[arch == cb]; d = pdist(A, B)
            M[i, j] = ((d.sum() - np.trace(d)) / max(len(A) * (len(A) - 1), 1)
                       if ca == cb else d.mean())
    return M


def get_scores(clf, Z):
    return clf.decision_function(Z) if hasattr(clf, "decision_function") else clf.predict_proba(Z)[:, 1]


def save_spectrum_fig(radial_un, arch, classes, path):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        print("  (matplotlib assente: salto la figura spettro)"); return None
    freq = np.arange(radial_un.shape[1]); plt.figure(figsize=(7, 5)); means = {}
    for c in classes:
        m = radial_un[arch == c].mean(0); means[c] = m; plt.plot(freq, m, label=c)
    plt.xlabel("frequenza radiale (bin, 0=DC .. Nyquist)"); plt.ylabel("log-potenza media")
    plt.title("Spettro di potenza azimutale medio per classe"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
    print(f"  figura spettro -> {path}")
    return {c: means[c].tolist() for c in classes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--lineage", default="ffhq")
    ap.add_argument("--train-fake", required=True)
    ap.add_argument("--holdout-fake", required=True)
    ap.add_argument("--features", choices=["radial", "dct", "both"], default="both")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--n-per-class", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jpeg", type=int, default=None,
                    help="se impostato (es. 85), riapplica JPEG a TUTTE le classi prima "
                         "delle feature: test del confound di preprocessing/resize")
    ap.add_argument("--out", default="report/handcrafted")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    H = args.holdout_fake
    use_radial = args.features in ("radial", "both")
    use_dct = args.features in ("dct", "both")
    if use_dct and not _HAS_SCIPY:
        print("scipy assente: uso solo feature radiali (per la DCT serve scipy).")
        use_dct = False

    df = read_manifest(args.manifest, args.lineage)
    real = df[df.architecture == "real"]
    tf = df[df.architecture == args.train_fake]
    hf = df[df.architecture == H]
    if len(real) == 0 or len(tf) == 0 or len(hf) == 0:
        print("Mancano classi (real / train-fake / holdout-fake) per questa lineage."); return

    real_tr, real_te = split_tt(real, seed=args.seed)
    tf_tr, tf_te = split_tt(tf, seed=args.seed + 1)
    tr = pd.concat([real_tr, tf_tr]); te = pd.concat([real_te, tf_te])

    feat_name = "+".join([n for n, u in [("radial", use_radial), ("dct", use_dct)] if u])
    jtag = f" | JPEG q={args.jpeg}" if args.jpeg else ""
    print(f"Feature [{feat_name}]{jtag} | lineage={args.lineage} | train real+{args.train_fake} "
          f"({len(tr)}) | test in-dist ({len(te)}) | holdout {H} ({len(hf)})")
    Xtr = extract(tr.path.tolist(), args.image_size, use_radial, use_dct, args.jpeg)
    Xte = extract(te.path.tolist(), args.image_size, use_radial, use_dct, args.jpeg)
    Xhf = extract(hf.path.tolist(), args.image_size, use_radial, use_dct, args.jpeg)
    ytr = (tr.label == "fake").astype(int).to_numpy()
    yte = (te.label == "fake").astype(int).to_numpy()

    sc = StandardScaler().fit(Xtr)
    Ztr, Zte, Zhf = sc.transform(Xtr), sc.transform(Xte), sc.transform(Xhf)
    real_te_mask = (te.architecture == "real").to_numpy()

    clfs = {
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
        "SVM_rbf": SVC(kernel="rbf", random_state=0),   # no probability -> decision_function
    }
    det = {}
    print("\n== Detection (real vs fake) ==")
    for name, clf in clfs.items():
        clf.fit(Ztr, ytr)
        s_te = get_scores(clf, Zte)
        auc = float(roc_auc_score(yte, s_te))
        acc = float(accuracy_score(yte, clf.predict(Zte)))
        fr = float((clf.predict(Zhf) == 1).mean())
        fpr = float((clf.predict(Zte)[real_te_mask] == 1).mean())
        # generalizzazione pulita: il detector separa l'holdout DAI REALI?
        s_real = get_scores(clf, Zte[real_te_mask]); s_hold = get_scores(clf, Zhf)
        y_gen = np.r_[np.zeros(len(s_real)), np.ones(len(s_hold))]
        gen_auc = float(roc_auc_score(y_gen, np.r_[s_real, s_hold]))
        det[name] = dict(indist_auc=auc, indist_acc=acc, holdout_fake_rate=fr,
                         real_test_fpr=fpr, holdout_vs_real_auc=gen_auc)
        print(f"  [{name}] in-dist real-vs-{args.train_fake}: AUC={auc:.3f} acc={acc:.3f} (FPR {fpr*100:.0f}%)")
        print(f"           generalizz. -> {H} vs reali: AUC={gen_auc:.3f} | {H} chiamato fake {fr*100:.0f}%")

    Zall = np.concatenate([Zte, Zhf]); arch_all = pd.concat([te, hf]).architecture.to_numpy()
    Zb, ab = balance(Zall, arch_all, args.n_per_class, args.seed)
    classes = sorted(np.unique(ab).tolist())
    M = class_dist(Zb, ab, classes)
    D = pdist(Zb, Zb); iu = np.triu_indices(len(Zb), 1); dp = D[iu]
    np.fill_diagonal(D, np.inf); nn = ab[D.argmin(1)]
    hf_nn = float((nn[ab == H] == H).mean())
    hi = (ab[iu[0]] == H) | (ab[iu[1]] == H)
    yh = ((ab[iu[0]] == H) & (ab[iu[1]] == H)).astype(int)
    hf_auc = float(roc_auc_score(yh[hi], -dp[hi])) if len(np.unique(yh[hi])) > 1 else None
    print(f"\n== Geometria spazio feature (held-out {H}) ==")
    print(f"  1-NN {H} in classe: {hf_nn*100:.1f}%   AUC focalizz.: "
          + (f"{hf_auc:.3f}" if hf_auc is not None else "n/d"))
    print("           " + "".join(f"{c[:9]:>11s}" for c in classes))
    for i, c in enumerate(classes):
        print(f"  {c[:9]:>9s} " + "".join(f"{M[i, j]:11.3f}" for j in range(len(classes))))

    spectra = None
    if use_radial:
        nb = args.image_size // 2
        radial_un = np.concatenate([Xtr, Xte, Xhf])[:, :nb]
        aun = pd.concat([tr, te, hf]).architecture.to_numpy()
        spectra = save_spectrum_fig(radial_un, aun, classes, out / "radial_spectrum.png")

    report = dict(lineage=args.lineage, train_fake=args.train_fake, holdout_fake=H,
                  features=feat_name, n_per_class=int(args.n_per_class),
                  detection=det, classes=classes,
                  feature_space=dict(holdout_nn_same_class=hf_nn, holdout_focused_auc=hf_auc,
                                     class_distance_matrix=M.tolist()),
                  mean_radial_spectrum=spectra)
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport -> {out/'report.json'}")


if __name__ == "__main__":
    main()
