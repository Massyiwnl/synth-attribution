"""
plot_patch_sweep.py - Fase 8
Figura e tabella dello sweep sulla dimensione della patch.

DOMANDA. L'attribuzione al dataset di addestramento e' una somiglianza VISIBILE
(StyleGAN3 produce volti che sembrano FFHQ) o una traccia di BASSO LIVELLO?
Sull'immagine intera le due spiegazioni sono indistinguibili: entrambe funzionano.
Rimpicciolendo la finestra di analisi si toglie progressivamente il contenuto
semantico - inquadratura, composizione, demografia - lasciando texture e rumore.

Si confrontano percio' TRE metodi sullo stesso identico dato, a ogni dimensione:
  deep          ResNet18 addestrata con la metrica di lineage sulle patch
  handcrafted   regressione logistica su FFT/DCT/colore/residuo (il pavimento)
  zero-shot     CLIP congelato, nessun addestramento, centroidi dai soli reali
                (misura quanta SEMANTICA resta nella finestra)

Se al ridursi della patch la curva zero-shot scende e quella deep tiene, i due
canali si separano: e' la dissociazione che l'articolo deve mostrare.

METRICA. Si usa la quota di immagini attribuite alla lineage CORRETTA, perche' e'
l'unica grandezza definita in modo identico per tutti e tre i metodi (l'AUC dei
tre script e' calcolata su insiemi diversi e non sarebbe confrontabile).
  pannello A: media sui tre generatori MAI VISTI (attgan, gdwct, stylegan3)
  pannello B: solo StyleGAN3 - l'evidenza forte, l'unico noise-GAN held-out:
              genera da rumore, quindi il segnale puo' venire solo dai dati di
              addestramento, non dall'immagine di input.

Uso:
  python -m src.eval.plot_patch_sweep --out report/patch_sweep
"""
import argparse
import json
from pathlib import Path

import numpy as np

# dimensione -> prefisso delle cartelle di report. 256 = immagine intera.
SIZES = [16, 32, 64, 128]
FULL = (256, "pm128")          # immagine intera: la policy di patch non si applica


def points(prefix):
    return [(s, f"{prefix}{s}") for s in SIZES] + [FULL]

# primi tre slot della palette categorica di riferimento: validati all-pairs
# in light mode. L'ordine e' fisso, mai ciclato.
COLORS = {"deep": "#2a78d6", "handcrafted": "#eb6834", "zero-shot": "#1baf7a"}
LABELS = {"deep": "deep (metrica addestrata)",
          "handcrafted": "handcrafted (pavimento)",
          "zero-shot": "zero-shot CLIP (semantica)"}
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
UNSEEN = ["attgan", "gdwct", "stylegan3"]


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def collect(root, prefix):
    out = []
    for size, tag in points(prefix):
        deep = load(root / f"lineage_{tag}_resnet18" / "report.json")
        sig = load(root / f"signatures_{tag}" / "report.json")
        cont = load(root / f"content_{tag}" / "report.json")
        if not (deep and sig and cont):
            print(f"  [salto {size}px] manca: "
                  + ", ".join(n for n, r in [("deep", deep), ("handcrafted", sig),
                                             ("zero-shot", cont)] if not r))
            continue

        # il pavimento e' la famiglia di descrittori migliore sui mai visti
        fam, fv = max(sig["floor"].items(), key=lambda kv: (kv[1]["auc_unseen"] or 0))

        def rate(src, c):
            if src is deep or src is cont:
                v = src["per_class"].get(c)
                return None if v is None else v["correct_rate"]
            return fv.get("per_class_correct", {}).get(c)

        row = dict(size=size, tag=tag, floor_family=fam,
                   deep_auc=deep["threshold"]["auc_unseen"],
                   floor_auc=fv["auc_unseen"], content_auc=cont["auc_fake_only"])
        for key, src in (("deep", deep), ("handcrafted", fv), ("zero-shot", cont)):
            vals = [rate(src, c) for c in UNSEEN]
            vals = [v for v in vals if v is not None]
            row[f"{key}_unseen"] = float(np.mean(vals)) if vals else None
            row[f"{key}_sg3"] = rate(src, "stylegan3")
        out.append(row)
    return out


def panel(ax, rows, suffix, title, subtitle):
    xs = [r["size"] for r in rows]
    ax.axhline(0.5, color=GRID, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(xs[0], 0.508, "caso", color=INK_SOFT, fontsize=8, va="bottom")

    ends = []
    for key in ("deep", "handcrafted", "zero-shot"):      # ordine fisso
        ys = [r[f"{key}_{suffix}"] for r in rows]
        ax.plot(xs, ys, color=COLORS[key], lw=2, marker="o", ms=8,
                markeredgecolor="white", markeredgewidth=1.5,
                label=LABELS[key], zorder=3, clip_on=False)
        ends.append([ys[-1], ys[-1], key])

    # etichette dirette in inchiostro (mai nel colore della serie: l'identita' la
    # porta la linea colorata accanto). Le posizioni vengono separate quando due
    # curve finiscono troppo vicine, altrimenti i numeri si sovrappongono.
    ends.sort(key=lambda e: e[0])
    MIN_GAP = 0.035
    for i in range(1, len(ends)):
        if ends[i][1] - ends[i - 1][1] < MIN_GAP:
            ends[i][1] = ends[i - 1][1] + MIN_GAP
    for val, ypos, key in ends:
        ax.annotate(f"{val*100:.0f}%", (xs[-1], val),
                    xytext=(xs[-1] * 1.12, ypos), textcoords="data",
                    color=INK, fontsize=9, va="center", fontweight="medium",
                    annotation_clip=False)

    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x}" if x != 256 else "256\n(intera)" for x in xs],
                       fontsize=9)
    ax.set_xlim(xs[0] * 0.85, xs[-1] * 1.30)
    ax.set_ylim(0.45, 1.03)
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.set_yticklabels([f"{round(v*100)}%" for v in np.arange(0.5, 1.01, 0.1)],
                       fontsize=9)
    ax.set_xlabel("dimensione della finestra di analisi (px)", fontsize=9.5,
                  color=INK_SOFT)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=14,
                 fontweight="semibold")
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9,
            color=INK_SOFT, va="bottom")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, length=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-root", default="report")
    ap.add_argument("--prefix", default="patch",
                    help="prefisso delle cartelle di report: patch (flat) | grid")
    ap.add_argument("--out", default="report/patch_sweep")
    args = ap.parse_args()
    root = Path(args.report_root)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rows = collect(root, args.prefix)
    if len(rows) < 2:
        raise SystemExit("servono almeno due punti dello sweep")
    print(f"Punti raccolti: {[r['size'] for r in rows]}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
    panel(axes[0], rows, "unseen",
          "A. Tutti i generatori mai visti",
          "media su AttGAN, GDWCT, StyleGAN3")
    panel(axes[1], rows, "sg3",
          "B. StyleGAN3 - evidenza forte",
          "noise-GAN mai visto: genera da rumore, nessun input reale")
    axes[0].set_ylabel("attribuito alla lineage corretta", fontsize=9.5,
                       color=INK_SOFT)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.02),
               labelcolor=INK)
    fig.suptitle("Rimpicciolendo la finestra si toglie il contenuto semantico: "
                 "la traccia di basso livello resta",
                 fontsize=12.5, color=INK, y=1.0, fontweight="semibold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(out / "patch_sweep.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    # tabella: obbligatoria come vista alternativa alla figura
    L = ["# Sweep sulla dimensione della patch", "",
         "Quota di immagini attribuite alla lineage corretta. `256` = immagine intera.",
         "Tutti e tre i metodi valutati sullo stesso dato, dataset a ricampionamento",
         "normalizzato (pm128).", "",
         "## StyleGAN3 - evidenza forte (pannello B)", "",
         "Unico noise-GAN held-out: genera da rumore, quindi il segnale puo' venire",
         "solo dai dati di addestramento.", "",
         "| px | deep | handcrafted | zero-shot | deep - zero-shot | famiglia pavimento |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        gap = (None if (r["deep_sg3"] is None or r["zero-shot_sg3"] is None)
               else r["deep_sg3"] - r["zero-shot_sg3"])
        L.append(f"| {r['size']} | {r['deep_sg3']*100:.1f}% | "
                 f"{r['handcrafted_sg3']*100:.1f}% | {r['zero-shot_sg3']*100:.1f}% | "
                 f"{'-' if gap is None else f'{gap*100:+.1f} pt'} | `{r['floor_family']}` |")
    L += ["", "## Tutti i generatori mai visti (pannello A)", "",
          "Media su AttGAN, GDWCT, StyleGAN3.", "",
          "| px | deep | handcrafted | zero-shot |", "|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['size']} | {r['deep_unseen']*100:.1f}% | "
                 f"{r['handcrafted_unseen']*100:.1f}% | {r['zero-shot_unseen']*100:.1f}% |")
    L += ["", "## Come si legge", "",
          "La colonna `deep - zero-shot` e' il punto: misura quanto il modello",
          "addestrato sa che la sola somiglianza semantica non sa. Sull'immagine",
          "intera e' vicina a zero - li' le due spiegazioni sono indistinguibili e",
          "il risultato non e' attribuibile a una traccia forense. Man mano che la",
          "finestra si stringe il divario si apre: e' la prova che esiste un canale",
          "di basso livello indipendente dal contenuto.", ""]
    (out / "patch_sweep.md").write_text("\n".join(L), encoding="utf-8")

    print("\n".join(L))
    print(f"\nFigura -> {out/'patch_sweep.png'}\nTabella -> {out/'patch_sweep.md'}")


if __name__ == "__main__":
    main()
