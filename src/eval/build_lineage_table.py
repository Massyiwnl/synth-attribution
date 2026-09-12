"""
build_lineage_table.py - Fase 8
Raccoglie i report di eval_lineage.py e class_signatures.py e costruisce la
tabella riassuntiva dell'articolo.

La tabella mette deliberatamente il modello deep ACCANTO al pavimento handcrafted
sullo stesso dataset, perche' il numero che conta non e' l'AUC del deep ma il suo
margine sopra un descrittore banale. Su ciascun dataset (originale / normalizzato)
si legge:

  AUC mai visti (deep)   quanto il modello attribuisce correttamente la lineage di
                         generatori mai visti in training
  AUC mai visti (floor)  quanto ci riesce una regressione logistica su FFT/DCT/colore
  delta                  la differenza. Se e' ~0, il deep non sta aggiungendo nulla
                         a una firma di ricampionamento.

Uso:
  python -m src.eval.build_lineage_table --report-root report --out report/lineage_summary.md
"""
import argparse
import json
from pathlib import Path


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def fmt(v, nd=4):
    return "-" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-root", default="report")
    ap.add_argument("--out", default="report/lineage_summary.md")
    args = ap.parse_args()
    root = Path(args.report_root)

    floors = {}
    for d in sorted(root.glob("signatures_*")):
        r = load(d / "report.json")
        if r:
            tag = d.name.replace("signatures_", "")
            best_fam = max(r["floor"].items(),
                           key=lambda kv: (kv[1]["auc_unseen"] or 0))
            floors[tag] = dict(best_family=best_fam[0],
                               auc_unseen=best_fam[1]["auc_unseen"],
                               auc_seen=best_fam[1]["auc_seen"],
                               per_family=r["floor"])

    # baseline ZERO-SHOT (nessun addestramento): quanto basta il solo contenuto
    content = {}
    for d in sorted(root.glob("content_*")):
        r = load(d / "report.json")
        if r:
            noise = [v["correct_rate"] for v in r["per_class"].values()
                     if v["kind"] == "noise"]
            content[d.name.replace("content_", "")] = dict(
                backbone=r["backbone"], auc=r["auc_fake_only"],
                noise=(sum(noise) / len(noise) if noise else None),
                per_class=r["per_class"])

    rows = []
    for d in sorted(root.glob("lineage_*")):
        r = load(d / "report.json")
        if not r:
            continue
        name = d.name[len("lineage_"):]
        # il prefisso identifica il REGIME sperimentale (raw / pm128 / patch64...)
        # e deve corrispondere al pavimento calcolato sullo stesso regime
        tag, _, rest = name.partition("_")
        if tag not in floors:                  # nomi senza prefisso -> dataset originale
            tag, rest = "raw", name
        backbone = rest
        t = r["threshold"]
        unseen = {c: v for c, v in r["per_class"].items() if not v["seen"]}
        noise = [v["correct_rate"] for v in unseen.values() if v["kind"] == "noise"]
        edit = [v["correct_rate"] for v in unseen.values() if v["kind"] == "editing"]
        rows.append(dict(dataset=tag, backbone=backbone,
                         auc_seen=t["auc_seen"], auc_unseen=t["auc_unseen"],
                         acc_unseen=t["acc_unseen"],
                         noise=(sum(noise) / len(noise) if noise else None),
                         edit=(sum(edit) / len(edit) if edit else None),
                         floor=(floors.get(tag) or {}).get("auc_unseen")))

    lines = ["# Fase 8 - Attribuzione dei dati di addestramento", ""]

    lines += ["## Pavimento handcrafted (regressione logistica su descrittori banali)", "",
              "AUC nel separare le lineage, allenando solo sulle architetture viste e",
              "testando sui generatori mai visti.", "",
              "| dataset | famiglia | AUC visti | AUC mai visti |", "|---|---|---|---|"]
    for tag, f in floors.items():
        for fam, v in f["per_family"].items():
            lines.append(f"| {tag} | {fam} | {fmt(v['auc_seen'])} | {fmt(v['auc_unseen'])} |")
    lines.append("")

    if content:
        lines += ["## Controllo del CONTENUTO: baseline zero-shot (nessun addestramento)", "",
                  "Feature grezze di un VLM congelato; i centroidi di lineage sono calcolati",
                  "SOLO sulle immagini reali. Misura quanto la sola somiglianza semantica",
                  "visibile spiega il risultato.", "",
                  "| dataset | backbone | AUC (soli generati) | noise-GAN corretti |",
                  "|---|---|---|---|"]
        for tag, c in content.items():
            lines.append(f"| {tag} | `{c['backbone']}` | {fmt(c['auc'])} | "
                         f"{fmt(c['noise'], 3)} |")
        lines.append("")

    lines += ["## Deep vs pavimento", "",
              "| dataset | backbone | AUC visti | AUC mai visti | acc mai visti | "
              "noise-GAN corretti | editing-GAN corretti | pavimento | delta |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["dataset"], x["backbone"])):
        delta = (None if (r["auc_unseen"] is None or r["floor"] is None)
                 else r["auc_unseen"] - r["floor"])
        lines.append(
            f"| {r['dataset']} | `{r['backbone']}` | {fmt(r['auc_seen'])} | "
            f"{fmt(r['auc_unseen'])} | {fmt(r['acc_unseen'], 3)} | "
            f"{fmt(r['noise'], 3)} | {fmt(r['edit'], 3)} | {fmt(r['floor'])} | "
            f"{'-' if delta is None else f'{delta:+.4f}'} |")
    lines += ["",
              "**Come si legge.** `delta` e' il margine del modello deep sopra un",
              "descrittore handcrafted banale. Un delta vicino a zero sul dataset",
              "originale significa che il risultato e' spiegato dalla catena di",
              "ricampionamento; il confronto fra la riga `raw` e la riga `pm128`",
              "(stessa catena per tutte le classi) dice quanta parte del segnale",
              "sopravvive alla rimozione del confound.",
              "",
              "`noise-GAN corretti` e' l'evidenza forte (generano da rumore: il segnale",
              "puo' venire solo dai dati di addestramento); `editing-GAN corretti` e'",
              "evidenza debole (il loro output contiene gia' i pixel dell'immagine reale",
              "di input).",
              "",
              "**Il confronto che conta.** Il modello addestrato va letto contro DUE",
              "riferimenti, non uno: il pavimento handcrafted (quanto basta una firma di",
              "basso livello) e il baseline zero-shot (quanto basta la somiglianza",
              "semantica visibile). Un risultato e' forense solo nella misura in cui",
              "supera ENTRAMBI.", ""]

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nTabella -> {out}")


if __name__ == "__main__":
    main()
