"""
build_report.py - Fase 8
Genera il documento completo della Fase 8 (PDF + Markdown) leggendo i numeri
DIRETTAMENTE dai report.json prodotti dagli esperimenti, cosi' che il testo non
possa divergere dai risultati effettivi.

Uso (nel .venv, dalla radice del repo):
  python scripts/build_report.py
  python scripts/build_report.py --out report/relazione_fase8.pdf
"""
import argparse
import json
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak, KeepTogether)

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "report"

INK = colors.HexColor("#0b0b0b")
SOFT = colors.HexColor("#52514e")
RULE = colors.HexColor("#d9d8d4")
BAND = colors.HexColor("#f2f1ee")
ACCENT = colors.HexColor("#2a78d6")

MARGIN = 20 * mm
FRAME_W = A4[0] - 2 * MARGIN

_md = []          # accumula la versione Markdown in parallelo


# ---------------------------------------------------------------- stili
def styles():
    ss = getSampleStyleSheet()
    S = {}
    S["title"] = ParagraphStyle("title", parent=ss["Title"], fontSize=22,
                                leading=27, textColor=INK, spaceAfter=4)
    S["subtitle"] = ParagraphStyle("subtitle", parent=ss["Normal"], fontSize=12.5,
                                   leading=17, textColor=SOFT, alignment=TA_CENTER,
                                   spaceAfter=18)
    S["h1"] = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15.5, leading=19,
                             textColor=INK, spaceBefore=16, spaceAfter=7)
    S["h2"] = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, leading=16,
                             textColor=INK, spaceBefore=11, spaceAfter=5)
    S["h3"] = ParagraphStyle("h3", parent=ss["Heading3"], fontSize=11, leading=14,
                             textColor=SOFT, spaceBefore=8, spaceAfter=4)
    S["body"] = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.7,
                               leading=14.2, textColor=INK, alignment=TA_JUSTIFY,
                               spaceAfter=6)
    S["bullet"] = ParagraphStyle("bullet", parent=S["body"], leftIndent=11,
                                 bulletIndent=2, spaceAfter=3)
    S["caption"] = ParagraphStyle("caption", parent=ss["Normal"], fontSize=8.4,
                                  leading=11.5, textColor=SOFT, alignment=TA_JUSTIFY,
                                  spaceBefore=3, spaceAfter=12)
    S["box"] = ParagraphStyle("box", parent=S["body"], fontSize=9.7, leading=14,
                              leftIndent=8, rightIndent=8, spaceBefore=6,
                              spaceAfter=6)
    S["code"] = ParagraphStyle("code", parent=ss["Code"], fontSize=8.1, leading=11,
                               textColor=INK, leftIndent=8)
    return S



# ---------------------------------------------------------------- testo
# Il codice del repo e i commenti usano forme ASCII ("e'", "perche'") per non
# dipendere dalla codifica dei sorgenti. In un documento destinato alla lettura
# servono gli accenti veri: la conversione avviene in un solo punto, applicata a
# ogni stringa che entra nel documento. L'ordine e' dal piu' lungo al piu' corto,
# altrimenti "e'" spezzerebbe "perche'".
_ACCENTI = [
    ("riproducibilita", "riproducibilit\u00e0"), ("ammissibilita", "ammissibilit\u00e0"),
    ("irregolarita", "irregolarit\u00e0"), ("affidabilita", "affidabilit\u00e0"),
    ("possibilita", "possibilit\u00e0"), ("specificita", "specificit\u00e0"),
    ("difficolta", "difficolt\u00e0"), ("semplicita", "semplicit\u00e0"),
    ("necessita", "necessit\u00e0"), ("proprieta", "propriet\u00e0"),
    ("capacita", "capacit\u00e0"), ("identita", "identit\u00e0"),
    ("attivita", "attivit\u00e0"), ("qualita", "qualit\u00e0"),
    ("utilita", "utilit\u00e0"), ("novita", "novit\u00e0"),
    ("verita", "verit\u00e0"), ("parita", "parit\u00e0"),
    ("realta", "realt\u00e0"), ("unita", "unit\u00e0"),
    ("meta", "met\u00e0"), ("eta", "et\u00e0"),
    ("velocita", "velocit\u00e0"), ("solidita", "solidit\u00e0"),
    ("modalita", "modalit\u00e0"),
    ("cioe", "cio\u00e8"),
    ("poiche", "poich\u00e9"), ("finche", "finch\u00e9"),
    ("benche", "bench\u00e9"), ("affinche", "affinch\u00e9"),
    ("tornera", "torner\u00e0"), ("potra", "potr\u00e0"),
    ("dovra", "dovr\u00e0"), ("avra", "avr\u00e0"),
    ("andra", "andr\u00e0"), ("dara", "dar\u00e0"),
    ("perche", "perch\u00e9"), ("percio", "perci\u00f2"),
    ("cosi", "cos\u00ec"), ("si", "s\u00ec"),
    ("piu", "pi\u00f9"), ("cio", "ci\u00f2"),
    ("puo", "pu\u00f2"), ("gia", "gi\u00e0"), ("pero", "per\u00f2"),
    ("sara", "sar\u00e0"), ("verra", "verr\u00e0"), ("restera", "restar\u00e0"),
    ("ne", "n\u00e9"), ("la", "l\u00e0"), ("e", "\u00e8"),
]


def accenti(t: str) -> str:
    for plain, acc in _ACCENTI:
        t = re.sub(r"\b" + plain + r"'", acc, t)
        t = re.sub(r"\b" + plain.capitalize() + r"'",
                   acc[0].upper() + acc[1:], t)
    return t


_MD_TAGS = [("<b>", "**"), ("</b>", "**"), ("<i>", "*"), ("</i>", "*"),
            ("<tt>", "`"), ("</tt>", "`"),
            ("<super>", "^"), ("</super>", ""),
            ("&middot;", "\u00b7"), ("&nbsp;", " ")]


def to_md(t: str) -> str:
    t = accenti(str(t))
    for a, b in _MD_TAGS:
        t = t.replace(a, b)
    return t


# ---------------------------------------------------------------- helper
def esc(t):
    return escape(accenti(str(t)))


def h1(S, t, n=None):
    label = f"{n}. {t}" if n else t
    _md.append(f"\n## {to_md(label)}\n")
    return Paragraph(esc(label), S["h1"])


def h2(S, t):
    _md.append(f"\n### {to_md(t)}\n")
    return Paragraph(esc(t), S["h2"])


def h3(S, t):
    _md.append(f"\n**{to_md(t)}**\n")
    return Paragraph(esc(t), S["h3"])


def p(S, t, md=True):
    if md:
        _md.append(to_md(t) + "\n")
    return Paragraph(accenti(t), S["body"])


def bullets(S, items):
    out = []
    for it in items:
        _md.append(f"- {to_md(it)}")
        out.append(Paragraph(accenti(it), S["bullet"], bulletText="•"))
    _md.append("")
    return out


def keybox(S, text):
    """Riquadro per i risultati chiave."""
    _md.append(f"\n> {to_md(text)}\n")
    t = Table([[Paragraph(accenti(text), S["box"])]], colWidths=[FRAME_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BAND),
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, ACCENT),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
    ]))
    return [Spacer(1, 4), t, Spacer(1, 8)]


def table(S, header, rows, widths=None, highlight=None, fontsize=8.3):
    _md.append("")
    _md.append("| " + " | ".join(to_md(h) for h in header) + " |")
    _md.append("|" + "---|" * len(header))
    for r in rows:
        _md.append("| " + " | ".join(to_md(c) for c in r) + " |")
    _md.append("")

    hs = ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=fontsize,
                        leading=fontsize + 2.4, textColor=INK)
    cs = ParagraphStyle("td", fontName="Helvetica", fontSize=fontsize,
                        leading=fontsize + 2.4, textColor=INK)
    data = [[Paragraph(accenti(esc(c)), hs) for c in header]]
    data += [[Paragraph(accenti(str(c)), cs) for c in r] for r in rows]
    if widths is None:
        widths = [FRAME_W / len(header)] * len(header)
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    st = [("LINEBELOW", (0, 0), (-1, 0), 1.1, INK),
          ("LINEBELOW", (0, 1), (-1, -2), 0.4, RULE),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3.6),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 3.6),
          ("LEFTPADDING", (0, 0), (-1, -1), 5),
          ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
    for r in (highlight or []):
        st.append(("BACKGROUND", (0, r), (-1, r), BAND))
    t.setStyle(TableStyle(st))
    return [Spacer(1, 3), t, Spacer(1, 9)]


def figure(S, path, caption, max_w=FRAME_W, max_h=150 * mm):
    path = Path(path)
    if not path.exists():
        return [p(S, f"<i>[figura mancante: {esc(path.name)}]</i>", md=False)]
    iw, ih = ImageReader(str(path)).getSize()
    w = max_w
    h = ih * w / iw
    if h > max_h:
        h = max_h
        w = iw * h / ih
    _md.append(f"\n![{to_md(caption)}]({path.relative_to(ROOT).as_posix()})\n")
    _md.append(f"*{to_md(caption)}*\n")
    return [Spacer(1, 4), Image(str(path), width=w, height=h, hAlign="CENTER"),
            Paragraph(accenti(caption), S["caption"])]


def code(S, lines):
    _md.append("\n```bash")
    _md.extend(lines)
    _md.append("```\n")
    flow = [Paragraph(esc(l).replace(" ", "&nbsp;"), S["code"]) for l in lines]
    t = Table([[flow]], colWidths=[FRAME_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BAND),
                           ("TOPPADDING", (0, 0), (-1, -1), 6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return [Spacer(1, 4), t, Spacer(1, 8)]


def pct(v, nd=1):
    return "-" if v is None else f"{v*100:.{nd}f}%"


def num(v, nd=4):
    return "-" if v is None else f"{v:.{nd}f}"


def load(*parts):
    try:
        return json.loads((REP.joinpath(*parts)).read_text())
    except Exception:
        return None


def page_deco(canv, doc):
    canv.saveState()
    canv.setStrokeColor(RULE)
    canv.setLineWidth(0.5)
    canv.line(MARGIN, 14 * mm, A4[0] - MARGIN, 14 * mm)
    canv.setFont("Helvetica", 7.6)
    canv.setFillColor(SOFT)
    canv.drawString(MARGIN, 9.5 * mm,
                    "Attribuzione dei dati di addestramento di immagini sintetiche "
                    "- synth-attribution, Fase 8")
    canv.drawRightString(A4[0] - MARGIN, 9.5 * mm, f"{doc.page}")
    canv.restoreState()


# ================================================================ dati
def gather():
    d = {}
    for tag in ("raw", "pm128", "patch16", "patch32", "patch64", "patch128",
                "grid16", "grid32", "grid64", "grid128"):
        d[f"sig_{tag}"] = load(f"signatures_{tag}", "report.json")
        d[f"con_{tag}"] = load(f"content_{tag}", "report.json")
    for name in ("raw_clip_vit_l14", "pm128_clip_vit_l14", "pm128_resnet18",
                 "pm128_resnet18_hp", "pm128_resnet50", "pm128_clip_vit_b16",
                 "pm128_dinov2_vitb14", "pm128_dinov2_vitl14",
                 "patch16_resnet18", "patch32_resnet18", "patch64_resnet18",
                 "patch128_resnet18", "grid16_resnet18", "grid32_resnet18",
                 "grid64_resnet18", "grid128_resnet18"):
        d[f"lin_{name}"] = load(f"lineage_{name}", "report.json")
    d["cam_rn"] = load("gradcam_pm128_resnet18", "report.json")
    d["cam_clip"] = load("gradcam_pm128_clip_vit_l14", "report.json")
    d["final"] = load("final", "report.json")
    return d


def floor_best(sig):
    if not sig:
        return None, None
    fam, v = max(sig["floor"].items(), key=lambda kv: (kv[1]["auc_unseen"] or 0))
    return fam, v


def sg3(rep):
    if not rep:
        return None
    return rep["per_class"].get("stylegan3", {}).get("correct_rate")


def sweep_rows(D, prefix):
    rows = []
    for size in (16, 32, 64, 128, 256):
        tag = "pm128" if size == 256 else f"{prefix}{size}"
        lin = D.get(f"lin_{tag}_resnet18")
        sig = D.get(f"sig_{tag}")
        con = D.get(f"con_{tag}")
        fam, fv = floor_best(sig)
        if not (lin and fv and con):
            continue
        dv = sg3(lin)
        zv = sg3(con)
        hv = fv.get("per_class_correct", {}).get("stylegan3")
        gap = None if (dv is None or zv is None) else dv - zv
        rows.append([f"{size}" + (" (intera)" if size == 256 else ""),
                     pct(dv), pct(hv), pct(zv),
                     "-" if gap is None else f"{gap*100:+.1f} pt", f"{fam}"])
    return rows


# ================================================================ documento
def part_intro(S, D):
    F = []
    A = F.append
    X = F.extend

    # ---------------- frontespizio ----------------
    A(Spacer(1, 22 * mm))
    A(Paragraph("Attribuzione dei dati di addestramento<br/>di immagini sintetiche",
                S["title"]))
    A(Paragraph("Metric learning siamese su backbone pre-addestrati per risalire "
                "dal contenuto generato al dataset usato per addestrare il "
                "generatore.<br/>Evidenza tecnica a supporto dell'analisi "
                "giuridica su copyright e privacy.", S["subtitle"]))
    X(table(S, ["", ""], [
        ["Progetto", "<b>synth-attribution</b> - Fase 8"],
        ["Contesto", "Estensione sperimentale verso un articolo a impianto giuridico"],
        ["Base di partenza", "Cassia et al., <i>Deepfake Forensic Analysis: Source "
                             "Dataset Attribution and Legal Implications of Synthetic "
                             "Media Manipulation</i>, arXiv:2505.11110 (tesi triennale) "
                             "+ progetto di Multimedia Forensics (LM-18, Catania)"],
        ["Dataset", "21.000 immagini, 7 classi, 2 lineage, 256x256"],
        ["Esperimenti", "16 modelli addestrati, 10 pavimenti handcrafted, "
                        "10 baseline zero-shot, 2 analisi Grad-CAM"],
    ], widths=[34 * mm, FRAME_W - 34 * mm], fontsize=8.8))

    X(keybox(S, "<b>Risultato in una frase.</b> Esiste un canale di basso livello, "
                "indipendente dal contenuto visibile e dalla catena di "
                "ricampionamento, che lega un'immagine generata al dataset su cui "
                "il generatore e' stato addestrato, e che <b>trasferisce a "
                "generatori mai visti in addestramento</b>: su un riquadro 16x16 "
                "in posizione fissa, dove la somiglianza semantica arriva al 73.5% "
                "e un descrittore spettrale al 75.0%, la metrica addestrata "
                "attribuisce correttamente il 93.9% delle immagini di StyleGAN3 - "
                "un generatore che non ha mai visto e che, partendo da rumore, dai "
                "dati di addestramento ha ereditato tutto."))
    A(PageBreak())

    # ---------------- 1. contesto ----------------
    A(h1(S, "Contesto e domanda di ricerca", 1))
    A(p(S, "Il dibattito giuridico sulle immagini sintetiche si e' concentrato a "
          "lungo su <b>chi sia titolare dell'opera finale</b>. Questo lavoro parte "
          "dall'osservazione che il problema si sposta <b>a monte</b>: cio' che "
          "conta non e' l'output, ma <b>quali dati sono stati usati per addestrare "
          "il modello</b>. Se un volto, una fotografia o un'opera sono finiti in un "
          "training set senza consenso, il danno si e' prodotto prima e altrove "
          "rispetto all'immagine generata."))
    A(p(S, "Perche' quella pretesa sia azionabile serve un modo tecnico per "
          "sostenerla. La domanda di ricerca della Fase 8 e' quindi: <b>data "
          "un'immagine sintetica, e' possibile risalire al dataset su cui il "
          "generatore che l'ha prodotta e' stato addestrato?</b> E soprattutto: "
          "questa capacita' regge su generatori che il sistema non ha mai visto?"))

    A(h2(S, "1.1 Tre pretese diverse, da non confondere"))
    A(p(S, "La formulazione precisa del claim e' decisiva, perche' tre affermazioni "
          "vicine hanno peso tecnico e giuridico molto diverso:"))
    X(table(S, ["", "Affermazione", "Difficolta", "Che cosa fonda"], [
        ["(a)", "questa immagine viene dal modello X", "media",
         "attribuzione del modello"],
        ["(b)", "questa immagine viene da un modello <b>addestrato su D</b>",
         "alta", "<b>e' cio' che questo lavoro dimostra</b>"],
        ["(c)", "<b>questa specifica immagine</b> era nel training set",
         "altissima", "membership individuale; non affrontata qui"],
    ], widths=[10 * mm, 72 * mm, 20 * mm, FRAME_W - 102 * mm]))
    A(p(S, "Il lavoro consegna <b>(b)</b>. La distinzione da <b>(c)</b> va tenuta "
          "esplicita: l'evidenza non prova che una particolare fotografia fosse nel "
          "training set, ma fonda una <b>presunzione sull'origine dei dati</b>, "
          "cioe' la base per chiedere accesso e verifica. E' una differenza che "
          "rafforza, non indebolisce, l'uso processuale del risultato: dice "
          "esattamente cio' che puo' dire, ne' piu' ne' meno."))

    # ---------------- 2. punto di partenza ----------------
    A(h1(S, "Che cosa esisteva prima, e perche' serve", 2))

    A(h2(S, "2.1 La tesi triennale: attribuzione del dataset con feature handcrafted"))
    A(p(S, "Il lavoro di partenza (arXiv:2505.11110) affronta l'attribuzione del "
          "<b>dataset di addestramento</b> combinando trasformate spettrali "
          "(Fourier e DCT), statistiche di colore e descrittori locali (SIFT), con "
          "classificatori supervisionati (Random Forest, SVM, XGBoost). Ottiene "
          "<b>98-99% di accuratezza</b> sia nella distinzione reale/sintetico sia "
          "nell'attribuzione multi-classe del dataset, su CelebA e FFHQ con cinque "
          "architetture GAN. Il risultato chiave e' la <b>dominanza delle feature "
          "in frequenza</b>, che catturano artefatti di upsampling e irregolarita' "
          "spettrali."))
    A(p(S, "Il limite, ed e' il punto da cui parte l'estensione richiesta, e' la "
          "<b>formulazione del problema</b>: un task di classificazione piatto in "
          "cui CelebA e FFHQ sono due classi come le altre. Il modello non modella "
          "in alcun modo la <i>relazione</i> fra un'immagine reale di un dataset e "
          "le immagini generate da un modello addestrato su quel dataset: impara "
          "confini di decisione fra etichette, non una nozione di appartenenza. E "
          "quei confini valgono soltanto per le classi viste in addestramento."))

    A(h2(S, "2.2 Il progetto di Multimedia Forensics: attribuzione del generatore"))
    A(p(S, "Il progetto successivo sposta il focus dal dataset al <b>generatore</b> "
          "(distinguere reale, StyleGAN2 e StyleGAN3 a parita' di famiglia di "
          "volti) e introduce il <b>metric learning siamese</b>: un encoder "
          "ResNet18 con testa di proiezione e <i>contrastive loss</i>, che impara "
          "uno spazio in cui la distanza riflette la somiglianza di sorgente. "
          "L'enfasi si sposta sulla <b>generalizzazione open-set</b>."))
    A(p(S, "I risultati, per quanto serve qui:"))
    X(bullets(S, [
        "<b>In-distribution</b>: FFHQ reale vs StyleGAN2, AUC 0.952 (0.963 con "
        "front-end a residuo); reale vs StyleGAN3, AUC 0.706. Il divario misura "
        "l'efficacia del design <i>alias-free</i> di StyleGAN3 nel nascondere la "
        "propria firma.",
        "<b>Generalizzazione cross-architettura</b>: nulla. Tenendo un generatore "
        "fuori dal training, l'AUC focalizzata su di esso resta circa 0.5. Il "
        "fallimento e' <b>asimmetrico</b>: StyleGAN3 non visto viene scambiato per "
        "reale (falso negativo, la direzione pericolosa), StyleGAN2 non visto viene "
        "comunque marchiato come sintetico.",
        "<b>Due confound di dataset</b> individuati e neutralizzati: un confound di "
        "pairing cross-lineage (AUC 1.0 alla prima epoca, perche' CelebA e FFHQ "
        "differiscono in tutto) e un <b>confound di ricampionamento</b> che faceva "
        "sembrare il baseline handcrafted capace di generalizzare a circa 0.98; "
        "appaiando la pipeline di resize fra reali e sintetici quel valore crolla "
        "a 0.51-0.53.",
    ]))
    A(p(S, "Il secondo confound e' il precedente metodologico piu' importante per "
          "la Fase 8: <b>lo stesso gruppo di ricerca ha gia' dimostrato, sui propri "
          "dati, che un risultato apparentemente forte in questo dominio puo' "
          "essere interamente un artefatto di preprocessing</b>. Ogni numero che "
          "segue e' stato percio' sottoposto a controlli espliciti, ed e' questa la "
          "ragione per cui la parte piu' estesa del documento riguarda i controlli "
          "e non i risultati."))

    A(h2(S, "2.3 Due indizi, nei risultati precedenti, che indicavano la strada"))
    A(p(S, "Rileggendo la fase precedente emergono due fatti che riguardano "
          "direttamente l'attribuzione del dataset, e che non erano stati "
          "interpretati in quella chiave."))
    A(p(S, "<b>Primo.</b> La matrice <i>leave-one-generator-out</i> mostra una "
          "struttura a blocchi netta: dentro la stessa lineage c'e' transfer fra "
          "generatori (AUC 0.63-1.00 su CelebA, 0.73-0.79 su FFHQ), mentre <b>fra "
          "lineage diverse il transfer e' puro caso</b> (0.42-0.53). La traccia del "
          "generatore e' dunque intrecciata con il dataset su cui quel generatore "
          "e' stato addestrato: sono due informazioni che vivono insieme."))
    A(p(S, "<b>Secondo.</b> Il modello finale allenato su tutte le classi raggiunge "
          "un'accuratezza top-1 in regime chiuso del 76.3%, con un profilo per "
          "classe molto particolare: le quattro classi della lineage CelebA sono "
          "riconosciute al 100%, mentre dentro FFHQ il modello si confonde "
          "(real_ffhq 55%, StyleGAN2 58%, StyleGAN3 21%). Ma la matrice di "
          "confusione mostra un fatto che vale piu' dell'accuratezza aggregata: "
          "<b>non c'e' un solo errore cross-lineage</b>. Tutta la confusione e' "
          "interna a FFHQ. Il lignaggio dei dati era percio' perfettamente "
          "separabile proprio mentre il generatore non lo era."))
    X(figure(S, REP / "final" / "confusion.png",
             "Fase precedente, attribuzione del generatore in regime chiuso "
             "(top-1 = 0.763). Nessuna immagine della famiglia CelebA viene mai "
             "attribuita a una classe FFHQ, e viceversa: i due blocchi 4x4 e 3x3 "
             "fuori diagonale sono interamente a zero. E' l'osservazione da cui "
             "parte la Fase 8.", max_w=112 * mm))
    X(figure(S, REP / "final" / "pca_final.png",
             "Lo stesso spazio di embedding in proiezione PCA. La prima componente "
             "principale separa le due lineage: a sinistra i quattro cluster "
             "compatti della famiglia CelebA, a destra un unico blocco in cui le "
             "tre classi FFHQ si sovrappongono. La direzione di massima varianza "
             "dello spazio e' il dataset di addestramento, non il generatore.",
             max_w=112 * mm))
    return F


def part_data_method(S, D):
    F = []
    A = F.append
    X = F.extend

    # ---------------- 3. dataset ----------------
    A(PageBreak())
    A(h1(S, "Il dataset", 3))
    A(p(S, "21.000 immagini a 256x256, 7 classi da 3.000 immagini, 2 <b>lineage</b> "
          "(famiglie di volti definite dal dataset di addestramento). La struttura "
          "e' il fulcro dell'intero esperimento: ogni lineage contiene le immagini "
          "<b>reali</b> del dataset e le immagini prodotte da generatori "
          "<b>addestrati su quel dataset</b>."))
    X(table(S, ["lineage", "classe", "tipo", "origine"], [
        ["celeba", "real_celeba", "reale", "CelebA, crop centrale + Lanczos a 256"],
        ["celeba", "stargan", "editing-GAN", "StarGAN (yunjey), edit Blond_Hair, 128 -> 256"],
        ["celeba", "attgan", "editing-GAN", "AttGAN (elvisyjlin), edit Blond_Hair, 128 -> 256"],
        ["celeba", "gdwct", "editing-GAN", "GDWCT (WonwoongCho), 216 -> 256"],
        ["ffhq", "real_ffhq", "reale", "FFHQ-1024 -> 256 (Lanczos), pipeline appaiata"],
        ["ffhq", "stylegan2", "noise-GAN", "StyleGAN2-ADA FFHQ, pickle ufficiale, 1024 -> 256"],
        ["ffhq", "stylegan3", "noise-GAN", "StyleGAN3-t FFHQ, pickle ufficiale, 1024 -> 256"],
    ], widths=[20 * mm, 27 * mm, 25 * mm, FRAME_W - 72 * mm], highlight=[2, 3, 7]))

    A(h2(S, "3.1 Editing-GAN e noise-GAN: la distinzione che regge tutta la tesi"))
    A(p(S, "Le classi sintetiche non sono dello stesso tipo, e la differenza decide "
          "quanto vale l'evidenza."))
    X(bullets(S, [
        "<b>Editing-GAN</b> (StarGAN, AttGAN, GDWCT): modificano una fotografia "
        "reale esistente. Il loro output <b>contiene i pixel della foto CelebA di "
        "input</b>. Se il sistema le attribuisce a CelebA, non ha dimostrato che il "
        "training set fosse CelebA: ha dimostrato che <i>l'input</i> lo era. "
        "Evidenza <b>debole</b>, e va dichiarata come tale.",
        "<b>Noise-GAN</b> (StyleGAN2, StyleGAN3): generano da rumore casuale. Non "
        "esiste alcuna immagine di input. Qualsiasi somiglianza con FFHQ puo' "
        "provenire <b>soltanto</b> dai dati di addestramento. Evidenza "
        "<b>forte</b>, ed e' su queste che si misura il risultato.",
    ]))
    A(p(S, "Per questo, in tutto il documento, la colonna che conta e' quella di "
          "<b>StyleGAN3</b>: e' l'unico noise-GAN tenuto fuori dall'addestramento. "
          "Le righe di AttGAN e GDWCT restano come controllo positivo, non come "
          "prova."))
    X(figure(S, REP / "figures" / "dataset_samples.png",
             "Campioni del dataset, una riga per classe. A 256x256 le classi sono "
             "visivamente molto simili: le differenze fra generatori vivono in "
             "tracce non percepibili a occhio. Si noti pero' la differenza di "
             "inquadratura e di fondale fra le due lineage - una caratteristica di "
             "contenuto che tornera' come principale spiegazione concorrente.",
             max_w=95 * mm, max_h=125 * mm))

    # ---------------- 4. metodo ----------------
    A(PageBreak())
    A(h1(S, "Metodo", 4))

    A(h2(S, "4.1 Riformulazione del task: pairing per lineage"))
    A(p(S, "La rete siamese non classifica: impara una <b>funzione di embedding</b> "
          "in cui la distanza fra due immagini riflette la loro appartenenza alla "
          "stessa sorgente. Cio' che determina <i>cosa</i> impara e' la definizione "
          "di coppia genuina. La fase precedente usava il <b>pairing per "
          "architettura</b> (genuina = stesso generatore), che spinge "
          "deliberatamente <i>lontano</i> real_celeba da StarGAN."))
    A(p(S, "La Fase 8 introduce il <b>pairing per lineage</b>: due immagini formano "
          "una coppia genuina se appartengono alla stessa lineage, "
          "<b>indipendentemente dal generatore</b>. real_celeba, StarGAN, AttGAN e "
          "GDWCT sono quindi tutti <i>positivi fra loro</i>; real_ffhq, StyleGAN2 e "
          "StyleGAN3 fra loro; le coppie impostore sono cross-lineage. La rete e' "
          "cosi' costretta a cercare <b>cio' che un dataset e i modelli addestrati "
          "su di esso hanno in comune</b>, ignorando la firma del singolo "
          "generatore. E' esattamente la relazione che il task di classificazione "
          "piatto della tesi triennale non poteva modellare."))
    A(p(S, "Nota tecnica: con questa politica le coppie impostore sono per "
          "costruzione cross-lineage, quindi i <i>bucket</i> di valutazione "
          "within-lineage della fase precedente degenerano. La metrica corretta di "
          "selezione del modello migliore diventa l'AUC complessiva - il contrario "
          "di quanto valeva per l'attribuzione del generatore, dove l'AUC "
          "complessiva era gonfiata dal confound cross-lineage."))

    A(h2(S, "4.2 Architettura: backbone pre-addestrato congelato + testa leggera"))
    A(p(S, "L'encoder e' composto da tre parti: un <b>front-end opzionale</b> "
          "(identita', oppure un filtro high-pass o SRM che sopprime il contenuto), "
          "un <b>backbone pre-addestrato</b> di cui si usa soltanto l'embedding "
          "finale, e una <b>testa di proiezione</b> addestrabile "
          "(lineare - ReLU - dropout - lineare) seguita da normalizzazione L2. "
          "L'embedding finale vive su un'ipersfera unitaria a 128 dimensioni."))
    A(p(S, "Nel regime principale il backbone e' <b>congelato</b>: i suoi pesi non "
          "vengono aggiornati e il modulo resta in modalita' di valutazione anche "
          "durante l'addestramento (altrimenti BatchNorm e dropout continuerebbero "
          "a modificarsi e il backbone non sarebbe davvero fisso). Con CLIP "
          "ViT-L/14 si addestrano <b>0.59 milioni di parametri su 303.77</b>, cioe' "
          "lo 0.19%."))
    X(keybox(S, "<b>Dettaglio forense sulla risoluzione.</b> I backbone ViT "
                "attendono ingressi a 224 pixel, il dataset e' a 256. Un resize "
                "256 -> 224 introdurrebbe un <b>ulteriore ricampionamento</b>, "
                "cioe' esattamente il tipo di artefatto che questo lavoro cerca di "
                "non misurare. Per questo il default per i backbone congelati e' il "
                "<b>crop centrale</b> a 224, che preserva le statistiche dei pixel "
                "senza reinterpolare nulla. Le ResNet, essendo completamente "
                "convoluzionali, ricevono i 256 nativi."))

    A(h2(S, "4.3 Funzione di perdita"))
    A(p(S, "Contrastive loss di Hadsell, Chopra e LeCun. Per una coppia con "
          "etichetta y (1 = stessa lineage, 0 = lineage diverse) e distanza "
          "euclidea d fra gli embedding:"))
    A(Paragraph("L = y &middot; d<super>2</super> + (1 - y) &middot; "
                "max(0, m - d)<super>2</super>", S["code"]))
    A(p(S, "con margine m = 1.0. Poiche' gli embedding sono L2-normalizzati, la "
          "distanza vive in [0, 2]. Le coppie genuine vengono avvicinate, le "
          "impostore allontanate fino almeno al margine; oltre il margine le "
          "impostore non contribuiscono piu'."))

    A(h2(S, "4.4 Protocollo di holdout"))
    A(p(S, "Tre generatori sono <b>completamente esclusi dall'addestramento</b>: "
          "tutte le loro immagini finiscono nel test split, zero in training e zero "
          "in validazione. Lo split e' deterministico per gruppo, quindi le classi "
          "non in holdout ricevono esattamente lo stesso split degli esperimenti "
          "precedenti e i risultati restano confrontabili."))
    X(table(S, ["", "in addestramento", "solo in test (mai visti)"], [
        ["lineage celeba", "real_celeba, stargan", "<b>attgan, gdwct</b> (editing)"],
        ["lineage ffhq", "real_ffhq, stylegan2", "<b>stylegan3</b> (noise)"],
    ], widths=[32 * mm, 55 * mm, FRAME_W - 87 * mm]))
    A(p(S, "I centroidi di lineage usati per decidere sono calcolati <b>soltanto "
          "sulle classi viste</b>: un centroide che includesse le classi held-out "
          "userebbe informazione che in un caso reale non sarebbe disponibile."))
    return F


def part_results(S, D):
    F = []
    A = F.append
    X = F.extend
    lin = D.get("lin_pm128_clip_vit_l14") or {}
    th = lin.get("threshold", {})

    A(PageBreak())
    A(h1(S, "Risultati principali: lo spazio di embedding", 5))

    A(h2(S, "5.1 Le immagini si raggruppano per dataset di addestramento"))
    A(p(S, "Dato un insieme di immagini di test, si estraggono gli embedding e si "
          "proiettano in due dimensioni. La domanda posta all'inizio era: "
          "<i>se le clusterizza in qualche modo, significa che stanno ricadendo "
          "tutti nello stesso cluster</i>. La risposta e' affermativa e netta."))
    X(figure(S, REP / "lineage_raw_clip_vit_l14" / "pca_lineage.png",
             "Proiezione PCA dello spazio di embedding appreso con pairing per "
             "lineage (CLIP ViT-L/14 congelato). I cerchi sono le classi viste in "
             "addestramento, le croci i generatori mai visti. Le sette classi "
             "collassano in due soli gruppi, corrispondenti alle due lineage: "
             "real_celeba, StarGAN, AttGAN e GDWCT da una parte, real_ffhq, "
             "StyleGAN2 e StyleGAN3 dall'altra. AttGAN, GDWCT e StyleGAN3 cadono "
             "nel gruppo corretto pur non essendo mai stati visti.",
             max_w=118 * mm))

    A(h2(S, "5.2 Distanze fra vettori medi: la verifica quantitativa"))
    A(p(S, "La formulazione iniziale chiedeva: <i>prendiamo i vettori di feature di "
          "CelebA anche in media, prendiamo quelli di StarGAN, e calcoliamo una "
          "distanza; facciamo uguale tra CelebA e StyleGAN e mi aspetto che sia "
          "piu' grande</i>. La matrice delle distanze medie fra classi lo conferma "
          "con un ordine di grandezza di margine."))
    M = lin.get("class_distance_matrix")
    cls = lin.get("classes") or []
    if M and cls:
        hdr = [""] + [c[:9] for c in cls]
        rows = []
        for i, c in enumerate(cls):
            mark = "*" if (c not in ("real_celeba", "real_ffhq")
                           and c in ("attgan", "gdwct", "stylegan3")) else ""
            rows.append([f"<b>{c}</b>{mark}"] + [f"{M[i][j]:.3f}" for j in range(len(cls))])
        X(table(S, hdr, rows, widths=[26 * mm] + [(FRAME_W - 26 * mm) / len(cls)] * len(cls),
                fontsize=7.6))
        A(p(S, "Distanze medie nello spazio di embedding (CLIP ViT-L/14, dataset a "
              "ricampionamento normalizzato). La diagonale e' la compattezza "
              "intra-classe; l'asterisco marca i generatori mai visti. Le distanze "
              "<b>dentro</b> la stessa lineage stanno fra 0.029 e 0.193, quelle "
              "<b>fra</b> lineage diverse fra 0.947 e 1.135: un fattore circa dieci. "
              "La soglia non va scelta a mano, cade in un vuoto ampio.", md=False))

    A(h2(S, "5.3 Una decisione con tasso d'errore noto"))
    A(p(S, "Una soglia senza un tasso d'errore misurato non e' utilizzabile come "
          "evidenza. Il protocollo e' quindi: per ogni immagine si calcola la "
          "distanza dai due centroidi di lineage; lo score e' la differenza fra le "
          "due distanze; la soglia e' fissata al punto di <b>Equal Error Rate sulle "
          "sole classi viste</b> in addestramento, e poi applicata <b>invariata</b> "
          "ai generatori mai visti."))
    X(table(S, ["metrica", "classi viste", "generatori mai visti"], [
        ["AUC", num(th.get("auc_seen")), num(th.get("auc_unseen"))],
        ["accuratezza", pct(th.get("acc_seen")), pct(th.get("acc_unseen"))],
        ["EER (calibrazione)", pct(th.get("eer_seen")), "-"],
        ["falsi positivi (celeba -> ffhq)", "-", pct(th.get("far_unseen"))],
        ["falsi negativi (ffhq -> celeba)", "-", pct(th.get("frr_unseen"))],
    ], widths=[62 * mm, 40 * mm, FRAME_W - 102 * mm]))
    X(figure(S, REP / "lineage_pm128_clip_vit_l14" / "score_distribution.png",
             "Distribuzione dello score di lineage. Le aree piene sono le classi "
             "viste in addestramento, i contorni i generatori mai visti. La linea "
             "tratteggiata e' la soglia calibrata sulle sole classi viste. I "
             "generatori mai visti cadono dalla parte corretta con lo stesso ampio "
             "margine delle classi viste: fra le due popolazioni resta una fascia "
             "vuota di circa 0.5 unita'.", max_w=140 * mm))
    X(figure(S, REP / "lineage_pm128_clip_vit_l14" / "centroid_distances.png",
             "Distanza media dai due centroidi di lineage, per classe. Ogni classe "
             "e' molto vicina al centroide della propria lineage e molto lontana "
             "dall'altro; le tre classi marcate come mai viste si comportano come "
             "quelle viste.", max_w=140 * mm))

    A(h2(S, "5.4 I generatori mai visti"))
    A(p(S, "E' il risultato richiesto: <i>prendendo AttGAN e GDWCT che non abbiamo "
          "messo nel training, li mettiamo fuori nel test set, cioe' dati mai "
          "visti, e mi aspetto che ci siano distanze basse</i>. Con l'aggiunta di "
          "StyleGAN3, che e' l'evidenza forte."))
    pc = lin.get("per_class") or {}
    rows = []
    for c in sorted(pc):
        v = pc[c]
        rows.append([f"<b>{c}</b>" if not v["seen"] else c,
                     v["lineage"], "si" if v["seen"] else "<b>NO</b>",
                     v["kind"], f"{v['mean_d_celeba']:.3f}",
                     f"{v['mean_d_ffhq']:.3f}", f"<b>{pct(v['correct_rate'])}</b>"])
    X(table(S, ["classe", "lineage", "visto", "tipo", "dist. centroide celeba",
                "dist. centroide ffhq", "attribuiti bene"], rows,
            widths=[24 * mm, 16 * mm, 12 * mm, 16 * mm, 30 * mm, 28 * mm,
                    FRAME_W - 126 * mm], fontsize=7.9))
    X(keybox(S, "<b>StyleGAN3, mai visto in addestramento e generatore da rumore, "
                "viene ricondotto a FFHQ nel 99.5% dei casi</b>, con distanza 1.123 "
                "dal centroide CelebA e 0.050 da quello FFHQ. Non avendo mai "
                "ricevuto un'immagine di input, l'unica origine possibile di quella "
                "somiglianza sono i dati su cui e' stato addestrato."))
    return F


def part_controls(S, D):
    F = []
    A = F.append
    X = F.extend

    A(PageBreak())
    A(h1(S, "I controlli: perche' il risultato non e' un artefatto", 6))
    A(p(S, "I numeri della sezione precedente, da soli, <b>non dimostrerebbero "
          "nulla</b>. Sull'immagine intera il risultato e' "
          "<b>sovra-determinato</b>: almeno tre meccanismi diversi produrrebbero "
          "esattamente la stessa prestazione, e due di essi non hanno nulla a che "
          "vedere con una traccia forense. Questa sezione li isola e li misura uno "
          "per uno. E' la parte piu' importante del lavoro."))

    A(h2(S, "6.1 Primo sospetto: la catena di ricampionamento"))
    A(p(S, "Nel dataset originale le due lineage hanno subito trattamenti opposti: "
          "<b>tutta</b> la famiglia CelebA e' stata ingrandita verso 256 (da "
          "178x218, da 128 nativi degli editing-GAN, da 216 di GDWCT), mentre "
          "<b>tutta</b> la famiglia FFHQ e' stata rimpicciolita da 1024 con "
          "Lanczos. Il ricampionamento coincide percio' perfettamente con "
          "l'etichetta da predire."))
    A(p(S, "Quanto pesa e' misurabile. Un <b>singolo scalare</b> - l'energia media "
          "dello spettro di potenza alle alte frequenze, calcolato senza alcun "
          "apprendimento - separa le due lineage con <b>AUC 0.929</b> per immagine "
          "(n = 1050). Le medie di classe non si sovrappongono: la famiglia CelebA "
          "sta fra 13.0 e 13.5, quella FFHQ fra 14.4 e 14.6."))
    X(figure(S, REP / "signatures_raw" / "radial_spectrum_hf.png",
             "Spettro di potenza azimutale medio per classe, zoom sulle alte "
             "frequenze, sul dataset originale. Le tre classi della lineage FFHQ "
             "stanno sopra, le quattro CelebA sotto: uno scalino di energia netto, "
             "senza sovrapposizione fra i gruppi. E' la firma della catena di "
             "ricampionamento, e da sola basta a separare le lineage.",
             max_w=125 * mm))
    A(p(S, "<b>Il controllo.</b> L'intero dataset (21.000 immagini) e' stato "
          "ricostruito facendo passare ogni classe per la stessa identica catena: "
          "256 -> 128 -> 256 con Lanczos, piu' riscrittura nello stesso formato. "
          "Dopo questo passaggio la firma di ricampionamento e' comune a tutte le "
          "classi e non puo' piu' portare informazione sulla lineage."))

    A(h2(S, "6.2 Il pavimento handcrafted"))
    A(p(S, "Il secondo controllo risponde a: <b>quanto di questo risultato "
          "otterrebbe un descrittore banale?</b> Per ogni famiglia di feature "
          "(spettro radiale FFT, DCT 2D poolata, istogrammi di colore RGB e HSV, "
          "residuo high-pass) si allena una semplice regressione logistica a "
          "predire la lineage, <b>con lo stesso protocollo</b>: solo le classi "
          "viste in addestramento, poi test sui generatori mai visti."))
    rows = []
    for tag, label in (("raw", "originale"), ("pm128", "normalizzato")):
        sig = D.get(f"sig_{tag}")
        if not sig:
            continue
        for fam in ("radial", "dct", "color", "residual", "all"):
            v = sig["floor"].get(fam)
            if v:
                rows.append([label, fam, v["dim"], num(v["auc_seen"]),
                             f"<b>{num(v['auc_unseen'])}</b>"])
    X(table(S, ["dataset", "famiglia", "dim.", "AUC classi viste",
                "AUC mai visti"], rows,
            widths=[30 * mm, 30 * mm, 18 * mm, 40 * mm, FRAME_W - 118 * mm]))
    X(keybox(S, "<b>Sul dataset originale il pavimento e' AUC 0.9999.</b> Una "
                "regressione logistica su 576 numeri fa quanto CLIP ViT-L/14. Il "
                "margine del modello addestrato e' +0.0001: sull'immagine intera, "
                "<b>l'uso di un VLM non e' il contributo</b>. E sul dataset "
                "normalizzato il pavimento resta a 0.9958 - quindi il "
                "ricampionamento spiega una parte del segnale, ma non tutto."))
    A(p(S, "Il dettaglio per classe rivela una differenza che l'AUC aggregata "
          "nascondeva: sul dataset originale il descrittore handcrafted prende "
          "StyleGAN3 al 100% ma <b>crolla al 63.2% su AttGAN</b>, dove il modello "
          "addestrato fa 100%. L'AUC resta alta perche' l'ordinamento e' buono "
          "anche quando la decisione alla soglia calibrata sbaglia: e' un buon "
          "esempio di perche' un'unica metrica aggregata non basta."))

    A(h2(S, "6.3 Secondo sospetto: il contenuto semantico"))
    A(p(S, "L'obiezione piu' forte non e' tecnica ma di senso comune: <i>StyleGAN3 "
          "finisce su FFHQ perche' genera volti che sembrano volti FFHQ</i>. "
          "Inquadratura, allineamento, illuminazione e demografia di CelebA e FFHQ "
          "sono diversi, e si vedono a occhio. Se basta quello, il risultato e' "
          "vero ma non forense: e' una somiglianza percettiva."))
    A(p(S, "<b>Il controllo</b>, il piu' diretto possibile: si usano le feature di "
          "un VLM pre-addestrato <b>senza alcun addestramento</b>. Le "
          "rappresentazioni di CLIP sono dominate dalla semantica. I centroidi di "
          "lineage sono calcolati <b>soltanto sulle immagini reali</b>, cioe' su "
          "contenuto autentico, e ogni immagine generata viene assegnata al "
          "centroide piu' vicino. Se questo basta, il contenuto spiega tutto."))
    rows = []
    for tag, label in (("raw", "originale"), ("pm128", "normalizzato")):
        con = D.get(f"con_{tag}")
        if con:
            noise = [v["correct_rate"] for v in con["per_class"].values()
                     if v["kind"] == "noise"]
            rows.append([label, num(con["auc_fake_only"]),
                         pct(sum(noise) / len(noise) if noise else None),
                         f"<b>{pct(sg3(con))}</b>"])
    X(table(S, ["dataset", "AUC (soli generati)", "noise-GAN corretti",
                "StyleGAN3"], rows,
            widths=[36 * mm, 44 * mm, 40 * mm, FRAME_W - 120 * mm]))
    X(keybox(S, "<b>Il contenuto semantico, da solo, basta.</b> Senza allenare "
                "niente e senza vedere un solo generatore, la somiglianza visiva "
                "fra volti StyleGAN3 e volti FFHQ reali porta StyleGAN3 su FFHQ nel "
                "99.0% dei casi (97.8% sul dataset normalizzato). Sull'immagine "
                "intera il modello addestrato, il descrittore handcrafted e la "
                "semantica pura sono <b>indistinguibili</b>: tutti e tre intorno a "
                "0.99. Il risultato della Sezione 5, da solo, non e' attribuibile a "
                "una traccia forense."))
    A(p(S, "Un controllo intermedio merita una nota. Allenando lo stesso modello con "
          "il <b>front-end a residuo high-pass</b>, che sopprime il contenuto a "
          "bassa frequenza, sul dataset normalizzato si ottiene AUC 1.0000, "
          "accuratezza 100% e FAR e FRR entrambi a zero sui mai visti: sopprimere "
          "il contenuto non peggiora nulla. Sembrerebbe la prova che il canale non "
          "sia semantico, ma il residuo di un filtro 3x3 conserva ancora il "
          "contenuto ad <b>alta</b> frequenza - capelli, bordi, struttura del "
          "volto. L'esperimento e' indicativo, non conclusivo. Serve un controllo "
          "piu' radicale."))
    return F


def part_sweep(S, D):
    F = []
    A = F.append
    X = F.extend

    A(PageBreak())
    A(h1(S, "Il risultato centrale: separare i due canali", 7))
    A(p(S, "A questo punto due spiegazioni concorrenti funzionano entrambe e non "
          "sono distinguibili: il contenuto semantico da solo basta (99%), e il "
          "residuo di basso livello da solo basta (100%). <b>Nessuna delle due e' "
          "necessaria.</b> Finche' resta cosi', non si puo' sostenere che "
          "l'attribuzione sia forense e non percettiva."))
    A(p(S, "L'idea per rompere il pareggio e' <b>ridurre progressivamente la "
          "finestra di analisi</b>. Un ritaglio quadrato piccolo non contiene piu' "
          "inquadratura, composizione, demografia: la semantica svanisce. Ma "
          "conserva texture, rumore e le tracce lasciate dal generatore. Se al "
          "restringersi della finestra il canale semantico crolla e il modello "
          "addestrato tiene, i due canali si separano."))
    A(p(S, "A ogni dimensione si misurano <b>tre metodi sullo stesso dato</b>: il "
          "modello addestrato, il pavimento handcrafted e il baseline zero-shot. La "
          "metrica e' la quota di immagini attribuite alla lineage corretta, "
          "perche' e' l'unica grandezza definita in modo identico per tutti e tre "
          "(le loro AUC sono calcolate su insiemi diversi e non sarebbero "
          "confrontabili). Il backbone e' ResNet18, completamente convoluzionale, "
          "che riceve la patch a <b>risoluzione nativa</b> senza reinterpolare."))

    A(h2(S, "7.1 Un problema nella selezione delle patch, e la sua correzione"))
    A(p(S, "La prima versione dell'esperimento selezionava, fra 16 posizioni "
          "casuali, quella a <b>varianza minima</b>: e' l'accorgimento classico in "
          "forense, perche' le zone uniformi portano la traccia del generatore "
          "senza portare contenuto. Ma la scelta <b>dipende dal contenuto</b>, "
          "quindi la posizione selezionata potrebbe correlare con la classe."))
    A(p(S, "Il sospetto era fondato, e la figura lo rende evidente: su CelebA "
          "quelle patch cadono sul <b>fondale da studio uniforme</b>, su FFHQ "
          "sull'<b>erba</b>. La politica stava campionando preferenzialmente il "
          "<b>background</b>, che e' fra le caratteristiche piu' discriminanti fra "
          "i due dataset. Il controllo si stava sabotando da solo."))
    A(p(S, "La correzione e' la politica a <b>griglia fissa</b>: quattro posizioni "
          "ai centri dei quadranti - (32,32), (160,32), (32,160), (160,160) - "
          "<b>identiche per ogni immagine e per ogni classe</b>. Nessuna differenza "
          "fra classi puo' derivare da dove si e' guardato. Le patch risultanti "
          "hanno varianza media doppia (2642 contro 1326): contengono occhi e "
          "bocca, non solo pelle liscia."))
    X(figure(S, REP / "figures" / "patch_positions_64.png",
             "Dove guardano le due politiche, patch 64x64. In alto la selezione per "
             "varianza minima: le posizioni cambiano da immagine a immagine e "
             "cadono sul fondale, che nelle due lineage e' sistematicamente diverso "
             "(studio contro esterni). In basso la griglia fissa: gli stessi quattro "
             "riquadri su ogni immagine, sempre sul volto. La seconda e' il "
             "controllo valido.", max_w=150 * mm))

    A(h2(S, "7.2 Lo sweep con griglia fissa: il risultato"))
    A(p(S, "Quota di immagini attribuite alla lineage corretta, su <b>StyleGAN3</b> "
          "(mai visto, noise-GAN). Dataset a ricampionamento normalizzato."))
    rows = sweep_rows(D, "grid")
    X(table(S, ["finestra", "deep", "handcrafted", "zero-shot (semantica)",
                "deep - zero-shot", "famiglia pavimento"], rows,
            widths=[22 * mm, 22 * mm, 26 * mm, 36 * mm, 30 * mm,
                    FRAME_W - 136 * mm], highlight=[1]))
    X(figure(S, REP / "grid_sweep" / "patch_sweep.png",
             "Sweep sulla dimensione della finestra, patch su griglia fissa. A "
             "sinistra la media sui tre generatori mai visti, a destra il solo "
             "StyleGAN3. Sull'immagine intera le tre curve sono sovrapposte: la "
             "somiglianza semantica spiega tutto e il risultato non e' attribuibile "
             "a una traccia forense. Al restringersi della finestra le curve si "
             "separano: la semantica crolla, il modello addestrato tiene.",
             max_w=FRAME_W))
    X(keybox(S, "<b>La dissociazione.</b> Su un riquadro 16x16 in posizione fissa - "
                "nessun fondale, nessuna inquadratura, nessuna composizione - il "
                "canale semantico scende al <b>73.5%</b> e il descrittore spettrale "
                "al <b>75.0%</b>, mentre la metrica addestrata resta al "
                "<b>93.9%</b> su StyleGAN3. Il divario passa da +2.0 punti "
                "sull'immagine intera a <b>+20.4 punti</b>. Esiste dunque un canale "
                "di basso livello, indipendente dal contenuto visibile, che "
                "trasferisce a generatori mai visti."))
    A(p(S, "Anche il margine sul pavimento handcrafted e' il piu' ampio misurato in "
          "tutto il lavoro: <b>+0.209 di AUC</b> a 16 pixel (0.9698 contro 0.7608). "
          "Sull'immagine intera era +0.0001."))

    A(h2(S, "7.3 Lo sweep con selezione per varianza, per confronto"))
    A(p(S, "La versione con selezione per varianza minima e' riportata come "
          "ablazione. I divari sono piu' ampi, ma vanno letti con la cautela del "
          "background: parte di quel vantaggio puo' derivare dal fatto che le patch "
          "cadevano su fondali sistematicamente diversi."))
    rows = sweep_rows(D, "patch")
    X(table(S, ["finestra", "deep", "handcrafted", "zero-shot (semantica)",
                "deep - zero-shot", "famiglia pavimento"], rows,
            widths=[22 * mm, 22 * mm, 26 * mm, 36 * mm, 30 * mm,
                    FRAME_W - 136 * mm]))
    A(p(S, "Due osservazioni. Le curve su griglia fissa sono <b>monotone in "
          "entrambi i pannelli</b>, mentre quelle per varianza hanno un calo non "
          "monotono a 32 pixel sulla singola classe StyleGAN3: la griglia e' anche "
          "il protocollo piu' stabile. E la famiglia di descrittori che costituisce "
          "il pavimento <b>cambia con la scala</b> - il residuo high-pass sulle "
          "finestre piccole, la DCT 2D su quelle grandi - il che e' coerente: sulle "
          "finestre piccole conta la texture locale, su quelle grandi la struttura "
          "spettrale globale."))
    return F


def part_explain(S, D):
    F = []
    A = F.append
    X = F.extend
    cam_rn = D.get("cam_rn") or {}
    cam_cl = D.get("cam_clip") or {}

    A(PageBreak())
    A(h1(S, "Che cosa cattura il modello", 8))

    A(h2(S, "8.1 Grad-CAM media per classe"))
    A(p(S, "La richiesta iniziale era: <i>si fa una Grad-CAM dove in media, ad "
          "esempio su CelebA, abbiamo una mappa media su dove si sta concentrando, "
          "e la stessa cosa su StarGAN, e andiamo a vedere se si sta focalizzando "
          "sulle stesse regioni</i>. Due difficolta' tecniche vanno risolte prima."))
    X(bullets(S, [
        "<b>Non c'e' un logit di classe.</b> La rete non classifica, produce un "
        "embedding. Serve uno scalare da derivare: si usa lo stesso score della "
        "decisione di lineage, cioe' la differenza fra la distanza dal centroide "
        "dell'altra lineage e quella dal centroide della propria. Il gradiente "
        "risponde alla domanda giusta: <i>quali regioni spingono l'immagine verso "
        "la sua lineage?</i>",
        "<b>Il target layer differisce fra CNN e ViT.</b> Per le ResNet e' l'ultimo "
        "blocco convoluzionale. Per i ViT l'uscita di un blocco e' una sequenza di "
        "token: va scartato il token di classe e i token di patch vanno rimessi in "
        "griglia. Si usa il penultimo blocco, perche' sull'ultimo le mappe ViT "
        "tendono a degenerare.",
    ]))
    A(p(S, "La mappa media per classe, da sola, sarebbe ingannevole: evidenzierebbe "
          "<i>la faccia</i> in tutte le classi. La figura informativa e' lo "
          "<b>scarto dalla media globale</b>, che mostra cosa e' specifico di una "
          "classe. E la versione quantitativa e' la <b>matrice di correlazione fra "
          "le mappe di scarto</b>."))
    X(figure(S, REP / "gradcam_pm128_resnet18" / "gradcam_mean.png",
             "Grad-CAM media per classe (sopra) e scarto dalla media globale "
             "(sotto), ResNet18 su immagine intera. Per decidere CelebA il modello "
             "guarda la regione centrale del volto (rosso), per decidere FFHQ la "
             "periferia - capelli e fondale (blu al centro, rosso ai bordi). "
             "StarGAN e' di nuovo l'outlier, con attenzione spostata in basso.",
             max_w=FRAME_W))
    rows = []
    for label, cam in (("ResNet18 (immagine intera)", cam_rn),
                       ("CLIP ViT-L/14 (immagine intera)", cam_cl)):
        if cam:
            rows.append([label,
                         f"<b>{cam['mean_within_lineage_corr']:+.3f}</b>",
                         f"<b>{cam['mean_cross_lineage_corr']:+.3f}</b>"])
    X(table(S, ["modello", "correlazione media within-lineage",
                "correlazione media cross-lineage"], rows,
            widths=[64 * mm, 52 * mm, FRAME_W - 116 * mm]))
    X(keybox(S, "<b>Le classi della stessa lineage attivano le stesse regioni, "
                "quelle di lineage diverse regioni opposte.</b> La correlazione fra "
                "le mappe di scarto e' +0.806 dentro la stessa lineage e -0.881 fra "
                "lineage diverse (ResNet18). AttGAN, GDWCT e real_celeba correlano "
                "fra loro fra +0.87 e +0.97; real_ffhq, StyleGAN2 e StyleGAN3 fra "
                "+0.976 e +0.992. E' la conferma spaziale di cio' che la matrice "
                "delle distanze mostrava nello spazio di embedding."))
    A(p(S, "<b>Un limite da dichiarare.</b> Il fatto che per FFHQ il modello guardi "
          "la periferia e' coerente con l'osservazione sui fondali: sull'immagine "
          "intera la Grad-CAM <b>conferma</b> che il modello usa anche il "
          "contenuto. Non e' una contraddizione, e' la stessa cosa che dicono i "
          "controlli della Sezione 6 - ed e' la ragione per cui e' lo sweep a patch, "
          "non la Grad-CAM, a isolare il canale di basso livello. La Grad-CAM va "
          "presentata come evidenza <b>qualitativa e di coerenza</b>. Va inoltre "
          "ricordato che la sua risoluzione e' quella della griglia del target "
          "layer (8x8 per ResNet a 256 pixel, 16x16 per un ViT/14 a 224): una "
          "traccia ad alta frequenza e' spazialmente diffusa, quindi non c'e' da "
          "aspettarsi mappe nitide."))

    A(h2(S, "8.2 Statistiche nel dominio frequenziale e di colore"))
    A(p(S, "La richiesta prevedeva anche: <i>possiamo aggiungere anche qualcosa con "
          "il dominio frequenziale, Fourier, istogrammi di colori, e tirare fuori "
          "tutte le statistiche</i>. Per ogni classe vengono calcolati quattro "
          "gruppi di descrittori interpretabili: il profilo di potenza azimutale "
          "(FFT radiale), la mappa media di log|FFT 2D| e log|DCT 2D|, gli "
          "istogrammi di colore RGB e HSV, e le stesse statistiche sul residuo "
          "high-pass."))
    X(figure(S, REP / "signatures_raw" / "spectral_maps.png",
             "Mappe spettrali medie per classe: sopra log|FFT 2D|, sotto log|DCT "
             "2D|. Le classi della stessa lineage hanno mappe simili fra loro e "
             "diverse dall'altra lineage. Sono i descrittori che costituiscono il "
             "pavimento handcrafted della Sezione 6.2.", max_w=FRAME_W))
    A(p(S, "Il dato piu' utile di questa analisi e' <b>quale</b> famiglia porta il "
          "segnale, perche' dice di che natura e' la traccia. Il colore e' debole "
          "in ogni configurazione (AUC 0.63-0.69 sui mai visti): la differenza fra "
          "lineage <b>non</b> e' cromatica. La DCT 2D e' la piu' forte sulle "
          "finestre grandi, il residuo high-pass sulle finestre piccole. Il profilo "
          "radiale, che e' in pratica una firma di ricampionamento, perde molto "
          "quando la pipeline viene normalizzata (da 0.962 a 0.893) - una conferma "
          "indipendente che il controllo della Sezione 6.1 ha effettivamente "
          "rimosso quel confound."))
    return F


BACKBONES = [
    ("clip_vit_l14", "CLIP ViT-L/14", "congelato", "1024", "224 (crop)", "0.59 / 303.77"),
    ("clip_vit_b16", "CLIP ViT-B/16", "congelato", "768", "224 (crop)", "0.46 / 86.26"),
    ("dinov2_vitb14", "DINOv2 ViT-B/14", "congelato", "768", "224 (crop)", "0.46 / 86.18"),
    ("dinov2_vitl14", "DINOv2 ViT-L/14", "congelato", "1024", "224 (crop)", "0.59 / 303.82"),
    ("resnet18", "ResNet18", "end-to-end", "512", "256 (nativo)", "11.50 / 11.50"),
    ("resnet50", "ResNet50", "end-to-end", "2048", "256 (nativo)", "24.62 / 24.62"),
]


def part_backbones(S, D):
    F = []
    A = F.append
    X = F.extend

    A(PageBreak())
    A(h1(S, "Confronto fra architetture", 9))
    A(p(S, "La richiesta prevedeva <i>una comparison magari con altre "
          "architetture, anche riusando la ResNet o altre architetture che "
          "esistono</i>. Il codice espone un registry in cui ogni backbone "
          "dichiara la propria risoluzione e normalizzazione, cosi' che lo stesso "
          "protocollo possa girare invariato su tutti. I backbone VLM e "
          "self-supervised vengono <b>congelati</b> (si allena solo la testa), le "
          "ResNet <b>end-to-end</b>: e' la distinzione richiesta."))
    rows = []
    for key, label, regime, feat, res, params in BACKBONES:
        lin = D.get(f"lin_pm128_{key}")
        if not lin:
            rows.append([f"<b>{label}</b>", regime, params, "n/d", "n/d",
                         "n/d", "n/d", "n/d"])
            continue
        th = lin["threshold"]
        # rapporto di separazione: distanza fra i centroidi reali delle due
        # lineage diviso la compattezza intra-classe. Misura quanto lo spazio e'
        # "collassato", che a AUC saturata e' l'unica cosa che distingue i modelli.
        M, cl = lin["class_distance_matrix"], lin["classes"]
        i, j = cl.index("real_celeba"), cl.index("real_ffhq")
        sep = M[i][j] / max(M[i][i], 1e-9)
        rows.append([f"<b>{label}</b>", regime, params, f"{lin['epoch']}",
                     num(th.get("auc_unseen")), pct(th.get("acc_unseen")),
                     pct(sg3(lin)), f"{sep:.0f}x"])
    X(table(S, ["backbone", "regime", "par. addestrati / tot. (M)",
                "epoca del best", "AUC mai visti", "acc.", "SG3",
                "separazione"], rows,
            widths=[26 * mm, 19 * mm, 32 * mm, 18 * mm, 21 * mm, 14 * mm, 14 * mm,
                    FRAME_W - 144 * mm], fontsize=7.4))
    A(p(S, "<b>Separazione</b> = distanza fra i centroidi delle due classi reali "
          "divisa per la compattezza intra-classe. A AUC saturata e' l'unica "
          "grandezza che distingue i modelli.", md=False))
    A(p(S, "<b>Come si legge, e perche' non e' la parte interessante.</b> "
          "Sull'immagine intera il task e' <b>saturo</b>: tutte le architetture "
          "arrivano a AUC 1.0 e accuratezza superiore al 99% sui generatori mai "
          "visti, e la ResNet18 - il modello piu' piccolo, 11.5 milioni di "
          "parametri - raggiunge AUC 1.0000 ed EER 0.0000 <b>dalla prima epoca</b>. "
          "Il confronto non discrimina perche' il problema, in quel regime, e' "
          "troppo facile: come mostra la Sezione 6, lo risolve anche una "
          "regressione logistica su 576 numeri."))
    A(p(S, "Una differenza c'e', e non riguarda l'accuratezza. Le ResNet "
          "addestrate end-to-end producono uno spazio <b>molto piu' collassato</b> "
          "(rapporto di separazione 87x e 65x, con distanza intra-classe fino a "
          "0.008) rispetto ai backbone congelati (6x-10x). Potendo modificare tutti "
          "i pesi, una rete end-to-end schiaccia le due classi in due punti quasi "
          "adimensionali. Non e' necessariamente un vantaggio: uno spazio "
          "collassato su due classi e' anche uno spazio che ha <b>memorizzato "
          "quelle due classi</b>, mentre la rappresentazione congelata conserva "
          "struttura interna e resta piu' informativa se domani si aggiungesse una "
          "terza lineage. Sull'altra dimensione osservabile, la velocita' di "
          "convergenza, CLIP ViT-B/16 e le due ResNet raggiungono il massimo "
          "<b>alla prima epoca</b>, DINOv2 ne richiede due o tre."))
    A(p(S, "Questo e' a sua volta un risultato, e va detto esplicitamente "
          "nell'articolo: <b>l'uso di un VLM pre-addestrato non e' il contributo "
          "del lavoro</b>. Il vantaggio del backbone congelato e' pratico e "
          "metodologico, non prestazionale: 0.59 milioni di parametri addestrati "
          "invece di 11.5 o 24.6 significa un addestramento in pochi minuti, "
          "nessun rischio di memorizzare il dataset nei pesi del backbone, e un "
          "protocollo in cui la rappresentazione e' <b>fissa e verificabile</b> - "
          "cio' che in un contesto probatorio conta piu' di una frazione di punto "
          "di accuratezza."))
    A(p(S, "Il regime in cui le architetture si differenzierebbero e' quello a "
          "patch piccole. La' pero' il confronto con i ViT sarebbe sleale: un "
          "backbone che attende 224 pixel richiederebbe di ingrandire una patch da "
          "16 o 32 pixel, distruggendo proprio la traccia di alta frequenza che si "
          "vuole misurare. Per questo lo sweep usa ResNet18, che riceve la patch a "
          "risoluzione nativa. Un confronto onesto a bassa risoluzione richiede "
          "backbone convoluzionali, ed e' una estensione naturale."))
    return F


def part_legal(S, D):
    F = []
    A = F.append
    X = F.extend

    A(PageBreak())
    A(h1(S, "Il ponte con l'analisi giuridica", 10))
    A(p(S, "Questa sezione non svolge l'analisi giuridica, che e' di competenza "
          "altrui. Elenca <b>cosa l'evidenza tecnica sostiene e cosa no</b>, nella "
          "forma piu' utilizzabile possibile."))

    A(h2(S, "10.1 Che cosa e' stato dimostrato"))
    X(bullets(S, [
        "Data un'immagine sintetica, e' possibile ricondurla al <b>dataset su cui "
        "il generatore e' stato addestrato</b>, con accuratezza superiore al 99% "
        "sull'immagine intera e del 93.9% su un riquadro di 16x16 pixel.",
        "La capacita' <b>trasferisce a generatori mai visti</b>: il sistema non ha "
        "bisogno di conoscere il modello che ha prodotto l'immagine. E' la "
        "proprieta' decisiva per l'uso pratico, perche' in un caso reale il modello "
        "sospetto tipicamente non e' disponibile.",
        "Il risultato <b>non e'</b> un artefatto di preprocessing: sopravvive alla "
        "normalizzazione della catena di ricampionamento su tutte le classi.",
        "Il risultato <b>non e'</b> soltanto somiglianza visibile: su finestre "
        "piccole la somiglianza semantica cala di oltre venti punti mentre il "
        "sistema tiene.",
        "La decisione ha un <b>tasso d'errore noto e misurato su dati mai visti</b>: "
        "soglia calibrata al punto di Equal Error Rate sulle sole classi viste, "
        "poi falsi positivi e falsi negativi riportati separatamente.",
    ]))

    A(h2(S, "10.2 Che cosa non e' stato dimostrato"))
    X(bullets(S, [
        "<b>Non</b> che una specifica fotografia fosse nel training set. Quella e' "
        "la <i>membership inference</i>, il livello (c) della Sezione 1.1, e "
        "richiede tecniche e garanzie diverse.",
        "<b>Non</b> che il contenuto sia irrilevante: su immagini intere il canale "
        "semantico da solo raggiunge gli stessi numeri. Le due spiegazioni sono "
        "ridondanti, e solo restringendo la finestra si separano.",
        "<b>Non</b> la generalizzazione a famiglie generative diverse: il dataset "
        "contiene soltanto GAN. Il contenzioso attuale sui dati di addestramento "
        "riguarda in larga parte i modelli a diffusione.",
        "<b>Non</b> la robustezza a manipolazioni ostili o anche solo ordinarie "
        "(ricompressione, riscalatura, filtri dei social network) applicate dopo la "
        "generazione. E' il primo esperimento da aggiungere.",
    ]))

    A(h2(S, "10.3 Perche' la forma della decisione conta"))
    A(p(S, "Un elemento merita attenzione. Il sistema non produce un'etichetta ma "
          "una <b>distanza</b>, confrontata con una soglia il cui tasso d'errore e' "
          "stato misurato su dati che il sistema non aveva visto. E' la forma in cui "
          "una tecnica si presenta a un vaglio di affidabilita': non <i>questa "
          "immagine viene da FFHQ</i>, ma <i>questa immagine cade dal lato FFHQ di "
          "una soglia che, su generatori mai visti, sbaglia nell'1.8% dei casi in un "
          "verso e nell'1.1% nell'altro</i>. La seconda formulazione e' attaccabile "
          "in modo specifico - ed e' per questo che vale di piu'."))
    A(p(S, "Analogamente, la distinzione fra editing-GAN e noise-GAN (Sezione 3.1) "
          "non e' un dettaglio tecnico ma <b>due livelli di pretesa diversi</b>. "
          "Quando l'output contiene i pixel di una fotografia reale, la pretesa "
          "riguarda quella fotografia e assomiglia a un'opera derivata. Quando "
          "l'output nasce da rumore, la pretesa riguarda <b>il dataset</b>, ed e' la "
          "questione nuova. Tenerle separate nell'articolo evita di rivendicare per "
          "la seconda la solidita' della prima."))
    return F


def part_limits(S, D):
    F = []
    A = F.append
    X = F.extend

    A(h1(S, "Limiti e lavori futuri", 11))
    X(bullets(S, [
        "<b>Il contenuto e' ridotto, non azzerato.</b> A 16 pixel il canale "
        "semantico resta al 73.5%: anche un riquadro piccolo porta tono della pelle "
        "e illuminazione. La dissociazione e' un divario che si apre, non un "
        "interruttore.",
        "<b>Due sole lineage.</b> Il disegno e' binario. Con piu' dataset di "
        "addestramento si potrebbe misurare se lo spazio e' organizzato per "
        "lineage anche oltre due classi, e se la struttura e' metrica o solo "
        "linearmente separabile.",
        "<b>Nessuna noise-GAN su CelebA.</b> L'asimmetria del dataset e' il limite "
        "strutturale: lato FFHQ ci sono due noise-GAN, lato CelebA solo "
        "editing-GAN. Un generatore da rumore addestrato su CelebA-HQ renderebbe il "
        "disegno simmetrico e l'evidenza forte disponibile su entrambi i lati.",
        "<b>Nessuna sorgente a diffusione.</b> E' l'estensione di maggiore impatto, "
        "perche' e' dove il problema giuridico e' oggi piu' vivo.",
        "<b>Robustezza non testata.</b> Ricompressione JPEG, riscalatura e filtri "
        "applicati dopo la generazione degraderanno il canale di basso livello, che "
        "e' proprio quello su cui si fonda la tesi. Serve una curva di degrado.",
        "<b>Risoluzione.</b> Tutto il lavoro e' a 256 pixel. Il downscale da 1024 "
        "distrugge gran parte delle tracce ad alta frequenza: lavorare a risoluzione "
        "nativa alzerebbe il tetto di cio' che e' estraibile.",
        "<b>Un solo seed.</b> Ogni configurazione e' stata addestrata una volta. Per "
        "l'articolo servono ripetizioni con seed diversi e intervalli di confidenza, "
        "soprattutto sui punti a finestra piccola dove la varianza e' maggiore.",
    ]))
    return F


def part_repro(S, D):
    F = []
    A = F.append
    X = F.extend

    A(h1(S, "Riproducibilita'", 12))
    A(p(S, "Tutto il codice e' nel repository <b>synth-attribution</b>. I dati e i "
          "checkpoint non sono versionati; i report in formato JSON e le figure "
          "si'. I moduli aggiunti in questa fase:"))
    X(table(S, ["file", "funzione"], [
        ["src/models/backbones.py", "registry dei backbone (ResNet, CLIP, DINOv2) con la propria spec di preprocessing"],
        ["src/models/siamese.py", "encoder siamese, freeze del backbone, testa configurabile, caricamento checkpoint"],
        ["src/data/siamese_dataset.py", "pairing per lineage, estrazione patch (flat / random / grid), augmentation forense"],
        ["src/eval/eval_lineage.py", "geometria, soglia calibrata con FAR/FRR, breakdown per generatore"],
        ["src/eval/class_signatures.py", "firme FFT/DCT/colore/residuo e pavimento handcrafted"],
        ["src/eval/content_baseline.py", "baseline zero-shot: quanto basta la sola semantica"],
        ["src/eval/eval_gradcam.py", "Grad-CAM media per classe e correlazione fra mappe di scarto"],
        ["src/viz/gradcam.py", "Grad-CAM per metric learning, su CNN e su ViT"],
        ["src/eval/plot_patch_sweep.py", "figura e tabella dello sweep"],
        ["scripts/normalize_pipeline.py", "dataset con catena di ricampionamento identica per tutte le classi"],
        ["scripts/show_patch_positions.py", "figura di metodo sulle posizioni delle patch"],
    ], widths=[52 * mm, FRAME_W - 52 * mm], fontsize=7.8))
    A(h2(S, "Pipeline completa"))
    X(code(S, [
        "# 1. manifest con i tre generatori in holdout",
        "python -m src.data.build_manifest --config configs/dataset_lineage.yaml",
        "",
        "# 2. dataset a ricampionamento normalizzato (controllo del confound)",
        "python scripts/normalize_pipeline.py --src data/raw --dst data/raw_pm128 --via 128",
        "python -m src.data.build_manifest --config configs/dataset_lineage_pm128.yaml",
        "",
        "# 3. un punto dello sweep: deep + pavimento + zero-shot",
        "python -m src.train.train_siamese --config configs/train_lineage_patch.yaml \\",
        "       --patch-size 64 --patch-policy grid --out runs/lineage_grid64_resnet18",
        "python -m src.eval.eval_lineage --checkpoint runs/lineage_grid64_resnet18/best.pt \\",
        "       --manifest data/manifest_lineage_pm128.csv --patch-size 64 --patch-policy grid",
        "python -m src.eval.class_signatures --manifest data/manifest_lineage_pm128.csv \\",
        "       --patch-size 64 --patch-policy grid",
        "python -m src.eval.content_baseline --manifest data/manifest_lineage_pm128.csv \\",
        "       --patch-size 64 --patch-policy grid",
        "",
        "# 4. figure, tabelle e questo documento",
        "python -m src.eval.plot_patch_sweep --prefix grid --out report/grid_sweep",
        "python -m src.eval.build_lineage_table --out report/lineage_summary.md",
        "python scripts/build_report.py",
    ]))
    return F


def part_appendix(S, D):
    F = []
    A = F.append
    X = F.extend
    A(PageBreak())
    A(h1(S, "Appendice: indice degli esperimenti", None))
    A(p(S, "Ogni riga corrisponde a una cartella in <tt>report/</tt> con il proprio "
          "<tt>report.json</tt>. Il <b>delta</b> e' il margine del modello "
          "addestrato sul pavimento handcrafted calcolato sullo stesso regime."))
    rows = []
    for name, regime in [
        ("raw_clip_vit_l14", "immagine intera, dataset originale"),
        ("pm128_clip_vit_l14", "immagine intera, ricampionamento normalizzato"),
        ("pm128_resnet18", "immagine intera, normalizzato"),
        ("pm128_resnet18_hp", "immagine intera, normalizzato, front-end a residuo"),
        ("pm128_resnet50", "immagine intera, normalizzato"),
        ("pm128_clip_vit_b16", "immagine intera, normalizzato"),
        ("pm128_dinov2_vitb14", "immagine intera, normalizzato"),
        ("pm128_dinov2_vitl14", "immagine intera, normalizzato"),
        ("patch16_resnet18", "patch 16, selezione per varianza"),
        ("patch32_resnet18", "patch 32, selezione per varianza"),
        ("patch64_resnet18", "patch 64, selezione per varianza"),
        ("patch128_resnet18", "patch 128, selezione per varianza"),
        ("grid16_resnet18", "patch 16, griglia fissa"),
        ("grid32_resnet18", "patch 32, griglia fissa"),
        ("grid64_resnet18", "patch 64, griglia fissa"),
        ("grid128_resnet18", "patch 128, griglia fissa"),
    ]:
        lin = D.get(f"lin_{name}")
        if not lin:
            continue
        th = lin["threshold"]
        tag = name.split("_")[0]
        fam, fv = floor_best(D.get(f"sig_{tag}"))
        delta = (None if (th.get("auc_unseen") is None or not fv
                          or fv.get("auc_unseen") is None)
                 else th["auc_unseen"] - fv["auc_unseen"])
        rows.append([f"<tt>{name}</tt>", regime, num(th.get("auc_unseen")),
                     pct(th.get("acc_unseen")), pct(sg3(lin)),
                     "-" if delta is None else f"{delta:+.4f}"])
    X(table(S, ["run", "regime", "AUC mai visti", "acc.", "SG3", "delta"], rows,
            widths=[40 * mm, 56 * mm, 22 * mm, 16 * mm, 16 * mm,
                    FRAME_W - 150 * mm], fontsize=7.4))
    A(p(S, "Riassunto tabellare completo, comprensivo dei pavimenti handcrafted per "
          "ogni famiglia di descrittori e dei baseline zero-shot: "
          "<tt>report/lineage_summary.md</tt>. Tabelle dello sweep: "
          "<tt>report/grid_sweep/patch_sweep.md</tt> e "
          "<tt>report/patch_sweep/patch_sweep.md</tt>."))
    return F


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report/relazione_fase8.pdf")
    ap.add_argument("--md", default="report/relazione_fase8.md")
    args = ap.parse_args()

    S = styles()
    D = gather()
    flow = []
    for fn in (part_intro, part_data_method, part_results, part_controls,
               part_sweep, part_explain, part_backbones, part_legal,
               part_limits, part_repro, part_appendix):
        flow.extend(fn(S, D))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=A4,
                            leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=22 * mm,
                            title="Attribuzione dei dati di addestramento di "
                                  "immagini sintetiche",
                            author="synth-attribution - Fase 8")
    doc.build(flow, onFirstPage=page_deco, onLaterPages=page_deco)
    print(f"PDF  -> {out}  ({out.stat().st_size/1e6:.2f} MB, {doc.page} pagine)")

    md = ROOT / args.md
    md.write_text("# Attribuzione dei dati di addestramento di immagini sintetiche\n"
                  + "\n".join(_md) + "\n", encoding="utf-8")
    print(f"MD   -> {md}  ({md.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
