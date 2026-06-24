# Synthetic Image Source Attribution (`synth-attribution`)

Progetto di **Multimedia Forensics** — Università di Catania (prof. Guarnera).
Attribuzione della **sorgente generativa** di immagini sintetiche di volti tramite
*metric learning* siamese: si impara uno spazio di embedding in cui immagini dello
stesso generatore (o reali) si raggruppano e generatori diversi si separano. Sopra
la baseline si studia la **generalizzazione a generatori mai visti**, un **front-end
a residuo** per esporre la traccia generativa, e (in arrivo) la localizzazione della
manipolazione e il confronto col baseline *handcrafted* precedente.

Estende un lavoro precedente basato su feature handcrafted (DCT/FFT/colore/SIFT +
classificatori classici, arXiv 2505.11110) verso un approccio deep, spostando il
focus dall'accuratezza in-distribution (già ~99% nel lavoro precedente) alla
**generalizzazione**, che è il problema aperto.

---

## 1. Domanda di ricerca

Data un'immagine di volto, a quale **sorgente generativa** appartiene (un certo
generatore, oppure reale)? E soprattutto: un embedding allenato su alcuni generatori
**generalizza a un generatore mai visto**? In quali condizioni la traccia è
trasferibile e in quali resta specifica del singolo modello?

## 2. Approccio

- **Encoder siamese** ResNet18 (pre-addestrato ImageNet) → testa di proiezione →
  embedding a 128-d **L2-normalizzato** (`src/models/siamese.py`).
- **Contrastive loss** (`src/losses/contrastive.py`): coppie *genuine* (stessa
  sorgente) avvicinate, *impostore* (sorgenti diverse) allontanate oltre un margine.
- **Framing di verifica** (distanze tra coppie): naturale per il setting *open-set*,
  cioè per valutare generatori non visti in training.
- **Front-end a residuo opzionale** (Fase 4): un pre-filtro high-pass / SRM davanti
  all'encoder, che sopprime il contenuto ed espone la traccia ad alta frequenza.

## 3. Struttura del repo

```
configs/                 config YAML
  dataset.yaml           lineage, split, holdout_architectures
  train.yaml             pairing, lineage_filter, front_end, iperparametri
data/                    dati locali (gitignorati) + README di acquisizione
src/
  data/
    prepare_reals.py     campiona/croppa/resize i reali, fissa il subset CelebA
    build_manifest.py    scansiona data/raw -> manifest.csv (con split e holdout)
    siamese_dataset.py   dataset a coppie (pairing per architettura) + lineage code
  models/siamese.py      encoder ResNet18 + proj + front-end a residuo
  losses/contrastive.py  contrastive loss
  train/train_siamese.py training + eval per-lineage, selezione best
  eval/eval_generalization.py  valutazione open-set su architettura held-out
scripts/                 generazione fake editing-GAN + utilità
  gen_stargan.py  gen_attgan.py  resize_dir.py
report/                  risultati (json/png) e, in fase finale, relazione LaTeX
runs/                    checkpoint (gitignorato)
```

## 4. Ambiente (Windows + Anaconda)

- **Training/valutazione**: `.venv` (Python 3.13, **PyTorch + CUDA**, es. cu124).
  Attivare sempre questo: `\.venv\Scripts\activate`. Verifica GPU:
  `python -c "import torch; print(torch.cuda.is_available())"` → `True`.
- **Solo generazione StyleGAN**: env conda separato (`stylegen`, py3.9,
  torch 1.9.1+cu111) usato col repo NVlabs `stylegan3`. Non usarlo per il training.
- `requirements.txt`: installare **prima** PyTorch con la build CUDA giusta, poi il
  resto (`numpy pandas pillow pyyaml scikit-learn matplotlib tqdm`).

## 5. Dataset (ricostruito da zero)

6 classi, 2 *lineage*, **3000 immagini per classe a 256×256** (totale **18000**):

| lineage | classe      | tipo         | origine / generatore                          |
|---------|-------------|--------------|-----------------------------------------------|
| celeba  | `real`      | reale        | CelebA (Kaggle `jessicali9530/celeba-dataset`)|
| celeba  | `stargan`   | editing-GAN  | StarGAN (yunjey), edit *Blond_Hair*           |
| celeba  | `attgan`    | editing-GAN  | AttGAN (elvisyjlin), edit *Blond_Hair*        |
| ffhq    | `real`      | reale        | FFHQ-256 (Kaggle `denislukovnikov/ffhq256...`)|
| ffhq    | `stylegan2` | noise-GAN    | StyleGAN2-ADA FFHQ (pickle ufficiale), 1024→256 |
| ffhq    | `stylegan3` | noise-GAN    | StyleGAN3-t FFHQ (pickle ufficiale), 1024→256 |

- **Editing-GAN** (StarGAN/AttGAN): immagine→immagine dal subset CelebA, quindi ogni
  fake ha una **reale sorgente allineata** (naming `<source_id>__<arch>.png`) → utile
  per coppie allineate e per la localizzazione (Fasi 4-5).
- **Noise-GAN** (StyleGAN2/3): generate da rumore su FFHQ, nessun allineamento.
- *Gotcha* di generazione: StarGAN va usato in `G.train()` (InstanceNorm: in `eval()`
  produce output corrotti); AttGAN in `G.eval()` (BatchNorm); pesi caricati con
  `weights_only=False`; StyleGAN ridimensionate 1024→256 con Lanczos.

Tutto il dataset è **gitignorato**. Procedura di acquisizione completa in
`data/README.md`. Il subset CelebA usato (`configs/celeba_subset.txt`) è committato
per riproducibilità.

## 6. Riproduzione della pipeline

```bash
# 1. Reali (subset deterministico + crop/resize 256)
python -m src.data.prepare_reals --celeba-dir <dir> --ffhq-dir <dir> --n 3000 --size 256

# 2. Fake editing-GAN sul subset CelebA (allineati)
python scripts/gen_stargan.py ...
python scripts/gen_attgan.py  ...
# 3. Fake noise-GAN: generare con il repo NVlabs stylegan3 (gen_images.py) dai
#    pickle ufficiali FFHQ, poi:  python scripts/resize_dir.py <dir> --size 256

# 4. Manifest (split 70/15/15 stratificato per classe, con eventuale holdout)
python -m src.data.build_manifest --config configs/dataset.yaml

# 5. Training (vedi config sotto)
python -m src.train.train_siamese --config configs/train.yaml

# 6. Valutazione di generalizzazione su un'architettura held-out
python -m src.eval.eval_generalization --checkpoint runs/<run>/best.pt \
    --manifest data/manifest.csv --lineage ffhq --holdout stylegan3 \
    --out report/<run>
```

### Leve di configurazione principali
- `dataset.yaml → split.holdout_architectures`: architetture spinte **tutte nel test
  split** (mai viste in training). Lo split è deterministico **per gruppo**, quindi
  cambiare l'holdout non perturba gli split delle altre classi.
- `train.yaml → data.pairing_policy`: `architecture` (genuine = stessa classe
  generativa; `real` separata per dataset in `real_celeba`/`real_ffhq`) |
  `same_source` | `aligned`.
- `train.yaml → data.lineage_filter`: `celeba` | `ffhq` | `null`. Allena **dentro
  una sola lineage** (negativi tutti difficili, zero confound di dataset) oppure su
  tutte (`null`, run misto con breakdown).
- `train.yaml → model.front_end`: `none` (RGB) | `highpass` | `srm` (residuo).

## 7. Protocollo sperimentale

Il punto delicato è **non misurare scorciatoie**. Allenare CelebA-vs-FFHQ produceva
AUC=1.0 dall'epoca 1: confound di dataset (risoluzione/JPEG triviali). Soluzione:
- **pairing per architettura** + **valutazione separata** within-celeba /
  within-ffhq / cross-lineage;
- **`lineage_filter`** per allenare entro una sola lineage (tutti i negativi sono
  distinzioni difficili a livello di architettura, nessun confound);
- selezione del **modello migliore sulla media delle AUC within-lineage**, non sulla
  AUC complessiva (gonfiata dal confound cross-lineage);
- **leave-one-architecture-out** (`holdout_architectures`) per la generalizzazione.

## 8. Risultati

### 8.1 Attribution in-distribution
| esperimento (lineage, pairing architettura) | AUC val |
|---|---|
| CelebA: `real_celeba` / `stargan` / `attgan` | ~1.00 (epoca 1) |
| FFHQ: `real_ffhq` vs `stylegan2` | **0.95** |
| FFHQ: `real_ffhq` vs `stylegan3` | **0.71** |

CelebA è triviale (artefatti forti delle GAN datate + scorciatoia di contenuto:
tutti i fake hanno i capelli biondi). FFHQ è un problema vero: la curva sale
gradualmente. **StyleGAN3 è molto più difficile da detectare di StyleGAN2 anche
in-distribution** (0.71 vs 0.95): quantifica l'effetto del design *alias-free* di
StyleGAN3, che rimuove gli artefatti ad alta frequenza che rendono SG2 facile.

### 8.2 Generalizzazione (leave-one-architecture-out, FFHQ, test open-set bilanciato)
| modello allenato su | held-out | AUC focalizz. held-out | 1-NN in classe | tende verso |
|---|---|---|---|---|
| `real` + `stylegan2` (RGB) | `stylegan3` | 0.51 | 45% | **reale** (SG3↔real 0.51 < SG3↔SG2 0.57) |
| `real` + `stylegan2` (residuo) | `stylegan3` | 0.52 | 42% | reale (ancora di più: SG3↔real 0.48) |
| `real` + `stylegan3` (RGB) | `stylegan2` | 0.57 | 35% | **SG3/fake** (SG2↔SG3 0.48 < SG2↔real 0.59) |

- **Nessuna generalizzazione pulita**: i fingerprint restano in larga parte specifici
  del generatore (AUC held-out ~0.5).
- **Il residuo non recupera, anzi**: l'high-pass enfatizza proprio gli artefatti che
  SG2 ha e SG3 (alias-free) non ha, quindi nello spazio del residuo SG3 somiglia
  *ancora di più* ai reali. È una conferma meccanicistica, non un fallimento del
  metodo.
- **Asimmetria**: l'alias-free SG3, se non visto, **passa per reale** (falso negativo,
  direzione pericolosa); il "rumoroso" SG2, se non visto, è comunque marchiato come
  **fake** (cade sul lato SG3). I generatori con artefatti forti lasciano tracce più
  generiche e riconoscibili; l'alias-free è *stealthy*.

## 9. Risultati chiave

1. Attribution in-distribution risolta (coerente col ~99% handcrafted), **ma non è il
   contributo**.
2. Diagnosi e neutralizzazione di un **confound di dataset** con un protocollo
   within-lineage.
3. **StyleGAN3 ≫ StyleGAN2 in difficoltà di detection** (0.71 vs 0.95): misura diretta
   dell'effetto alias-free.
4. L'embedding deep (RGB **o** residuo) **non generalizza** a un'architettura nuova; il
   *modo* in cui fallisce è **asimmetrico** e spiegabile con la "rumorosità" degli
   artefatti.

## 10. Roadmap / stato

- [x] **Fase 1** — Ricostruzione dataset (manca solo GDWCT, opzionale)
- [x] **Fase 2** — Baseline siamese (ResNet18 + contrastive loss)
- [x] **Fase 3** — Valutazione e generalizzazione leave-one-architecture-out
- [?] **Fase 4** — Front-end a residuo *(fatto: high-pass/SRM; da fare: analisi della
  traccia / autoencoder, e l'esperimento multi-source, idealmente cross-famiglia con
  un modello a diffusione)*
- [?] **Fase 5** — Localizzazione e percentuale di manipolazione (Grad-CAM/salienza
  sulle coppie allineate)
- [x] **Fase 6** — Confronto col baseline handcrafted sulla stessa prova di
  generalizzazione
- [x] **Fase 7** — Relazione LaTeX

## 11. Note e caveat

- **Scorciatoia "biondo"** (CelebA): i fake editing-GAN usano un edit fisso
  (`Blond_Hair`), quindi real-vs-fake su CelebA è in parte risolto dal contenuto. La
  distinzione architettura-vs-architettura resta pulita. Refinement previsto in Fase 5:
  variare l'attributo per immagine.
- **Resize**: i reali FFHQ sono nativi 256, le StyleGAN sono 1024→256. Il fatto che
  SG3 (downscalato come SG2) cada coi reali e non con SG2 indica che il modello non
  poggia sul solo artefatto di downscaling; per blindarlo si potrebbero rigenerare i
  reali con la stessa pipeline.
- **PCA 2D**: ingannevole quando una classe ha varianza molto alta (es. SG3); fidarsi
  della matrice delle distanze, o usare una proiezione che massimizzi la separazione.
