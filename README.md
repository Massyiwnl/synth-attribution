# Synthetic Image Source Attribution

Progetto di Multimedia Forensics (Universita' di Catania): attribution della
*source* di immagini sintetiche di volti tramite metric learning siamese,
estrazione della traccia generativa (residuo) e localizzazione della
manipolazione. Riprende ed estende un lavoro precedente basato su feature
handcrafted (DCT/FFT + classificatori classici) verso un approccio deep.

## Setup

```bash
git clone <repo-url>
cd synth-attribution
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
# Installa PyTorch per la tua CUDA da https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

## Fase 1 - Dataset (ricostruzione da zero)

1. Scarica i reali (TIENILI FUORI dal repo): CelebA
   (Kaggle `jessicali9530/celeba-dataset`) e FFHQ-256
   (Kaggle `denislukovnikov/ffhq256-images-only`).
2. Prepara i reali e fissa il sottoinsieme CelebA:
   ```bash
   python -m src.data.prepare_reals --celeba-dir <dir> --ffhq-dir <dir> --n 3000 --size 256
   ```
3. Genera i fake editing-GAN sul sottoinsieme (allineati): `scripts/gen_*.py`.
4. Costruisci il manifest:
   ```bash
   python -m src.data.build_manifest --config configs/dataset.yaml
   ```

Dettagli di acquisizione in `data/README.md`.

## Struttura

```
configs/    config YAML (dataset, training)
data/       dati locali (gitignorati) + README di acquisizione
src/data/   preparazione reali, manifest, dataset siamese
scripts/    generazione fake (StarGAN, AttGAN, GDWCT, StyleGAN)
src/models  src/losses  src/train  src/eval  src/viz   (Fasi 2+)
report/     relazione LaTeX (fase finale)
```

## Roadmap

- Fase 1 - Ricostruzione dataset *(in corso)*
- Fase 2 - Baseline siamese (ResNet18 + contrastive loss)
- Fase 3 - Valutazione e generalizzazione (tabella riassuntiva)
- Fase 4 - Analisi del residuo / autoencoder (traccia)
- Fase 5 - Localizzazione e percentuale di manipolazione
- Fase 6 - Confronto col baseline handcrafted
- Fase 7 - Relazione LaTeX
