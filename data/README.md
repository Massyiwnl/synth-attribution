# Fase 1 - Dataset

Riproduce il dataset IPLab/Guarnera (CelebA/FFHQ + GAN) per l'attribution
di immagini sintetiche. Il codice e' agnostico rispetto all'acquisizione:
legge solo `data/manifest.csv`.

## Acquisizione (in ordine di preferenza)

1. **Recupero dal laboratorio** (consigliato): reali (CelebA, FFHQ) + i fake
   delle due StyleGAN dal dataset del lavoro precedente. Coerenza garantita.
2. **Reali da scaricare**: CelebA allineato (`img_align_celeba`, 178x218 JPEG)
   e thumbnail FFHQ 128 (`NVlabs/ffhq-dataset`).
3. **Editing-GAN da rigenerare** (StarGAN/AttGAN/GDWCT) su un sottoinsieme
   CelebA fissato, **preservando la corrispondenza** real->fake nel nome file.

Repo dei modelli pre-addestrati:
- StarGAN  256x256  -> github.com/yunjey/stargan
- AttGAN   256x256  -> github.com/elvisyjlin/AttGAN-PyTorch
- GDWCT    216x216  -> github.com/WonwoongCho/GDWCT
- StyleGAN  1024    -> github.com/NVlabs/stylegan      (TF1; preferire il recupero)
- StyleGAN2 1024    -> github.com/NVlabs/stylegan2  (o stylegan2-ada-pytorch)

Target: ~3000 immagini per classe (come nel lavoro precedente).

## Alberatura attesa (`data/raw/`)

```
data/raw/
  real/celeba/<id>.jpg
  real/ffhq/<id>.png
  fake/stargan/<source_id>__stargan.png   # source_id = id CelebA di partenza
  fake/attgan/<source_id>__attgan.png
  fake/gdwct/<source_id>__gdwct.png
  fake/stylegan/<id>.png                   # noise-GAN: nessuna sorgente
  fake/stylegan2/<id>.png
```

La convenzione `<source_id>__<arch>` per le editing-GAN e' cio' che abilita le
coppie allineate (Fasi 4-5). La impostiamo noi quando generiamo i fake.

## Costruzione del manifest

```
python -m src.data.build_manifest --config configs/dataset.yaml
```

Produce `data/manifest.csv` con colonne:
`path, label, source_dataset, architecture, source_id, width, height, split`
