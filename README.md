# Synthetic Image Source Attribution (`synth-attribution`)

**Multimedia Forensics** project, University of Catania (Prof. Guarnera).

The repository studies two attribution questions for synthetic face images, using siamese metric learning:

1. **Generator attribution** (Phases 1–7): which generator produced an image, and does the learned embedding transfer to a generator never seen in training?
2. **Training-data attribution** (Phase 8): on which **dataset** was the generator trained? This is where copyright and privacy questions now concentrate, because they concern the training data rather than the final output. Phase 8 is the technical part of a paper in preparation with legal co-authors.

The project extends earlier work based on handcrafted features (DCT/FFT/colour/SIFT with classical classifiers, [arXiv:2505.11110](https://arxiv.org/abs/2505.11110)) towards a deep approach, moving the focus from in-distribution accuracy (already about 99% in the earlier work) to **generalisation**, which is the open problem.

---

## Contents

1. [Research questions](#1-research-questions)
2. [Approach](#2-approach)
3. [Repository layout](#3-repository-layout)
4. [Environment](#4-environment)
5. [Dataset](#5-dataset)
6. [Phases 1–7: generator attribution](#6-phases-17-generator-attribution)
7. [Phase 8: training-data attribution](#7-phase-8-training-data-attribution)
8. [Phase 8 follow-up (October 2026)](#8-phase-8-follow-up-october-2026)
9. [Reproduction](#9-reproduction)
10. [Status and next steps](#10-status-and-next-steps)
11. [Notes and caveats](#11-notes-and-caveats)
12. [Reports and paper material](#12-reports-and-paper-material)

---

## 1. Research questions

- Given a face image, which **generative source** produced it (a given generator, or a real camera)? Does an embedding trained on some generators **generalise to an unseen generator**?
- Given a synthetic image, can one recover the **dataset on which its generator was trained**? Does this capability hold for generators never seen in training, and is it a forensic trace rather than visible semantic similarity?

## 2. Approach

- **Siamese encoder**: a pretrained backbone (ResNet18/50 end-to-end, or frozen CLIP / DINOv2), a projection head (linear, ReLU, dropout, linear) and L2 normalisation to a 128-d embedding (`src/models/siamese.py`).
- **Contrastive loss** (`src/losses/contrastive.py`): `L = y·d² + (1 − y)·max(0, m − d)²`, margin `m = 1`, with `y = 1` for genuine pairs.
- **Pairing policy**: by *architecture* (genuine = same generator; Phases 1–7) or by *lineage* (genuine = same training dataset, regardless of the generator; Phase 8).
- **Decision with a calibrated threshold**: distance to the lineage centroids, or (Phase 8 follow-up) a **binary linear head** on the frozen encoder. The threshold is set at the equal error rate of the seen classes and applied unchanged to unseen generators, so that the error rate on unseen data is measured, not assumed.
- **Optional residual front end** (Phase 4): a high-pass / SRM pre-filter that suppresses content and exposes high-frequency traces.

## 3. Repository layout

```
configs/                  YAML configurations
  dataset*.yaml           lineages, splits, held-out architectures (main, normalised, real-only)
  train*.yaml             pairing policy, lineage filter, front end, backbone, patches
data/                     local data (git-ignored) + acquisition README
src/
  data/                   manifest building, pair datasets, patch extraction (flat / random / grid)
  models/                 backbone registry (ResNet, CLIP, DINOv2) and siamese encoder
  losses/contrastive.py   contrastive loss
  train/train_siamese.py  training with per-lineage evaluation and best-model selection
  eval/
    eval_generalization.py   open-set evaluation on a held-out architecture (Phases 1-7)
    eval_lineage.py          lineage geometry, calibrated threshold, per-generator breakdown (Phase 8)
    eval_binary.py           binary lineage head on the frozen encoder + regression test (follow-up)
    eval_robustness.py       robustness to rescaling, without retraining (follow-up)
    build_binary_table.py    summary table of the binary-head runs (follow-up)
    class_signatures.py      FFT/DCT/colour/residual signatures and handcrafted floor
    content_baseline.py      zero-shot semantic baseline (CLIP, centroids from real images)
    eval_gradcam.py          mean Grad-CAM per class and correlation of deviation maps
    plot_patch_sweep.py      window-sweep figure and table
    pairing_stats.py         composition of the training pairs
  viz/gradcam.py          Grad-CAM for metric learning, on CNNs and ViTs
scripts/                  data generation, pipeline normalisation, figures, report builder
report/                   results (JSON, PNG, Markdown) and the Phase 8 report
paper/figs/               English figures used in the paper
runs/                     checkpoints and logs (git-ignored)
```

## 4. Environment

- **Training and evaluation**: `.venv` (Python 3.12, PyTorch with CUDA). Check the GPU with `python -c "import torch; print(torch.cuda.is_available())"`.
- **StyleGAN generation only**: a separate conda environment (Python 3.9, torch 1.9.1+cu111) used with the NVlabs `stylegan3` repository.
- `requirements.txt`: install PyTorch with the right CUDA build first, then the rest (`numpy pandas pillow pyyaml scikit-learn matplotlib tqdm`).

## 5. Dataset

7 classes, 2 lineages, **3,000 images per class at 256×256** (21,000 images in total).

| Lineage | Class | Type | Origin |
|---|---|---|---|
| CelebA | `real_celeba` | real | CelebA, central crop, Lanczos to 256 |
| CelebA | `stargan` | editing GAN | StarGAN (yunjey), *Blond_Hair* edit, 128 → 256 |
| CelebA | `attgan` | editing GAN | AttGAN (elvisyjlin), *Blond_Hair* edit, 128 → 256 |
| CelebA | `gdwct` | editing GAN | GDWCT (WonwoongCho), 216 → 256 |
| FFHQ | `real_ffhq` | real | FFHQ-1024 → 256 (Lanczos), same pipeline as the generators |
| FFHQ | `stylegan2` | noise GAN | StyleGAN2-ADA FFHQ (official pickle), 1024 → 256 |
| FFHQ | `stylegan3` | noise GAN | StyleGAN3-t FFHQ (official pickle), 1024 → 256 |

- **Editing GANs** (StarGAN, AttGAN, GDWCT) modify a real CelebA photograph, so each fake has an aligned real source (`<source_id>__<arch>.png`). Their output contains the pixels of the input photograph.
- **Noise GANs** (StyleGAN2/3) generate from random noise: no input image exists, so any similarity to FFHQ can only come from the training data. They are the strong evidence in Phase 8.
- Generation notes: StarGAN must run in `G.train()` (InstanceNorm), AttGAN in `G.eval()` (BatchNorm); weights are loaded with `weights_only=False`; StyleGAN outputs are resized 1024 → 256 with Lanczos.

The dataset is git-ignored. The acquisition procedure is in `data/README.md`; the CelebA subset (`configs/celeba_subset.txt`) is committed for reproducibility.

## 6. Phases 1–7: generator attribution

### Protocol

The delicate point is not to measure shortcuts. Training CelebA against FFHQ gave AUC 1.0 from the first epoch: a dataset confound (resolution and compression). The protocol therefore uses pairing by architecture, separate within-CelebA / within-FFHQ / cross-lineage evaluation, a `lineage_filter` to train inside a single lineage, best-model selection on the mean within-lineage AUC, and leave-one-architecture-out for generalisation.

### In-distribution attribution

| Experiment (lineage, pairing by architecture) | Validation AUC |
|---|---|
| CelebA: `real_celeba` / `stargan` / `attgan` | about 1.00 (epoch 1) |
| FFHQ: `real_ffhq` vs `stylegan2` | **0.95** |
| FFHQ: `real_ffhq` vs `stylegan3` | **0.71** |

CelebA is trivial (strong artefacts of older GANs, and a content shortcut: every fake has blond hair). FFHQ is a real problem. **StyleGAN3 is much harder to detect than StyleGAN2 even in distribution** (0.71 vs 0.95), which quantifies the effect of its alias-free design.

### Generalisation (leave-one-architecture-out, FFHQ, balanced open-set test)

| Trained on | Held out | AUC on held-out | 1-NN in class | Leans towards |
|---|---|---|---|---|
| `real` + `stylegan2` (RGB) | `stylegan3` | 0.51 | 45% | **real** |
| `real` + `stylegan2` (residual) | `stylegan3` | 0.52 | 42% | real (even more) |
| `real` + `stylegan3` (RGB) | `stylegan2` | 0.57 | 35% | **fake** |

- **No clean generalisation**: fingerprints remain largely generator-specific (held-out AUC about 0.5).
- **The residual front end does not help**: the high-pass filter emphasises exactly the artefacts that StyleGAN2 has and alias-free StyleGAN3 lacks.
- **Asymmetry**: an unseen StyleGAN3 passes for real (false negative, the dangerous direction), while an unseen StyleGAN2 is still marked as fake.
- In closed-set attribution over the seven classes the final model reaches top-1 = 0.763, and **no error crosses lineages**: the training dataset is perfectly separable even where the generator is not. This observation started Phase 8.

## 7. Phase 8: training-data attribution

The question changes from *which generator produced this image* to **on which dataset the generator was trained**. With **lineage pairing**, real CelebA, StarGAN, AttGAN and GDWCT are all positives of each other, and so are real FFHQ, StyleGAN2 and StyleGAN3. **AttGAN, GDWCT and StyleGAN3 are held out of training** (`configs/dataset_lineage.yaml`). 51.1% of the genuine training pairs are (real, generated): the same pair that was a negative example in the flat classification of the earlier work (`report/pairing_composition.json`).

### The whole-image result is over-determined

On whole images the attribution is perfect (AUC 1.000, unseen StyleGAN3 at 99.8%), but three different mechanisms give the same number:

| Explanation | How it is measured | AUC on unseen |
|---|---|---|
| Resampling chain | the CelebA lineage is upsampled, the FFHQ lineage downsampled | 0.9999 (handcrafted floor, logistic regression) |
| Semantic content | frozen CLIP, **no training**, centroids from real images only | 0.9976 |
| Low-level trace | the trained model | 1.0000 |

The resampling confound was neutralised by passing every class through the same chain (`scripts/normalize_pipeline.py`, 256 → 128 → 256 with Lanczos). The semantic channel, however, remained indistinguishable from the forensic one.

### Separating the channels: window sweep

Reducing the analysis window to native square patches removes framing, composition and demography while keeping texture and noise. Fraction of **unseen StyleGAN3** attributed to the correct lineage, normalised dataset.

Minimum-variance patch selection (`flat`):

| Window | Trained | Handcrafted | Zero-shot (semantics) | Trained − zero-shot |
|---|---|---|---|---|
| 16 px | 86.9% | 73.3% | 71.2% | +15.6 pt |
| 32 px | 96.5% | 80.1% | 63.6% | **+32.9 pt** |
| 64 px | 98.9% | 86.2% | 86.8% | +12.1 pt |
| 128 px | 99.5% | 92.4% | 96.1% | +3.4 pt |
| 256 px (whole) | 99.8% | 94.2% | 97.8% | +2.0 pt |

The `flat` policy depends on content: on CelebA its patches fell on the uniform studio background, on FFHQ on grass (`report/figures/patch_positions_64.png`). The **fixed grid** (`grid`) uses the same four quadrant-centre positions for every image and class, so no difference can come from where the model looks:

| Window | Trained | Handcrafted | Zero-shot | Trained − zero-shot |
|---|---|---|---|---|
| 16 px | 93.9% | 75.0% | 73.5% | **+20.4 pt** |
| 32 px | 99.0% | 82.2% | 82.2% | +16.8 pt |
| 64 px | 99.9% | 86.8% | 95.1% | +4.7 pt |
| 128 px | 99.9% | 88.2% | 95.1% | +4.7 pt |
| 256 px (whole) | 99.8% | 94.2% | 97.8% | +2.0 pt |

**Reading.** On whole images the three methods cannot be told apart. As the window shrinks the semantic channel collapses while the trained model holds: a **low-level channel, independent of visible content and of the resampling chain, links a generated image to its training dataset and transfers to generators never seen in training**. Figures and tables: `report/grid_sweep/`, `report/patch_sweep/`; all runs: `report/lineage_summary.md`.

### Training on authentic data only

In the main regime the model has seen two generators (StarGAN, StyleGAN2). The real-only variant (`configs/dataset_lineage_realonly.yaml`) puts **all five generators in hold-out**: the model never sees a generated image. This is also the situation of a person who claims that their data were used: they only own their authentic archive.

| Regime | Two generators in training | **Real images only** | Handcrafted floor (real only) |
|---|---|---|---|
| Whole image | 99.8% | **100.0%** | 91.0% |
| Patch 64 px, grid | 99.9% | **99.9%** | 80.2% |
| Patch 16 px, grid | 93.9% | **89.2%** | 67.8% |

Removing the generators from training costs the model between 0 and 4.6 points, but costs the handcrafted floor much more, so **the margin grows** (from +13.1 to +19.6 points on 64-px patches). The signal lives in the authentic data, not in the exposure to synthetic examples.

## 8. Phase 8 follow-up (October 2026)

Changes discussed with the supervisor and run after the Phase 8 report.

### Binary lineage head on the frozen encoder

`src/eval/eval_binary.py` trains a single linear layer `Linear(128, 2)` (cross-entropy, Adam, learning rate 1e-2, weight decay 1e-4, 25 epochs) on top of the **frozen** encoder, with lineage labels and seen classes only. The threshold is the EER point on the **validation** set. The same script recomputes the centroid rule with the original protocol and with a clean protocol, on the same embedding and the same test sample.

- **Regression test.** With the original protocol the script reproduces the published `report.json` of all 14 runs exactly (difference 0.0 on AUC, accuracy and threshold), provided the same test sample size is used: 400 per class for whole images on the normalised dataset, 800 per class for patch models and for the real-only regime.
- **Result.** The binary head does not outperform the centroid rule. On unit-norm embeddings both rules are hyperplanes (`‖z − c_A‖² − ‖z − c_B‖²` is affine in `z`), so a large gain was not expected.

| Model | Regime | AUC unseen: original / clean / binary | Accuracy unseen: original / clean / binary |
|---|---|---|---|
| ResNet18 | normalised, patch 16 | 0.9698 / 0.9710 / 0.9720 | 89.9 / 90.0 / 90.0 % |
| ResNet18 | normalised, patch 32 | 0.9977 / 0.9977 / 0.9976 | 97.5 / 97.5 / 97.5 % |
| ResNet18 | normalised, patch 64 | 1.0000 / 1.0000 / 1.0000 | 99.2 / 99.4 / 99.4 % |
| ResNet18 | real-only, patch 16 | 0.9446 / 0.9450 / 0.9457 | 84.7 / 85.8 / 85.9 % |
| ResNet18 | real-only, patch 64 | 0.9991 / 0.9983 / **0.9923** | 98.4 / 98.1 / 98.1 % |
| 7 backbones and variants | normalised, whole image | ≥ 0.9999 everywhere | 99.8–100 % |

The only exception is the real-only patch-64 model, where the binary head transfers less well to unseen generators. Full table: `report/binary_summary.md`; per-run outputs: `report/binary_<run>/`.

- **The two-dimensional output of the head is one-dimensional in practice**: the two logits are perfectly anti-correlated (correlation −1.000), because softmax cross-entropy only constrains their difference. The score `logit_FFHQ − logit_CelebA` carries all the information.

### Robustness to rescaling

`src/eval/eval_robustness.py` reduces the test images by a factor `f` and restores them to 256 px (bilinear), without retraining and with the threshold fixed on clean validation data (ResNet18, normalised dataset, whole image):

| Resolution before restoring 256 px | AUC seen | AUC unseen | Accuracy unseen | StyleGAN3 |
|---|---|---|---|---|
| 256 | 1.0000 | 1.0000 | 100.0% | 100.0% |
| 192 | 1.0000 | 1.0000 | 100.0% | 100.0% |
| 128 | 1.0000 | 1.0000 | 99.1% | 97.2% |
| 64 | 0.7597 | 0.7199 | 66.7% | 0.0% |

The signal survives down to 128 px and is lost at 64 px. The 128-px test adds a second 256 → 128 → 256 reduction to images that already went through the normalisation chain.

### Grad-CAM with the binary score

`src/eval/eval_gradcam.py --head report/binary_<run>/head.pt` uses the binary logit, oriented towards the image's own lineage, instead of the centroid distance. Correlation between deviation maps (ResNet18, whole image, same sampling): centroid score +0.787 within lineage / −0.868 across lineages; binary logit +0.880 / −0.929. The values published in the Phase 8 report (+0.806 / −0.881 for ResNet18, +0.631 / −0.759 for CLIP) cannot be reproduced exactly because the sampling parameters of those runs were not recorded; the values above are regenerated with the code defaults.

### Issues found in the original protocol

- `eval_lineage.py` computes centroids and threshold on the **test** split (seen classes only). Recomputing them on training and validation data changes the AUC by at most 0.0012 and the accuracy by at most 1.1 points: the published numbers are not inflated in a meaningful way, but the clean protocol is the one to report.
- The grid-sweep evaluations were run with `--patches-per-image 4 --n-per-class 800`. Without `--patches-per-image 4`, the grid policy evaluates a single central patch, which is not one of the four training positions. The command printed in the Phase 8 report omits both options.

### English figures

All plotting scripts now use English labels, and the PCA of `eval_lineage.py` is coloured by lineage (green: CelebA, red: FFHQ; circles: seen classes, crosses: unseen generators). The figures used in the paper are regenerated in `paper/figs/`; the regenerated numbers match the published ones (lineage thresholds and AUCs, closed-set top-1 0.7629, handcrafted floor).

## 9. Reproduction

Phases 1–7:

```bash
python -m src.data.prepare_reals --celeba-dir <dir> --ffhq-dir <dir> --n 3000 --size 256
python scripts/gen_stargan.py ...
python scripts/gen_attgan.py ...
# noise GANs: generate with the NVlabs stylegan3 repository, then
python scripts/resize_dir.py <dir> --size 256
python -m src.data.build_manifest --config configs/dataset.yaml
python -m src.train.train_siamese --config configs/train.yaml
python -m src.eval.eval_generalization --checkpoint runs/<run>/best.pt \
    --manifest data/manifest.csv --lineage ffhq --holdout stylegan3 --out report/<run>
```

Phase 8:

```bash
python -m src.data.build_manifest --config configs/dataset_lineage.yaml
python scripts/normalize_pipeline.py --src data/raw --dst data/raw_pm128 --via 128
python -m src.data.build_manifest --config configs/dataset_lineage_pm128.yaml

# one point of the grid sweep: trained model + handcrafted floor + zero-shot baseline
python -m src.train.train_siamese --config configs/train_lineage_patch.yaml \
    --patch-size 64 --patch-policy grid --out runs/lineage_grid64_resnet18
python -m src.eval.eval_lineage --checkpoint runs/lineage_grid64_resnet18/best.pt \
    --manifest data/manifest_lineage_pm128.csv --patch-size 64 --patch-policy grid \
    --patches-per-image 4 --n-per-class 800 --out report/lineage_grid64_resnet18
python -m src.eval.class_signatures --manifest data/manifest_lineage_pm128.csv --patch-size 64 --patch-policy grid --patches-per-image 4
python -m src.eval.content_baseline --manifest data/manifest_lineage_pm128.csv --patch-size 64 --patch-policy grid --patches-per-image 4
python -m src.eval.plot_patch_sweep --prefix grid --out report/grid_sweep
python scripts/build_report.py
```

Phase 8 follow-up (patch size, policy and patches per image are read from the checkpoint):

```bash
python -m src.eval.eval_binary --checkpoint runs/lineage_grid16_resnet18/best.pt \
    --n-per-class 800 --published report/lineage_grid16_resnet18/report.json \
    --out report/binary_grid16_resnet18
python -m src.eval.build_binary_table --out report/binary_summary.md
python -m src.eval.eval_robustness --checkpoint runs/lineage_pm128_resnet18/best.pt \
    --binary-report report/binary_pm128_resnet18 --out report/robustness_pm128_resnet18
python -m src.eval.eval_gradcam --checkpoint runs/lineage_pm128_resnet18/best.pt \
    --manifest data/manifest_lineage_pm128.csv --head report/binary_pm128_resnet18/head.pt \
    --out report/binary_gradcam_pm128_resnet18
```

## 10. Status and next steps

- [x] **Phase 1** – Dataset reconstruction (GDWCT included)
- [x] **Phase 2** – Siamese baseline (ResNet18 + contrastive loss)
- [x] **Phase 3** – Evaluation and leave-one-architecture-out generalisation
- [?] **Phase 4** – Residual front end (high-pass/SRM done; trace analysis and a multi-source, cross-family experiment still open)
- [?] **Phase 5** – Localisation and percentage of manipulation (Grad-CAM / saliency on the aligned pairs)
- [x] **Phase 6** – Comparison with the handcrafted baseline on the same generalisation test
- [x] **Phase 7** – LaTeX report
- [x] **Phase 8** – Training-data attribution (lineage pairing, resampling and content controls, window sweep, real-only regime)
- [x] **Phase 8 follow-up** – Binary head, regression tests, rescaling robustness, English figures

Next steps: diffusion-based generators; robustness to JPEG recompression and social-network processing; repeated runs with several seeds and confidence intervals; a direct measurement of the frequency hypothesis (traces produced at 1024 px that persist after reduction to 256 px); more source datasets than two; masks of the edited regions for editing GANs; extension to video; agent-based explanations of the attribution.

## 11. Notes and caveats

- **"Blond" shortcut** (CelebA): the editing GANs use a fixed edit (*Blond_Hair*), so real-vs-fake on CelebA is partly solved by content. Architecture-vs-architecture remains clean.
- **Content is reduced, not removed**: even a 16-px patch carries skin tone and illumination. The dissociation in the sweep is a gap that opens, not a switch.
- **Two lineages only, GANs only, faces only, a single seed per configuration.** No noise GAN is trained on CelebA, so the strong evidence is available on the FFHQ side only.
- **PCA**: misleading when one class has a much larger variance; the distance matrices in the reports are more reliable.

## 12. Reports and paper material

- `report/binary_summary.md`: binary head against the centroid rule, all 14 runs.
- `report/binary_*`, `report/robustness_pm128_resnet18/`, `report/binary_gradcam_pm128_resnet18/`: outputs of the follow-up experiments.
- `paper/figs/`: English figures used in the paper. The manuscript itself is in preparation with co-authors and is not part of this repository.
