# Binary head on the frozen encoder: comparison with centroids

Test set sampled as in `eval_lineage.py`. Threshold: EER on the validation set (clean) or on the test set (original protocol, for comparison).

## Methods compared

| run | regime | AUC unseen: original | AUC unseen: clean | AUC unseen: binary | acc. unseen: original | acc. unseen: clean | acc. unseen: binary | FAR binary | FRR binary | tau binary |
|---|---|---|---|---|---|---|---|---|---|---|
| grid128_resnet18 | normalized, patch 128 px, grid | 1.0000 | 1.0000 | 1.0000 | 100.0% | 100.0% | 100.0% | 0.0% | 0.1% | +0.0378 |
| grid16_resnet18 | normalized, patch 16 px, grid | 0.9698 | 0.9710 | 0.9720 | 89.9% | 90.0% | 90.0% | 11.8% | 6.4% | +0.1328 |
| grid32_resnet18 | normalized, patch 32 px, grid | 0.9977 | 0.9977 | 0.9976 | 97.5% | 97.5% | 97.5% | 3.2% | 1.0% | -0.7300 |
| grid64_resnet18 | normalized, patch 64 px, grid | 1.0000 | 1.0000 | 1.0000 | 99.2% | 99.4% | 99.4% | 0.8% | 0.1% | +1.2293 |
| pm128_clip_vit_b16 | normalized, whole image 256 | 1.0000 | 1.0000 | 1.0000 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | -0.0677 |
| pm128_clip_vit_l14 | normalized, whole image 256 | 1.0000 | 1.0000 | 1.0000 | 99.8% | 99.9% | 99.9% | 0.0% | 0.2% | +0.7263 |
| pm128_dinov2_vitb14 | normalized, whole image 256 | 1.0000 | 1.0000 | 1.0000 | 99.9% | 99.9% | 99.9% | 0.1% | 0.0% | +0.2695 |
| pm128_dinov2_vitl14 | normalized, whole image 256 | 1.0000 | 1.0000 | 1.0000 | 99.8% | 99.8% | 99.9% | 0.0% | 0.2% | +0.8711 |
| pm128_resnet18 | normalized, whole image 256 | 1.0000 | 1.0000 | 1.0000 | 99.9% | 100.0% | 100.0% | 0.0% | 0.0% | +6.0393 |
| pm128_resnet18_hp | normalized, whole image 256 + high-pass residual | 1.0000 | 1.0000 | 1.0000 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | +5.9582 |
| pm128_resnet50 | normalized, whole image 256 | 1.0000 | 1.0000 | 1.0000 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | +4.6797 |
| realonly256_resnet18 | real-only, whole image 256 | 1.0000 | 1.0000 | 1.0000 | 100.0% | 100.0% | 100.0% | 0.0% | 0.1% | +4.1529 |
| realonlygrid16_resnet18 | real-only, patch 16 px, grid | 0.9446 | 0.9450 | 0.9457 | 84.7% | 85.8% | 85.9% | 16.3% | 10.8% | -0.1173 |
| realonlygrid64_resnet18 | real-only, patch 64 px, grid | 0.9991 | 0.9983 | 0.9923 | 98.4% | 98.1% | 98.1% | 3.1% | 0.1% | +2.1823 |

## Per-generator accuracy with the binary head

`*` = generator unseen in that run.

| run | stargan | attgan | gdwct | stylegan2 | stylegan3 | real_celeba | real_ffhq |
|---|---|---|---|---|---|---|---|
| grid128_resnet18 | 100.0% | 100.0% * | 100.0% * | 100.0% | 99.9% * | 100.0% | 100.0% |
| grid16_resnet18 | 97.4% | 91.2% * | 85.1% * | 94.1% | 93.6% * | 89.4% | 91.2% |
| grid32_resnet18 | 99.4% | 97.8% * | 95.8% * | 99.2% | 99.0% * | 98.9% | 99.0% |
| grid64_resnet18 | 100.0% | 99.1% * | 99.2% * | 100.0% | 99.9% * | 99.8% | 99.6% |
| pm128_clip_vit_b16 | 100.0% | 100.0% * | 100.0% * | 99.8% | 100.0% * | 99.8% | 100.0% |
| pm128_clip_vit_l14 | 100.0% | 100.0% * | 100.0% * | 100.0% | 99.8% * | 99.8% | 99.8% |
| pm128_dinov2_vitb14 | 100.0% | 99.8% * | 100.0% * | 99.8% | 100.0% * | 99.8% | 99.8% |
| pm128_dinov2_vitl14 | 100.0% | 100.0% * | 100.0% * | 99.8% | 99.8% * | 99.8% | 99.2% |
| pm128_resnet18 | 100.0% | 100.0% * | 100.0% * | 100.0% | 100.0% * | 100.0% | 100.0% |
| pm128_resnet18_hp | 100.0% | 100.0% * | 100.0% * | 100.0% | 100.0% * | 100.0% | 99.8% |
| pm128_resnet50 | 100.0% | 100.0% * | 100.0% * | 100.0% | 100.0% * | 100.0% | 100.0% |
| realonly256_resnet18 | 100.0% * | 100.0% * | 100.0% * | 100.0% * | 99.9% * | 100.0% | 99.8% |
| realonlygrid16_resnet18 | 77.8% * | 91.1% * | 82.1% * | 90.8% * | 87.8% * | 90.5% | 85.4% |
| realonlygrid64_resnet18 | 91.2% * | 99.6% * | 99.8% * | 100.0% * | 99.9% * | 99.9% | 99.8% |
