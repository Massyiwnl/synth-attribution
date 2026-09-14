# Fase 8 - Attribuzione dei dati di addestramento

## Pavimento handcrafted (regressione logistica su descrittori banali)

AUC nel separare le lineage, allenando solo sulle architetture viste e
testando sui generatori mai visti.

| dataset | famiglia | AUC visti | AUC mai visti |
|---|---|---|---|
| grid128 | radial | 0.8963 | 0.8499 |
| grid128 | dct | 0.9897 | 0.9804 |
| grid128 | color | 0.8905 | 0.6856 |
| grid128 | residual | 0.9989 | 0.9787 |
| grid128 | all | 1.0000 | 0.9755 |
| grid16 | radial | 0.6184 | 0.5276 |
| grid16 | dct | 0.9126 | 0.7551 |
| grid16 | color | 0.8111 | 0.6753 |
| grid16 | residual | 0.9022 | 0.7182 |
| grid16 | all | 0.9688 | 0.7608 |
| grid32 | radial | 0.7924 | 0.7090 |
| grid32 | dct | 0.9692 | 0.8736 |
| grid32 | color | 0.8318 | 0.6871 |
| grid32 | residual | 0.9681 | 0.8708 |
| grid32 | all | 0.9998 | 0.8472 |
| grid64 | radial | 0.8949 | 0.8579 |
| grid64 | dct | 0.9838 | 0.9489 |
| grid64 | color | 0.8849 | 0.7341 |
| grid64 | residual | 0.9894 | 0.9563 |
| grid64 | all | 1.0000 | 0.9419 |
| patch128 | radial | 0.8430 | 0.7814 |
| patch128 | dct | 0.9675 | 0.9943 |
| patch128 | color | 0.8646 | 0.6605 |
| patch128 | residual | 0.9672 | 0.9934 |
| patch128 | all | 0.9882 | 0.9700 |
| patch16 | radial | 0.7349 | 0.7303 |
| patch16 | dct | 0.8447 | 0.7843 |
| patch16 | color | 0.7533 | 0.6001 |
| patch16 | residual | 0.8463 | 0.7890 |
| patch16 | all | 0.8956 | 0.7840 |
| patch32 | radial | 0.7739 | 0.7840 |
| patch32 | dct | 0.8936 | 0.8657 |
| patch32 | color | 0.7796 | 0.5964 |
| patch32 | residual | 0.8952 | 0.8669 |
| patch32 | all | 0.9309 | 0.8542 |
| patch64 | radial | 0.7777 | 0.7169 |
| patch64 | dct | 0.9212 | 0.9331 |
| patch64 | color | 0.8016 | 0.6352 |
| patch64 | residual | 0.9145 | 0.9293 |
| patch64 | all | 0.9499 | 0.9105 |
| pm128 | radial | 0.8912 | 0.8930 |
| pm128 | dct | 0.9979 | 0.9958 |
| pm128 | color | 0.8963 | 0.6760 |
| pm128 | residual | 0.9994 | 0.9932 |
| pm128 | all | 1.0000 | 0.9846 |
| raw | radial | 0.9834 | 0.9618 |
| raw | dct | 1.0000 | 0.9997 |
| raw | color | 0.9039 | 0.6895 |
| raw | residual | 1.0000 | 0.9991 |
| raw | all | 1.0000 | 0.9999 |
| realonly256 | radial | 0.7997 | 0.8288 |
| realonly256 | dct | 0.9999 | 0.8408 |
| realonly256 | color | 0.8799 | 0.6593 |
| realonly256 | residual | 1.0000 | 0.9100 |
| realonly256 | all | 1.0000 | 0.8810 |
| realonlygrid16 | radial | 0.6045 | 0.5016 |
| realonlygrid16 | dct | 0.9334 | 0.7285 |
| realonlygrid16 | color | 0.8421 | 0.6403 |
| realonlygrid16 | residual | 0.9400 | 0.6502 |
| realonlygrid16 | all | 1.0000 | 0.7276 |
| realonlygrid64 | radial | 0.8222 | 0.8341 |
| realonlygrid64 | dct | 0.9918 | 0.8594 |
| realonlygrid64 | color | 0.8710 | 0.6903 |
| realonlygrid64 | residual | 0.9980 | 0.8570 |
| realonlygrid64 | all | 1.0000 | 0.8729 |

## Controllo del CONTENUTO: baseline zero-shot (nessun addestramento)

Feature grezze di un VLM congelato; i centroidi di lineage sono calcolati
SOLO sulle immagini reali. Misura quanto la sola somiglianza semantica
visibile spiega il risultato.

| dataset | backbone | AUC (soli generati) | noise-GAN corretti |
|---|---|---|---|
| grid128 | `clip_vit_l14` | 0.9844 | 0.961 |
| grid16 | `clip_vit_l14` | 0.7638 | 0.738 |
| grid32 | `clip_vit_l14` | 0.8959 | 0.849 |
| grid64 | `clip_vit_l14` | 0.9050 | 0.956 |
| patch128 | `clip_vit_l14` | 0.9937 | 0.967 |
| patch16 | `clip_vit_l14` | 0.6105 | 0.708 |
| patch32 | `clip_vit_l14` | 0.7223 | 0.651 |
| patch64 | `clip_vit_l14` | 0.8799 | 0.883 |
| pm128 | `clip_vit_l14` | 0.9945 | 0.981 |
| raw | `clip_vit_l14` | 0.9976 | 0.992 |
| realonly256 | `clip_vit_l14` | 0.9925 | 0.969 |
| realonlygrid16 | `clip_vit_l14` | 0.7714 | 0.755 |
| realonlygrid64 | `clip_vit_l14` | 0.9013 | 0.948 |

## Deep vs pavimento

| dataset | backbone | AUC visti | AUC mai visti | acc mai visti | noise-GAN corretti | editing-GAN corretti | pavimento | delta |
|---|---|---|---|---|---|---|---|---|
| grid128 | `resnet18` | 1.0000 | 1.0000 | 1.000 | 0.999 | 1.000 | 0.9804 | +0.0196 |
| grid16 | `resnet18` | 0.9810 | 0.9698 | 0.899 | 0.939 | 0.879 | 0.7608 | +0.2091 |
| grid32 | `resnet18` | 0.9996 | 0.9977 | 0.975 | 0.990 | 0.967 | 0.8736 | +0.1240 |
| grid64 | `resnet18` | 1.0000 | 1.0000 | 0.992 | 0.999 | 0.989 | 0.9563 | +0.0436 |
| patch128 | `resnet18` | 1.0000 | 1.0000 | 0.998 | 0.995 | 1.000 | 0.9943 | +0.0057 |
| patch16 | `resnet18` | 0.9286 | 0.8841 | 0.770 | 0.869 | 0.721 | 0.7890 | +0.0951 |
| patch32 | `resnet18` | 0.9918 | 0.9819 | 0.902 | 0.965 | 0.870 | 0.8669 | +0.1149 |
| patch64 | `resnet18` | 0.9999 | 0.9990 | 0.984 | 0.989 | 0.982 | 0.9331 | +0.0659 |
| pm128 | `clip_vit_b16` | 1.0000 | 1.0000 | 1.000 | 1.000 | 1.000 | 0.9958 | +0.0042 |
| pm128 | `clip_vit_l14` | 1.0000 | 1.0000 | 0.998 | 0.995 | 1.000 | 0.9958 | +0.0042 |
| pm128 | `dinov2_vitb14` | 1.0000 | 1.0000 | 0.999 | 1.000 | 0.999 | 0.9958 | +0.0042 |
| pm128 | `dinov2_vitl14` | 0.9999 | 1.0000 | 0.998 | 0.998 | 0.999 | 0.9958 | +0.0042 |
| pm128 | `resnet18` | 1.0000 | 1.0000 | 0.999 | 0.998 | 1.000 | 0.9958 | +0.0042 |
| pm128 | `resnet18_hp` | 1.0000 | 1.0000 | 1.000 | 1.000 | 1.000 | 0.9958 | +0.0042 |
| pm128 | `resnet50` | 1.0000 | 1.0000 | 1.000 | 1.000 | 1.000 | 0.9958 | +0.0042 |
| raw | `clip_vit_l14` | 1.0000 | 1.0000 | 1.000 | 1.000 | 1.000 | 0.9999 | +0.0001 |
| realonly256 | `resnet18` | 1.0000 | 1.0000 | 1.000 | 1.000 | 1.000 | 0.9100 | +0.0900 |
| realonlygrid16 | `resnet18` | 0.9561 | 0.9446 | 0.847 | 0.910 | 0.805 | 0.7285 | +0.2161 |
| realonlygrid64 | `resnet18` | 1.0000 | 0.9991 | 0.984 | 0.999 | 0.973 | 0.8729 | +0.1262 |

**Come si legge.** `delta` e' il margine del modello deep sopra un
descrittore handcrafted banale. Un delta vicino a zero sul dataset
originale significa che il risultato e' spiegato dalla catena di
ricampionamento; il confronto fra la riga `raw` e la riga `pm128`
(stessa catena per tutte le classi) dice quanta parte del segnale
sopravvive alla rimozione del confound.

`noise-GAN corretti` e' l'evidenza forte (generano da rumore: il segnale
puo' venire solo dai dati di addestramento); `editing-GAN corretti` e'
evidenza debole (il loro output contiene gia' i pixel dell'immagine reale
di input).

**Il confronto che conta.** Il modello addestrato va letto contro DUE
riferimenti, non uno: il pavimento handcrafted (quanto basta una firma di
basso livello) e il baseline zero-shot (quanto basta la somiglianza
semantica visibile). Un risultato e' forense solo nella misura in cui
supera ENTRAMBI.
