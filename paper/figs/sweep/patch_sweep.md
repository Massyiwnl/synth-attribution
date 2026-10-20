# Sweep sulla dimensione della patch

Quota di immagini attribuite alla lineage corretta. `256` = immagine intera.
Tutti e tre i metodi valutati sullo stesso dato, dataset a ricampionamento
normalizzato (pm128).

## StyleGAN3 - evidenza forte (pannello B)

Unico noise-GAN held-out: genera da rumore, quindi il segnale puo' venire
solo dai dati di addestramento.

| px | deep | handcrafted | zero-shot | deep - zero-shot | famiglia pavimento |
|---|---|---|---|---|---|
| 16 | 93.9% | 75.0% | 73.5% | +20.4 pt | `all` |
| 32 | 99.0% | 82.2% | 82.2% | +16.8 pt | `dct` |
| 64 | 99.9% | 86.8% | 95.1% | +4.7 pt | `residual` |
| 128 | 99.9% | 88.2% | 95.1% | +4.7 pt | `dct` |
| 256 | 99.8% | 94.2% | 97.8% | +2.0 pt | `dct` |

## Tutti i generatori mai visti (pannello A)

Media su AttGAN, GDWCT, StyleGAN3.

| px | deep | handcrafted | zero-shot |
|---|---|---|---|
| 16 | 89.9% | 67.6% | 66.3% |
| 32 | 97.5% | 77.2% | 87.4% |
| 64 | 99.2% | 88.3% | 94.0% |
| 128 | 100.0% | 93.2% | 96.0% |
| 256 | 99.9% | 96.8% | 96.2% |

## Come si legge

La colonna `deep - zero-shot` e' il punto: misura quanto il modello
addestrato sa che la sola somiglianza semantica non sa. Sull'immagine
intera e' vicina a zero - li' le due spiegazioni sono indistinguibili e
il risultato non e' attribuibile a una traccia forense. Man mano che la
finestra si stringe il divario si apre: e' la prova che esiste un canale
di basso livello indipendente dal contenuto.
