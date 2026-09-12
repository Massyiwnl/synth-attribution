# Sweep sulla dimensione della patch

Quota di immagini attribuite alla lineage corretta. `256` = immagine intera.
Tutti e tre i metodi valutati sullo stesso dato, dataset a ricampionamento
normalizzato (pm128).

## StyleGAN3 - evidenza forte (pannello B)

Unico noise-GAN held-out: genera da rumore, quindi il segnale puo' venire
solo dai dati di addestramento.

| px | deep | handcrafted | zero-shot | deep - zero-shot | famiglia pavimento |
|---|---|---|---|---|---|
| 16 | 86.9% | 73.3% | 71.2% | +15.6 pt | `residual` |
| 32 | 96.5% | 80.1% | 63.6% | +32.9 pt | `residual` |
| 64 | 98.9% | 86.2% | 86.8% | +12.1 pt | `dct` |
| 128 | 99.5% | 92.4% | 96.1% | +3.4 pt | `dct` |
| 256 | 99.8% | 94.2% | 97.8% | +2.0 pt | `dct` |

## Tutti i generatori mai visti (pannello A)

Media su AttGAN, GDWCT, StyleGAN3.

| px | deep | handcrafted | zero-shot |
|---|---|---|---|
| 16 | 77.0% | 71.1% | 56.5% |
| 32 | 90.2% | 78.6% | 65.0% |
| 64 | 98.4% | 86.2% | 88.8% |
| 128 | 99.8% | 96.8% | 95.5% |
| 256 | 99.9% | 96.8% | 96.2% |

## Come si legge

La colonna `deep - zero-shot` e' il punto: misura quanto il modello
addestrato sa che la sola somiglianza semantica non sa. Sull'immagine
intera e' vicina a zero - li' le due spiegazioni sono indistinguibili e
il risultato non e' attribuibile a una traccia forense. Man mano che la
finestra si stringe il divario si apre: e' la prova che esiste un canale
di basso livello indipendente dal contenuto.
