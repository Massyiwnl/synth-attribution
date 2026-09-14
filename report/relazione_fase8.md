# Attribuzione dei dati di addestramento di immagini sintetiche

|  |  |
|---|---|
| Progetto | **synth-attribution** - Fase 8 |
| Contesto | Estensione sperimentale verso un articolo a impianto giuridico |
| Base di partenza | Cassia et al., *Deepfake Forensic Analysis: Source Dataset Attribution and Legal Implications of Synthetic Media Manipulation*, arXiv:2505.11110 (tesi triennale) + progetto di Multimedia Forensics (LM-18, Catania) |
| Dataset | 21.000 immagini, 7 classi, 2 lineage, 256x256 |
| Esperimenti | 19 modelli addestrati, 13 pavimenti handcrafted, 13 baseline zero-shot, 2 analisi Grad-CAM |


> **Risultato in una frase.** Esiste un canale di basso livello, indipendente dal contenuto visibile e dalla catena di ricampionamento, che lega un'immagine generata al dataset su cui il generatore è stato addestrato, e che **trasferisce a generatori mai visti in addestramento**: su un riquadro 16x16 in posizione fissa, dove la somiglianza semantica arriva al 73.5% e un descrittore spettrale al 75.0%, la metrica addestrata attribuisce correttamente il 93.9% delle immagini di StyleGAN3 - un generatore che non ha mai visto e che, partendo da rumore, dai dati di addestramento ha ereditato tutto.


## 1. Contesto e domanda di ricerca

Il dibattito giuridico sulle immagini sintetiche si è concentrato a lungo su **chi sia titolare dell'opera finale**. Questo lavoro parte dall'osservazione che il problema si sposta **a monte**: ciò che conta non è l'output, ma **quali dati sono stati usati per addestrare il modello**. Se un volto, una fotografia o un'opera sono finiti in un training set senza consenso, il danno si è prodotto prima e altrove rispetto all'immagine generata.

Perché quella pretesa sia azionabile serve un modo tecnico per sostenerla. La domanda di ricerca della Fase 8 è quindi: **data un'immagine sintetica, è possibile risalire al dataset su cui il generatore che l'ha prodotta è stato addestrato?** E soprattutto: questa capacità regge su generatori che il sistema non ha mai visto?


### 1.1 Tre pretese diverse, da non confondere

La formulazione precisa del claim è decisiva, perché tre affermazioni vicine hanno peso tecnico e giuridico molto diverso:


|  | Affermazione | Difficolta | Che cosa fonda |
|---|---|---|---|
| (a) | questa immagine viene dal modello X | media | attribuzione del modello |
| (b) | questa immagine viene da un modello **addestrato su D** | alta | **è ciò che questo lavoro dimostra** |
| (c) | **questa specifica immagine** era nel training set | altissima | membership individuale; non affrontata qui |

Il lavoro consegna **(b)**. La distinzione da **(c)** va tenuta esplicita: l'evidenza non prova che una particolare fotografia fosse nel training set, ma fonda una **presunzione sull'origine dei dati**, cioè la base per chiedere accesso e verifica. È una differenza che rafforza, non indebolisce, l'uso processuale del risultato: dice esattamente ciò che può dire, né più né meno.


## 2. Che cosa esisteva prima, e perché serve


### 2.1 La tesi triennale: attribuzione del dataset con feature handcrafted

Il lavoro di partenza (arXiv:2505.11110) affronta l'attribuzione del **dataset di addestramento** combinando trasformate spettrali (Fourier e DCT), statistiche di colore e descrittori locali (SIFT), con classificatori supervisionati (Random Forest, SVM, XGBoost). Ottiene **98-99% di accuratezza** sia nella distinzione reale/sintetico sia nell'attribuzione multi-classe del dataset, su CelebA e FFHQ con cinque architetture GAN. Il risultato chiave è la **dominanza delle feature in frequenza**, che catturano artefatti di upsampling e irregolarità spettrali.

Il limite, ed è il punto da cui parte l'estensione richiesta, è la **formulazione del problema**: un task di classificazione piatto in cui CelebA e FFHQ sono due classi come le altre. Il modello non modella in alcun modo la *relazione* fra un'immagine reale di un dataset e le immagini generate da un modello addestrato su quel dataset: impara confini di decisione fra etichette, non una nozione di appartenenza. E quei confini valgono soltanto per le classi viste in addestramento.


### 2.2 Il progetto di Multimedia Forensics: attribuzione del generatore

Il progetto successivo sposta il focus dal dataset al **generatore** (distinguere reale, StyleGAN2 e StyleGAN3 a parità di famiglia di volti) e introduce il **metric learning siamese**: un encoder ResNet18 con testa di proiezione e *contrastive loss*, che impara uno spazio in cui la distanza riflette la somiglianza di sorgente. L'enfasi si sposta sulla **generalizzazione open-set**.

I risultati, per quanto serve qui:

- **In-distribution**: FFHQ reale vs StyleGAN2, AUC 0.952 (0.963 con front-end a residuo); reale vs StyleGAN3, AUC 0.706. Il divario misura l'efficacia del design *alias-free* di StyleGAN3 nel nascondere la propria firma.
- **Generalizzazione cross-architettura**: nulla. Tenendo un generatore fuori dal training, l'AUC focalizzata su di esso resta circa 0.5. Il fallimento è **asimmetrico**: StyleGAN3 non visto viene scambiato per reale (falso negativo, la direzione pericolosa), StyleGAN2 non visto viene comunque marchiato come sintetico.
- **Due confound di dataset** individuati e neutralizzati: un confound di pairing cross-lineage (AUC 1.0 alla prima epoca, perché CelebA e FFHQ differiscono in tutto) e un **confound di ricampionamento** che faceva sembrare il baseline handcrafted capace di generalizzare a circa 0.98; appaiando la pipeline di resize fra reali e sintetici quel valore crolla a 0.51-0.53.

Il secondo confound è il precedente metodologico più importante per la Fase 8: **lo stesso gruppo di ricerca ha già dimostrato, sui propri dati, che un risultato apparentemente forte in questo dominio può essere interamente un artefatto di preprocessing**. Ogni numero che segue è stato perciò sottoposto a controlli espliciti, ed è questa la ragione per cui la parte più estesa del documento riguarda i controlli e non i risultati.


### 2.3 Due indizi, nei risultati precedenti, che indicavano la strada

Rileggendo la fase precedente emergono due fatti che riguardano direttamente l'attribuzione del dataset, e che non erano stati interpretati in quella chiave.

**Primo.** La matrice *leave-one-generator-out* mostra una struttura a blocchi netta: dentro la stessa lineage c'è transfer fra generatori (AUC 0.63-1.00 su CelebA, 0.73-0.79 su FFHQ), mentre **fra lineage diverse il transfer è puro caso** (0.42-0.53). La traccia del generatore è dunque intrecciata con il dataset su cui quel generatore è stato addestrato: sono due informazioni che vivono insieme.

**Secondo.** Il modello finale allenato su tutte le classi raggiunge un'accuratezza top-1 in regime chiuso del 76.3%, con un profilo per classe molto particolare: le quattro classi della lineage CelebA sono riconosciute al 100%, mentre dentro FFHQ il modello si confonde (real_ffhq 55%, StyleGAN2 58%, StyleGAN3 21%). Ma la matrice di confusione mostra un fatto che vale più dell'accuratezza aggregata: **non c'è un solo errore cross-lineage**. Tutta la confusione è interna a FFHQ. Il lignaggio dei dati era perciò perfettamente separabile proprio mentre il generatore non lo era.


![Fase precedente, attribuzione del generatore in regime chiuso (top-1 = 0.763). Nessuna immagine della famiglia CelebA viene mai attribuita a una classe FFHQ, e viceversa: i due blocchi 4x4 e 3x3 fuori diagonale sono interamente a zero. È l'osservazione da cui parte la Fase 8.](report/final/confusion.png)

*Fase precedente, attribuzione del generatore in regime chiuso (top-1 = 0.763). Nessuna immagine della famiglia CelebA viene mai attribuita a una classe FFHQ, e viceversa: i due blocchi 4x4 e 3x3 fuori diagonale sono interamente a zero. È l'osservazione da cui parte la Fase 8.*


![Lo stesso spazio di embedding in proiezione PCA. La prima componente principale separa le due lineage: a sinistra i quattro cluster compatti della famiglia CelebA, a destra un unico blocco in cui le tre classi FFHQ si sovrappongono. La direzione di massima varianza dello spazio è il dataset di addestramento, non il generatore.](report/final/pca_final.png)

*Lo stesso spazio di embedding in proiezione PCA. La prima componente principale separa le due lineage: a sinistra i quattro cluster compatti della famiglia CelebA, a destra un unico blocco in cui le tre classi FFHQ si sovrappongono. La direzione di massima varianza dello spazio è il dataset di addestramento, non il generatore.*


## 3. Il dataset

21.000 immagini a 256x256, 7 classi da 3.000 immagini, 2 **lineage** (famiglie di volti definite dal dataset di addestramento). La struttura è il fulcro dell'intero esperimento: ogni lineage contiene le immagini **reali** del dataset e le immagini prodotte da generatori **addestrati su quel dataset**.


| lineage | classe | tipo | origine |
|---|---|---|---|
| celeba | real_celeba | reale | CelebA, crop centrale + Lanczos a 256 |
| celeba | stargan | editing-GAN | StarGAN (yunjey), edit Blond_Hair, 128 -> 256 |
| celeba | attgan | editing-GAN | AttGAN (elvisyjlin), edit Blond_Hair, 128 -> 256 |
| celeba | gdwct | editing-GAN | GDWCT (WonwoongCho), 216 -> 256 |
| ffhq | real_ffhq | reale | FFHQ-1024 -> 256 (Lanczos), pipeline appaiata |
| ffhq | stylegan2 | noise-GAN | StyleGAN2-ADA FFHQ, pickle ufficiale, 1024 -> 256 |
| ffhq | stylegan3 | noise-GAN | StyleGAN3-t FFHQ, pickle ufficiale, 1024 -> 256 |


### 3.1 Editing-GAN e noise-GAN: la distinzione che regge tutta la tesi

Le classi sintetiche non sono dello stesso tipo, e la differenza decide quanto vale l'evidenza.

- **Editing-GAN** (StarGAN, AttGAN, GDWCT): modificano una fotografia reale esistente. Il loro output **contiene i pixel della foto CelebA di input**. Se il sistema le attribuisce a CelebA, non ha dimostrato che il training set fosse CelebA: ha dimostrato che *l'input* lo era. Evidenza **debole**, e va dichiarata come tale.
- **Noise-GAN** (StyleGAN2, StyleGAN3): generano da rumore casuale. Non esiste alcuna immagine di input. Qualsiasi somiglianza con FFHQ può provenire **soltanto** dai dati di addestramento. Evidenza **forte**, ed è su queste che si misura il risultato.

Per questo, in tutto il documento, la colonna che conta è quella di **StyleGAN3**: è l'unico noise-GAN tenuto fuori dall'addestramento. Le righe di AttGAN e GDWCT restano come controllo positivo, non come prova.


![Campioni del dataset, una riga per classe. A 256x256 le classi sono visivamente molto simili: le differenze fra generatori vivono in tracce non percepibili a occhio. Si noti però la differenza di inquadratura e di fondale fra le due lineage - una caratteristica di contenuto che tornerà come principale spiegazione concorrente.](report/figures/dataset_samples.png)

*Campioni del dataset, una riga per classe. A 256x256 le classi sono visivamente molto simili: le differenze fra generatori vivono in tracce non percepibili a occhio. Si noti però la differenza di inquadratura e di fondale fra le due lineage - una caratteristica di contenuto che tornerà come principale spiegazione concorrente.*


## 4. Metodo


### 4.1 Riformulazione del task: pairing per lineage

La rete siamese non classifica: impara una **funzione di embedding** in cui la distanza fra due immagini riflette la loro appartenenza alla stessa sorgente. Ciò che determina *cosa* impara è la definizione di coppia genuina. La fase precedente usava il **pairing per architettura** (genuina = stesso generatore), che spinge deliberatamente *lontano* real_celeba da StarGAN.

La Fase 8 introduce il **pairing per lineage**: due immagini formano una coppia genuina se appartengono alla stessa lineage, **indipendentemente dal generatore**. real_celeba, StarGAN, AttGAN e GDWCT sono quindi tutti *positivi fra loro*; real_ffhq, StyleGAN2 e StyleGAN3 fra loro; le coppie impostore sono cross-lineage. La rete è così costretta a cercare **ciò che un dataset e i modelli addestrati su di esso hanno in comune**, ignorando la firma del singolo generatore. È esattamente la relazione che il task di classificazione piatto della tesi triennale non poteva modellare.


**Che cosa cambia, concretamente, per i dati reali**

È il punto che distingue questo lavoro dal precedente, e vale la pena renderlo misurabile. Nel task di classificazione piatto della tesi triennale **real_celeba e StarGAN erano due classi diverse**: la supervisione insegnava a separarle, quindi il modello non poteva costruire alcuna nozione di appartenenza fra un dataset e i modelli addestrati su di esso. Con il pairing per lineage sono la **stessa classe**, e la loss le avvicina esplicitamente.


|  | valore |
|---|---|
| immagini nel training | 8400 |
| di cui reali | **4200 (50%)** |
| di cui generate | 4200 (50%) |
| coppie genuine (generata, generata) | 1045 (25.2%) |
| coppie genuine (reale, generata) | **2123 (51.1%)** |
| coppie genuine (reale, reale) | 983 (23.7%) |


> **Il 51.1% delle coppie che la rete impara ad avvicinare è formato da una fotografia autentica e un'immagine generata.** I dati reali non sono una classe fra le altre: sono l'**ancoraggio** rispetto a cui tutto il resto viene misurato. Nel task piatto precedente quella stessa coppia era un esempio **negativo**. Misurabile con `python -m src.eval.pairing_stats`.

Nota tecnica: con questa politica le coppie impostore sono per costruzione cross-lineage, quindi i *bucket* di valutazione within-lineage della fase precedente degenerano. La metrica corretta di selezione del modello migliore diventa l'AUC complessiva - il contrario di quanto valeva per l'attribuzione del generatore, dove l'AUC complessiva era gonfiata dal confound cross-lineage.


### 4.2 Architettura: backbone pre-addestrato congelato + testa leggera

L'encoder è composto da tre parti: un **front-end opzionale** (identità, oppure un filtro high-pass o SRM che sopprime il contenuto), un **backbone pre-addestrato** di cui si usa soltanto l'embedding finale, e una **testa di proiezione** addestrabile (lineare - ReLU - dropout - lineare) seguita da normalizzazione L2. L'embedding finale vive su un'ipersfera unitaria a 128 dimensioni.

Nel regime principale il backbone è **congelato**: i suoi pesi non vengono aggiornati e il modulo resta in modalità di valutazione anche durante l'addestramento (altrimenti BatchNorm e dropout continuerebbero a modificarsi e il backbone non sarebbe davvero fisso). Con CLIP ViT-L/14 si addestrano **0.59 milioni di parametri su 303.77**, cioè lo 0.19%.


> **Dettaglio forense sulla risoluzione.** I backbone ViT attendono ingressi a 224 pixel, il dataset è a 256. Un resize 256 -> 224 introdurrebbe un **ulteriore ricampionamento**, cioè esattamente il tipo di artefatto che questo lavoro cerca di non misurare. Per questo il default per i backbone congelati è il **crop centrale** a 224, che preserva le statistiche dei pixel senza reinterpolare nulla. Le ResNet, essendo completamente convoluzionali, ricevono i 256 nativi.


### 4.3 Funzione di perdita

Contrastive loss di Hadsell, Chopra e LeCun. Per una coppia con etichetta y (1 = stessa lineage, 0 = lineage diverse) e distanza euclidea d fra gli embedding:

con margine m = 1.0. Poiché gli embedding sono L2-normalizzati, la distanza vive in [0, 2]. Le coppie genuine vengono avvicinate, le impostore allontanate fino almeno al margine; oltre il margine le impostore non contribuiscono più.


### 4.4 Protocollo di holdout

Tre generatori sono **completamente esclusi dall'addestramento**: tutte le loro immagini finiscono nel test split, zero in training e zero in validazione. Lo split è deterministico per gruppo, quindi le classi non in holdout ricevono esattamente lo stesso split degli esperimenti precedenti e i risultati restano confrontabili.


|  | in addestramento | solo in test (mai visti) |
|---|---|---|
| lineage celeba | real_celeba, stargan | **attgan, gdwct** (editing) |
| lineage ffhq | real_ffhq, stylegan2 | **stylegan3** (noise) |

I centroidi di lineage usati per decidere sono calcolati **soltanto sulle classi viste**: un centroide che includesse le classi held-out userebbe informazione che in un caso reale non sarebbe disponibile.


## 5. Risultati principali: lo spazio di embedding


### 5.1 Le immagini si raggruppano per dataset di addestramento

Dato un insieme di immagini di test, si estraggono gli embedding e si proiettano in due dimensioni. La domanda posta all'inizio era: *se le clusterizza in qualche modo, significa che stanno ricadendo tutti nello stesso cluster*. La risposta è affermativa e netta.


![Proiezione PCA dello spazio di embedding appreso con pairing per lineage (CLIP ViT-L/14 congelato). I cerchi sono le classi viste in addestramento, le croci i generatori mai visti. Le sette classi collassano in due soli gruppi, corrispondenti alle due lineage: real_celeba, StarGAN, AttGAN e GDWCT da una parte, real_ffhq, StyleGAN2 e StyleGAN3 dall'altra. AttGAN, GDWCT e StyleGAN3 cadono nel gruppo corretto pur non essendo mai stati visti.](report/lineage_raw_clip_vit_l14/pca_lineage.png)

*Proiezione PCA dello spazio di embedding appreso con pairing per lineage (CLIP ViT-L/14 congelato). I cerchi sono le classi viste in addestramento, le croci i generatori mai visti. Le sette classi collassano in due soli gruppi, corrispondenti alle due lineage: real_celeba, StarGAN, AttGAN e GDWCT da una parte, real_ffhq, StyleGAN2 e StyleGAN3 dall'altra. AttGAN, GDWCT e StyleGAN3 cadono nel gruppo corretto pur non essendo mai stati visti.*


### 5.2 Distanze fra vettori medi: la verifica quantitativa

La formulazione iniziale chiedeva: *prendiamo i vettori di feature di CelebA anche in media, prendiamo quelli di StarGAN, e calcoliamo una distanza; facciamo uguale tra CelebA e StyleGAN e mi aspetto che sia più grande*. La matrice delle distanze medie fra classi lo conferma con un ordine di grandezza di margine.


|  | attgan | gdwct | real_cele | real_ffhq | stargan | stylegan2 | stylegan3 |
|---|---|---|---|---|---|---|---|
| **attgan*** | 0.113 | 0.114 | 0.121 | 1.074 | 0.098 | 1.082 | 1.077 |
| **gdwct*** | 0.114 | 0.082 | 0.098 | 1.132 | 0.066 | 1.140 | 1.135 |
| **real_celeba** | 0.121 | 0.098 | 0.109 | 1.113 | 0.079 | 1.121 | 1.116 |
| **real_ffhq** | 1.074 | 1.132 | 1.113 | 0.078 | 1.130 | 0.070 | 0.076 |
| **stargan** | 0.098 | 0.066 | 0.079 | 1.130 | 0.038 | 1.138 | 1.133 |
| **stylegan2** | 1.082 | 1.140 | 1.121 | 0.070 | 1.138 | 0.061 | 0.067 |
| **stylegan3*** | 1.077 | 1.135 | 1.116 | 0.076 | 1.133 | 0.067 | 0.073 |


### 5.3 Una decisione con tasso d'errore noto

Una soglia senza un tasso d'errore misurato non è utilizzabile come evidenza. Il protocollo è quindi: per ogni immagine si calcola la distanza dai due centroidi di lineage; lo score è la differenza fra le due distanze; la soglia è fissata al punto di **Equal Error Rate sulle sole classi viste** in addestramento, e poi applicata **invariata** ai generatori mai visti.


| metrica | classi viste | generatori mai visti |
|---|---|---|
| AUC | 1.0000 | 1.0000 |
| accuratezza | 99.9% | 99.8% |
| EER (calibrazione) | 0.1% | - |
| falsi positivi (celeba -> ffhq) | - | 0.0% |
| falsi negativi (ffhq -> celeba) | - | 0.5% |


![Distribuzione dello score di lineage. Le aree piene sono le classi viste in addestramento, i contorni i generatori mai visti. La linea tratteggiata è la soglia calibrata sulle sole classi viste. I generatori mai visti cadono dalla parte corretta con lo stesso ampio margine delle classi viste: fra le due popolazioni resta una fascia vuota di circa 0.5 unità.](report/lineage_pm128_clip_vit_l14/score_distribution.png)

*Distribuzione dello score di lineage. Le aree piene sono le classi viste in addestramento, i contorni i generatori mai visti. La linea tratteggiata è la soglia calibrata sulle sole classi viste. I generatori mai visti cadono dalla parte corretta con lo stesso ampio margine delle classi viste: fra le due popolazioni resta una fascia vuota di circa 0.5 unità.*


![Distanza media dai due centroidi di lineage, per classe. Ogni classe è molto vicina al centroide della propria lineage e molto lontana dall'altro; le tre classi marcate come mai viste si comportano come quelle viste.](report/lineage_pm128_clip_vit_l14/centroid_distances.png)

*Distanza media dai due centroidi di lineage, per classe. Ogni classe è molto vicina al centroide della propria lineage e molto lontana dall'altro; le tre classi marcate come mai viste si comportano come quelle viste.*


### 5.4 I generatori mai visti

È il risultato richiesto: *prendendo AttGAN e GDWCT che non abbiamo messo nel training, li mettiamo fuori nel test set, cioè dati mai visti, e mi aspetto che ci siano distanze basse*. Con l'aggiunta di StyleGAN3, che è l'evidenza forte.


| classe | lineage | visto | tipo | dist. centroide celeba | dist. centroide ffhq | attribuiti bene |
|---|---|---|---|---|---|---|
| **attgan** | celeba | **NO** | editing | 0.087 | 1.078 | **100.0%** |
| **gdwct** | celeba | **NO** | editing | 0.061 | 1.135 | **100.0%** |
| real_celeba | celeba | si | reale | 0.073 | 1.116 | **100.0%** |
| real_ffhq | ffhq | si | reale | 1.120 | 0.053 | **99.8%** |
| stargan | celeba | si | editing | 0.029 | 1.133 | **100.0%** |
| stylegan2 | ffhq | si | noise | 1.128 | 0.043 | **100.0%** |
| **stylegan3** | ffhq | **NO** | noise | 1.123 | 0.050 | **99.5%** |


> **StyleGAN3, mai visto in addestramento e generatore da rumore, viene ricondotto a FFHQ nel 99.5% dei casi**, con distanza 1.123 dal centroide CelebA e 0.050 da quello FFHQ. Non avendo mai ricevuto un'immagine di input, l'unica origine possibile di quella somiglianza sono i dati su cui è stato addestrato.


## 6. I controlli: perché il risultato non è un artefatto

I numeri della sezione precedente, da soli, **non dimostrerebbero nulla**. Sull'immagine intera il risultato è **sovra-determinato**: almeno tre meccanismi diversi produrrebbero esattamente la stessa prestazione, e due di essi non hanno nulla a che vedere con una traccia forense. Questa sezione li isola e li misura uno per uno. È la parte più importante del lavoro.


### 6.1 Primo sospetto: la catena di ricampionamento

Nel dataset originale le due lineage hanno subito trattamenti opposti: **tutta** la famiglia CelebA è stata ingrandita verso 256 (da 178x218, da 128 nativi degli editing-GAN, da 216 di GDWCT), mentre **tutta** la famiglia FFHQ è stata rimpicciolita da 1024 con Lanczos. Il ricampionamento coincide perciò perfettamente con l'etichetta da predire.

Quanto pesa è misurabile. Un **singolo scalare** - l'energia media dello spettro di potenza alle alte frequenze, calcolato senza alcun apprendimento - separa le due lineage con **AUC 0.929** per immagine (n = 1050). Le medie di classe non si sovrappongono: la famiglia CelebA sta fra 13.0 e 13.5, quella FFHQ fra 14.4 e 14.6.


![Spettro di potenza azimutale medio per classe, zoom sulle alte frequenze, sul dataset originale. Le tre classi della lineage FFHQ stanno sopra, le quattro CelebA sotto: uno scalino di energia netto, senza sovrapposizione fra i gruppi. È la firma della catena di ricampionamento, e da sola basta a separare le lineage.](report/signatures_raw/radial_spectrum_hf.png)

*Spettro di potenza azimutale medio per classe, zoom sulle alte frequenze, sul dataset originale. Le tre classi della lineage FFHQ stanno sopra, le quattro CelebA sotto: uno scalino di energia netto, senza sovrapposizione fra i gruppi. È la firma della catena di ricampionamento, e da sola basta a separare le lineage.*

**Il controllo.** L'intero dataset (21.000 immagini) è stato ricostruito facendo passare ogni classe per la stessa identica catena: 256 -> 128 -> 256 con Lanczos, più riscrittura nello stesso formato. Dopo questo passaggio la firma di ricampionamento è comune a tutte le classi e non può più portare informazione sulla lineage.


### 6.2 Il pavimento handcrafted

Il secondo controllo risponde a: **quanto di questo risultato otterrebbe un descrittore banale?** Per ogni famiglia di feature (spettro radiale FFT, DCT 2D poolata, istogrammi di colore RGB e HSV, residuo high-pass) si allena una semplice regressione logistica a predire la lineage, **con lo stesso protocollo**: solo le classi viste in addestramento, poi test sui generatori mai visti.


| dataset | famiglia | dim. | AUC classi viste | AUC mai visti |
|---|---|---|---|---|
| originale | radial | 128 | 0.9834 | **0.9618** |
| originale | dct | 256 | 1.0000 | **0.9997** |
| originale | color | 192 | 0.9039 | **0.6895** |
| originale | residual | 384 | 1.0000 | **0.9991** |
| originale | all | 576 | 1.0000 | **0.9999** |
| normalizzato | radial | 128 | 0.8912 | **0.8930** |
| normalizzato | dct | 256 | 0.9979 | **0.9958** |
| normalizzato | color | 192 | 0.8963 | **0.6760** |
| normalizzato | residual | 384 | 0.9994 | **0.9932** |
| normalizzato | all | 576 | 1.0000 | **0.9846** |


> **Sul dataset originale il pavimento è AUC 0.9999.** Una regressione logistica su 576 numeri fa quanto CLIP ViT-L/14. Il margine del modello addestrato è +0.0001: sull'immagine intera, **l'uso di un VLM non è il contributo**. E sul dataset normalizzato il pavimento resta a 0.9958 - quindi il ricampionamento spiega una parte del segnale, ma non tutto.

Il dettaglio per classe rivela una differenza che l'AUC aggregata nascondeva: sul dataset originale il descrittore handcrafted prende StyleGAN3 al 100% ma **crolla al 63.2% su AttGAN**, dove il modello addestrato fa 100%. L'AUC resta alta perché l'ordinamento è buono anche quando la decisione alla soglia calibrata sbaglia: è un buon esempio di perché un'unica metrica aggregata non basta.


### 6.3 Secondo sospetto: il contenuto semantico

L'obiezione più forte non è tecnica ma di senso comune: *StyleGAN3 finisce su FFHQ perché genera volti che sembrano volti FFHQ*. Inquadratura, allineamento, illuminazione e demografia di CelebA e FFHQ sono diversi, e si vedono a occhio. Se basta quello, il risultato è vero ma non forense: è una somiglianza percettiva.

**Il controllo**, il più diretto possibile: si usano le feature di un VLM pre-addestrato **senza alcun addestramento**. Le rappresentazioni di CLIP sono dominate dalla semantica. I centroidi di lineage sono calcolati **soltanto sulle immagini reali**, cioè su contenuto autentico, e ogni immagine generata viene assegnata al centroide più vicino. Se questo basta, il contenuto spiega tutto.


| dataset | AUC (soli generati) | noise-GAN corretti | StyleGAN3 |
|---|---|---|---|
| originale | 0.9976 | 99.2% | **99.0%** |
| normalizzato | 0.9945 | 98.1% | **97.8%** |


> **Il contenuto semantico, da solo, basta.** Senza allenare niente e senza vedere un solo generatore, la somiglianza visiva fra volti StyleGAN3 e volti FFHQ reali porta StyleGAN3 su FFHQ nel 99.0% dei casi (97.8% sul dataset normalizzato). Sull'immagine intera il modello addestrato, il descrittore handcrafted e la semantica pura sono **indistinguibili**: tutti e tre intorno a 0.99. Il risultato della Sezione 5, da solo, non è attribuibile a una traccia forense.

Un controllo intermedio merita una nota. Allenando lo stesso modello con il **front-end a residuo high-pass**, che sopprime il contenuto a bassa frequenza, sul dataset normalizzato si ottiene AUC 1.0000, accuratezza 100% e FAR e FRR entrambi a zero sui mai visti: sopprimere il contenuto non peggiora nulla. Sembrerebbe la prova che il canale non sia semantico, ma il residuo di un filtro 3x3 conserva ancora il contenuto ad **alta** frequenza - capelli, bordi, struttura del volto. L'esperimento è indicativo, non conclusivo. Serve un controllo più radicale.


## 7. Il risultato centrale: separare i due canali

A questo punto due spiegazioni concorrenti funzionano entrambe e non sono distinguibili: il contenuto semantico da solo basta (99%), e il residuo di basso livello da solo basta (100%). **Nessuna delle due è necessaria.** Finché resta così, non si può sostenere che l'attribuzione sia forense e non percettiva.

L'idea per rompere il pareggio è **ridurre progressivamente la finestra di analisi**. Un ritaglio quadrato piccolo non contiene più inquadratura, composizione, demografia: la semantica svanisce. Ma conserva texture, rumore e le tracce lasciate dal generatore. Se al restringersi della finestra il canale semantico crolla e il modello addestrato tiene, i due canali si separano.

A ogni dimensione si misurano **tre metodi sullo stesso dato**: il modello addestrato, il pavimento handcrafted e il baseline zero-shot. La metrica è la quota di immagini attribuite alla lineage corretta, perché è l'unica grandezza definita in modo identico per tutti e tre (le loro AUC sono calcolate su insiemi diversi e non sarebbero confrontabili). Il backbone è ResNet18, completamente convoluzionale, che riceve la patch a **risoluzione nativa** senza reinterpolare.


### 7.1 Un problema nella selezione delle patch, e la sua correzione

La prima versione dell'esperimento selezionava, fra 16 posizioni casuali, quella a **varianza minima**: è l'accorgimento classico in forense, perché le zone uniformi portano la traccia del generatore senza portare contenuto. Ma la scelta **dipende dal contenuto**, quindi la posizione selezionata potrebbe correlare con la classe.

Il sospetto era fondato, e la figura lo rende evidente: su CelebA quelle patch cadono sul **fondale da studio uniforme**, su FFHQ sull'**erba**. La politica stava campionando preferenzialmente il **background**, che è fra le caratteristiche più discriminanti fra i due dataset. Il controllo si stava sabotando da solo.

La correzione è la politica a **griglia fissa**: quattro posizioni ai centri dei quadranti - (32,32), (160,32), (32,160), (160,160) - **identiche per ogni immagine e per ogni classe**. Nessuna differenza fra classi può derivare da dove si è guardato. Le patch risultanti hanno varianza media doppia (2642 contro 1326): contengono occhi e bocca, non solo pelle liscia.


![Dove guardano le due politiche, patch 64x64. In alto la selezione per varianza minima: le posizioni cambiano da immagine a immagine e cadono sul fondale, che nelle due lineage è sistematicamente diverso (studio contro esterni). In basso la griglia fissa: gli stessi quattro riquadri su ogni immagine, sempre sul volto. La seconda è il controllo valido.](report/figures/patch_positions_64.png)

*Dove guardano le due politiche, patch 64x64. In alto la selezione per varianza minima: le posizioni cambiano da immagine a immagine e cadono sul fondale, che nelle due lineage è sistematicamente diverso (studio contro esterni). In basso la griglia fissa: gli stessi quattro riquadri su ogni immagine, sempre sul volto. La seconda è il controllo valido.*


### 7.2 Lo sweep con griglia fissa: il risultato

Quota di immagini attribuite alla lineage corretta, su **StyleGAN3** (mai visto, noise-GAN). Dataset a ricampionamento normalizzato.


| finestra | deep | handcrafted | zero-shot (semantica) | deep - zero-shot | famiglia pavimento |
|---|---|---|---|---|---|
| 16 | 93.9% | 75.0% | 73.5% | +20.4 pt | all |
| 32 | 99.0% | 82.2% | 82.2% | +16.8 pt | dct |
| 64 | 99.9% | 86.8% | 95.1% | +4.7 pt | residual |
| 128 | 99.9% | 88.2% | 95.1% | +4.7 pt | dct |
| 256 (intera) | 99.8% | 94.2% | 97.8% | +2.0 pt | dct |


![Sweep sulla dimensione della finestra, patch su griglia fissa. A sinistra la media sui tre generatori mai visti, a destra il solo StyleGAN3. Sull'immagine intera le tre curve sono sovrapposte: la somiglianza semantica spiega tutto e il risultato non è attribuibile a una traccia forense. Al restringersi della finestra le curve si separano: la semantica crolla, il modello addestrato tiene.](report/grid_sweep/patch_sweep.png)

*Sweep sulla dimensione della finestra, patch su griglia fissa. A sinistra la media sui tre generatori mai visti, a destra il solo StyleGAN3. Sull'immagine intera le tre curve sono sovrapposte: la somiglianza semantica spiega tutto e il risultato non è attribuibile a una traccia forense. Al restringersi della finestra le curve si separano: la semantica crolla, il modello addestrato tiene.*


> **La dissociazione.** Su un riquadro 16x16 in posizione fissa - nessun fondale, nessuna inquadratura, nessuna composizione - il canale semantico scende al **73.5%** e il descrittore spettrale al **75.0%**, mentre la metrica addestrata resta al **93.9%** su StyleGAN3. Il divario passa da +2.0 punti sull'immagine intera a **+20.4 punti**. Esiste dunque un canale di basso livello, indipendente dal contenuto visibile, che trasferisce a generatori mai visti.

Anche il margine sul pavimento handcrafted è il più ampio misurato in tutto il lavoro: **+0.209 di AUC** a 16 pixel (0.9698 contro 0.7608). Sull'immagine intera era +0.0001.


### 7.3 Lo sweep con selezione per varianza, per confronto

La versione con selezione per varianza minima è riportata come ablazione. I divari sono più ampi, ma vanno letti con la cautela del background: parte di quel vantaggio può derivare dal fatto che le patch cadevano su fondali sistematicamente diversi.


| finestra | deep | handcrafted | zero-shot (semantica) | deep - zero-shot | famiglia pavimento |
|---|---|---|---|---|---|
| 16 | 86.9% | 73.3% | 71.2% | +15.6 pt | residual |
| 32 | 96.5% | 80.1% | 63.6% | +32.9 pt | residual |
| 64 | 98.9% | 86.2% | 86.8% | +12.1 pt | dct |
| 128 | 99.5% | 92.4% | 96.1% | +3.4 pt | dct |
| 256 (intera) | 99.8% | 94.2% | 97.8% | +2.0 pt | dct |

Due osservazioni. Le curve su griglia fissa sono **monotone in entrambi i pannelli**, mentre quelle per varianza hanno un calo non monotono a 32 pixel sulla singola classe StyleGAN3: la griglia è anche il protocollo più stabile. E la famiglia di descrittori che costituisce il pavimento **cambia con la scala** - il residuo high-pass sulle finestre piccole, la DCT 2D su quelle grandi - il che è coerente: sulle finestre piccole conta la texture locale, su quelle grandi la struttura spettrale globale.


## 8. Il caso limite: addestrare sui soli dati autentici

In tutti gli esperimenti visti finora il modello, pur non conoscendo i tre generatori held-out, ne aveva visti **due**: StarGAN e StyleGAN2 erano nel training set. Si può quindi obiettare che il sistema abbia imparato qualcosa sulle immagini generate in quanto tali, e non soltanto sui dati che le hanno prodotte.

Questa sezione elimina l'obiezione nel modo più radicale: il training contiene **esclusivamente immagini autentiche**. Il modello vede solo real_celeba e real_ffhq, impara unicamente a distinguere CelebA autentico da FFHQ autentico, e **non incontra mai una singola immagine generata**. Poi si verifica dove cadono i cinque generatori, tutti quindi mai visti. Con questa configurazione StyleGAN2 e StyleGAN3 diventano **entrambi** evidenza forte: sono noise-GAN, generano da rumore, e nessuno dei due è mai stato mostrato al modello.


> **Perché è la variante decisiva, e non solo la più pulita.** È lo scenario reale di chi lamenta l'uso dei propri dati: **ha soltanto i propri dati**. Non possiede il generatore sospetto, non ha le immagini che ha prodotto, non può addestrare nulla su di esse. Se una metrica costruita sui soli dati autentici riconosce comunque un generatore addestrato su quei dati, la pretesa diventa sostenibile **senza richiedere accesso al modello sospetto** - cioè esattamente nella situazione in cui la parte lesa si trova prima di qualunque richiesta di accesso.


### 8.1 Protocollo


|  | in addestramento | solo in test (mai visti) |
|---|---|---|
| lineage celeba | **real_celeba** (2130 immagini) | stargan, attgan, gdwct |
| lineage ffhq | **real_ffhq** (2070 immagini) | **stylegan2, stylegan3** (noise-GAN) |

Configurazione in `configs/dataset_lineage_realonly.yaml`: tutti e cinque i generatori in `holdout_architectures`. Dataset a ricampionamento normalizzato, quindi quel confound resta neutralizzato. Tre regimi, per poterli confrontare con le curve della Sezione 7.


### 8.2 Risultati


| regime | AUC (5 generatori mai visti) | acc. | StyleGAN3 | noise-GAN (SG2+SG3) | editing-GAN |
|---|---|---|---|---|---|
| immagine intera (256 px) | 1.0000 | 100.0% | **100.0%** | 100.0% | 100.0% |
| patch 64 px, griglia fissa | 0.9991 | 98.4% | **99.9%** | 99.9% | 97.3% |
| patch 16 px, griglia fissa | 0.9446 | 84.7% | **89.2%** | 91.0% | 80.5% |


### 8.3 Confronto con il regime a due generatori visti

La domanda che conta è quanto si perde togliendo i due generatori dal training. Quota di StyleGAN3 - mai visto in entrambi i casi - attribuito alla lineage corretta:


| regime | con StarGAN e StyleGAN2 in training | **con soli dati autentici** | differenza |
|---|---|---|---|
| immagine intera (256 px) | 99.8% | 100.0% | +0.2 pt |
| patch 64 px, griglia fissa | 99.9% | 99.9% | +0.0 pt |
| patch 16 px, griglia fissa | 93.9% | 89.2% | -4.6 pt |


### 8.4 Il pavimento e la semantica, sullo stesso protocollo

Come nelle sezioni precedenti, il numero del modello va letto contro i due riferimenti. Qui entrambi sono calcolati con lo stesso vincolo: la regressione logistica del pavimento è allenata sui soli reali, e il baseline zero-shot usa per costruzione i soli reali come centroidi.


| regime | deep (soli reali) | handcrafted | zero-shot | famiglia pavimento |
|---|---|---|---|---|
| immagine intera (256 px) | **100.0%** | 91.0% | 97.5% | residual |
| patch 64 px, griglia fissa | **99.9%** | 80.2% | 94.1% | all |
| patch 16 px, griglia fissa | **89.2%** | 67.8% | 73.2% | dct |


### 8.5 Lettura

Il risultato risponde a un'obiezione precisa: l'attribuzione non dipende dall'aver visto immagini generate. Un modello che conosce **soltanto come sono fatte le fotografie autentiche** di due dataset riconduce alla sorgente corretta anche generatori che non ha mai incontrato, compresi due noise-GAN che non hanno mai ricevuto un'immagine in ingresso.

C'è però un dato più interessante della semplice tenuta, e riguarda il **costo** che ciascun metodo paga quando i due generatori vengono tolti dal training. Il modello addestrato non perde quasi nulla; il descrittore handcrafted, che senza esempi generati non può più calibrarsi su di essi, perde molto. Il margine quindi **cresce**:


| regime | costo per il deep | costo per l'handcrafted | margine con 2 gen. visti | margine con soli reali |
|---|---|---|---|---|
| immagine intera (256 px) | +0.2 pt | -3.2 pt | +5.5 pt | **+9.0 pt** |
| patch 64 px, griglia fissa | +0.0 pt | -6.5 pt | +13.1 pt | **+19.6 pt** |
| patch 16 px, griglia fissa | -4.6 pt | -7.3 pt | +18.9 pt | **+21.5 pt** |

Su patch 64 px il margine sul pavimento **cresce di oltre sei punti** pur avendo tolto due generatori dal training. La lettura è che la metrica appresa dipende dall'aver visto esempi sintetici **molto meno** di quanto ne dipenda un descrittore spettrale: ciò che le serve è già contenuto nei dati autentici. Un caso isolato lo rende evidente: sul dataset originale il pavimento handcrafted attribuisce StarGAN alla lineage sbagliata nel 69.5% dei casi (30.5% corretti sull'immagine intera), mentre il modello addestrato sui soli reali lo colloca correttamente nel 100%.

Va detto con altrettanta chiarezza che questa configurazione **non elimina** le spiegazioni concorrenti già discusse: il confronto con il pavimento handcrafted e con il baseline zero-shot resta il metro di giudizio, e vale qui esattamente come nella Sezione 7. Il contributo specifico di questa sezione è un altro: mostra che il segnale **vive nei dati autentici**, non nell'esposizione a esempi sintetici.


## 9. Che cosa cattura il modello


### 8.1 Grad-CAM media per classe

La richiesta iniziale era: *si fa una Grad-CAM dove in media, ad esempio su CelebA, abbiamo una mappa media su dove si sta concentrando, e la stessa cosa su StarGAN, e andiamo a vedere se si sta focalizzando sulle stesse regioni*. Due difficoltà tecniche vanno risolte prima.

- **Non c'è un logit di classe.** La rete non classifica, produce un embedding. Serve uno scalare da derivare: si usa lo stesso score della decisione di lineage, cioè la differenza fra la distanza dal centroide dell'altra lineage e quella dal centroide della propria. Il gradiente risponde alla domanda giusta: *quali regioni spingono l'immagine verso la sua lineage?*
- **Il target layer differisce fra CNN e ViT.** Per le ResNet è l'ultimo blocco convoluzionale. Per i ViT l'uscita di un blocco è una sequenza di token: va scartato il token di classe e i token di patch vanno rimessi in griglia. Si usa il penultimo blocco, perché sull'ultimo le mappe ViT tendono a degenerare.

La mappa media per classe, da sola, sarebbe ingannevole: evidenzierebbe *la faccia* in tutte le classi. La figura informativa è lo **scarto dalla media globale**, che mostra cosa è specifico di una classe. E la versione quantitativa è la **matrice di correlazione fra le mappe di scarto**.


![Grad-CAM media per classe (sopra) e scarto dalla media globale (sotto), ResNet18 su immagine intera. Per decidere CelebA il modello guarda la regione centrale del volto (rosso), per decidere FFHQ la periferia - capelli e fondale (blu al centro, rosso ai bordi). StarGAN è di nuovo l'outlier, con attenzione spostata in basso.](report/gradcam_pm128_resnet18/gradcam_mean.png)

*Grad-CAM media per classe (sopra) e scarto dalla media globale (sotto), ResNet18 su immagine intera. Per decidere CelebA il modello guarda la regione centrale del volto (rosso), per decidere FFHQ la periferia - capelli e fondale (blu al centro, rosso ai bordi). StarGAN è di nuovo l'outlier, con attenzione spostata in basso.*


| modello | correlazione media within-lineage | correlazione media cross-lineage |
|---|---|---|
| ResNet18 (immagine intera) | **+0.806** | **-0.881** |
| CLIP ViT-L/14 (immagine intera) | **+0.631** | **-0.759** |


> **Le classi della stessa lineage attivano le stesse regioni, quelle di lineage diverse regioni opposte.** La correlazione fra le mappe di scarto è +0.806 dentro la stessa lineage e -0.881 fra lineage diverse (ResNet18). AttGAN, GDWCT e real_celeba correlano fra loro fra +0.87 e +0.97; real_ffhq, StyleGAN2 e StyleGAN3 fra +0.976 e +0.992. È la conferma spaziale di ciò che la matrice delle distanze mostrava nello spazio di embedding.

**Un limite da dichiarare.** Il fatto che per FFHQ il modello guardi la periferia è coerente con l'osservazione sui fondali: sull'immagine intera la Grad-CAM **conferma** che il modello usa anche il contenuto. Non è una contraddizione, è la stessa cosa che dicono i controlli della Sezione 6 - ed è la ragione per cui è lo sweep a patch, non la Grad-CAM, a isolare il canale di basso livello. La Grad-CAM va presentata come evidenza **qualitativa e di coerenza**. Va inoltre ricordato che la sua risoluzione è quella della griglia del target layer (8x8 per ResNet a 256 pixel, 16x16 per un ViT/14 a 224): una traccia ad alta frequenza è spazialmente diffusa, quindi non c'è da aspettarsi mappe nitide.


### 8.2 Statistiche nel dominio frequenziale e di colore

La richiesta prevedeva anche: *possiamo aggiungere anche qualcosa con il dominio frequenziale, Fourier, istogrammi di colori, e tirare fuori tutte le statistiche*. Per ogni classe vengono calcolati quattro gruppi di descrittori interpretabili: il profilo di potenza azimutale (FFT radiale), la mappa media di log|FFT 2D| e log|DCT 2D|, gli istogrammi di colore RGB e HSV, e le stesse statistiche sul residuo high-pass.


![Mappe spettrali medie per classe: sopra log|FFT 2D|, sotto log|DCT 2D|. Le classi della stessa lineage hanno mappe simili fra loro e diverse dall'altra lineage. Sono i descrittori che costituiscono il pavimento handcrafted della Sezione 6.2.](report/signatures_raw/spectral_maps.png)

*Mappe spettrali medie per classe: sopra log|FFT 2D|, sotto log|DCT 2D|. Le classi della stessa lineage hanno mappe simili fra loro e diverse dall'altra lineage. Sono i descrittori che costituiscono il pavimento handcrafted della Sezione 6.2.*

Il dato più utile di questa analisi è **quale** famiglia porta il segnale, perché dice di che natura è la traccia. Il colore è debole in ogni configurazione (AUC 0.63-0.69 sui mai visti): la differenza fra lineage **non** è cromatica. La DCT 2D è la più forte sulle finestre grandi, il residuo high-pass sulle finestre piccole. Il profilo radiale, che è in pratica una firma di ricampionamento, perde molto quando la pipeline viene normalizzata (da 0.962 a 0.893) - una conferma indipendente che il controllo della Sezione 6.1 ha effettivamente rimosso quel confound.


## 10. Confronto fra architetture

La richiesta prevedeva *una comparison magari con altre architetture, anche riusando la ResNet o altre architetture che esistono*. Il codice espone un registry in cui ogni backbone dichiara la propria risoluzione e normalizzazione, così che lo stesso protocollo possa girare invariato su tutti. I backbone VLM e self-supervised vengono **congelati** (si allena solo la testa), le ResNet **end-to-end**: è la distinzione richiesta.


| backbone | regime | par. addestrati / tot. (M) | epoca del best | AUC mai visti | acc. | SG3 | separazione |
|---|---|---|---|---|---|---|---|
| **CLIP ViT-L/14** | congelato | 0.59 / 303.77 | 3 | 1.0000 | 99.8% | 99.5% | 10x |
| **CLIP ViT-B/16** | congelato | 0.46 / 86.26 | 1 | 1.0000 | 100.0% | 100.0% | 6x |
| **DINOv2 ViT-B/14** | congelato | 0.46 / 86.18 | 3 | 1.0000 | 99.9% | 100.0% | 6x |
| **DINOv2 ViT-L/14** | congelato | 0.59 / 303.82 | 2 | 1.0000 | 99.8% | 99.8% | 8x |
| **ResNet18** | end-to-end | 11.50 / 11.50 | 1 | 1.0000 | 99.9% | 99.8% | 87x |
| **ResNet50** | end-to-end | 24.62 / 24.62 | 1 | 1.0000 | 100.0% | 100.0% | 65x |

**Come si legge, e perché non è la parte interessante.** Sull'immagine intera il task è **saturo**: tutte le architetture arrivano a AUC 1.0 e accuratezza superiore al 99% sui generatori mai visti, e la ResNet18 - il modello più piccolo, 11.5 milioni di parametri - raggiunge AUC 1.0000 ed EER 0.0000 **dalla prima epoca**. Il confronto non discrimina perché il problema, in quel regime, è troppo facile: come mostra la Sezione 6, lo risolve anche una regressione logistica su 576 numeri.

Una differenza c'è, e non riguarda l'accuratezza. Le ResNet addestrate end-to-end producono uno spazio **molto più collassato** (rapporto di separazione 87x e 65x, con distanza intra-classe fino a 0.008) rispetto ai backbone congelati (6x-10x). Potendo modificare tutti i pesi, una rete end-to-end schiaccia le due classi in due punti quasi adimensionali. Non è necessariamente un vantaggio: uno spazio collassato su due classi è anche uno spazio che ha **memorizzato quelle due classi**, mentre la rappresentazione congelata conserva struttura interna e resta più informativa se domani si aggiungesse una terza lineage. Sull'altra dimensione osservabile, la velocità di convergenza, CLIP ViT-B/16 e le due ResNet raggiungono il massimo **alla prima epoca**, DINOv2 ne richiede due o tre.

Questo è a sua volta un risultato, e va detto esplicitamente nell'articolo: **l'uso di un VLM pre-addestrato non è il contributo del lavoro**. Il vantaggio del backbone congelato è pratico e metodologico, non prestazionale: 0.59 milioni di parametri addestrati invece di 11.5 o 24.6 significa un addestramento in pochi minuti, nessun rischio di memorizzare il dataset nei pesi del backbone, e un protocollo in cui la rappresentazione è **fissa e verificabile** - ciò che in un contesto probatorio conta più di una frazione di punto di accuratezza.

Il regime in cui le architetture si differenzierebbero è quello a patch piccole. Là però il confronto con i ViT sarebbe sleale: un backbone che attende 224 pixel richiederebbe di ingrandire una patch da 16 o 32 pixel, distruggendo proprio la traccia di alta frequenza che si vuole misurare. Per questo lo sweep usa ResNet18, che riceve la patch a risoluzione nativa. Un confronto onesto a bassa risoluzione richiede backbone convoluzionali, ed è una estensione naturale.


## 11. Il ponte con l'analisi giuridica

Questa sezione non svolge l'analisi giuridica, che è di competenza altrui. Elenca **cosa l'evidenza tecnica sostiene e cosa no**, nella forma più utilizzabile possibile.


### 10.1 Che cosa è stato dimostrato

- Data un'immagine sintetica, è possibile ricondurla al **dataset su cui il generatore è stato addestrato**, con accuratezza superiore al 99% sull'immagine intera e del 93.9% su un riquadro di 16x16 pixel.
- La capacità **trasferisce a generatori mai visti**: il sistema non ha bisogno di conoscere il modello che ha prodotto l'immagine. È la proprietà decisiva per l'uso pratico, perché in un caso reale il modello sospetto tipicamente non è disponibile.
- **Non serve nemmeno disporre di immagini generate.** Un modello addestrato sui soli dati autentici, che non ha mai visto una singola immagine sintetica, riconduce comunque alla sorgente corretta cinque generatori diversi (Sezione 8). È la condizione in cui si trova chi lamenta l'uso dei propri dati prima di qualunque richiesta di accesso: possiede soltanto il proprio archivio autentico.
- Il risultato **non è** un artefatto di preprocessing: sopravvive alla normalizzazione della catena di ricampionamento su tutte le classi.
- Il risultato **non è** soltanto somiglianza visibile: su finestre piccole la somiglianza semantica cala di oltre venti punti mentre il sistema tiene.
- La decisione ha un **tasso d'errore noto e misurato su dati mai visti**: soglia calibrata al punto di Equal Error Rate sulle sole classi viste, poi falsi positivi e falsi negativi riportati separatamente.


### 10.2 Che cosa non è stato dimostrato

- **Non** che una specifica fotografia fosse nel training set. Quella è la *membership inference*, il livello (c) della Sezione 1.1, e richiede tecniche e garanzie diverse.
- **Non** che il contenuto sia irrilevante: su immagini intere il canale semantico da solo raggiunge gli stessi numeri. Le due spiegazioni sono ridondanti, e solo restringendo la finestra si separano.
- **Non** la generalizzazione a famiglie generative diverse: il dataset contiene soltanto GAN. Il contenzioso attuale sui dati di addestramento riguarda in larga parte i modelli a diffusione.
- **Non** la robustezza a manipolazioni ostili o anche solo ordinarie (ricompressione, riscalatura, filtri dei social network) applicate dopo la generazione. È il primo esperimento da aggiungere.


### 10.3 Perché la forma della decisione conta

Un elemento merita attenzione. Il sistema non produce un'etichetta ma una **distanza**, confrontata con una soglia il cui tasso d'errore è stato misurato su dati che il sistema non aveva visto. È la forma in cui una tecnica si presenta a un vaglio di affidabilità: non *questa immagine viene da FFHQ*, ma *questa immagine cade dal lato FFHQ di una soglia che, su generatori mai visti, sbaglia nell'1.8% dei casi in un verso e nell'1.1% nell'altro*. La seconda formulazione è attaccabile in modo specifico - ed è per questo che vale di più.

Analogamente, la distinzione fra editing-GAN e noise-GAN (Sezione 3.1) non è un dettaglio tecnico ma **due livelli di pretesa diversi**. Quando l'output contiene i pixel di una fotografia reale, la pretesa riguarda quella fotografia e assomiglia a un'opera derivata. Quando l'output nasce da rumore, la pretesa riguarda **il dataset**, ed è la questione nuova. Tenerle separate nell'articolo evita di rivendicare per la seconda la solidità della prima.


## 12. Limiti e lavori futuri

- **Il contenuto è ridotto, non azzerato.** A 16 pixel il canale semantico resta al 73.5%: anche un riquadro piccolo porta tono della pelle e illuminazione. La dissociazione è un divario che si apre, non un interruttore.
- **Due sole lineage.** Il disegno è binario. Con più dataset di addestramento si potrebbe misurare se lo spazio è organizzato per lineage anche oltre due classi, e se la struttura è metrica o solo linearmente separabile.
- **Nessuna noise-GAN su CelebA.** L'asimmetria del dataset è il limite strutturale: lato FFHQ ci sono due noise-GAN, lato CelebA solo editing-GAN. Un generatore da rumore addestrato su CelebA-HQ renderebbe il disegno simmetrico e l'evidenza forte disponibile su entrambi i lati.
- **Nessuna sorgente a diffusione.** È l'estensione di maggiore impatto, perché è dove il problema giuridico è oggi più vivo.
- **Robustezza non testata.** Ricompressione JPEG, riscalatura e filtri applicati dopo la generazione degraderanno il canale di basso livello, che è proprio quello su cui si fonda la tesi. Serve una curva di degrado.
- **Risoluzione.** Tutto il lavoro è a 256 pixel. Il downscale da 1024 distrugge gran parte delle tracce ad alta frequenza: lavorare a risoluzione nativa alzerebbe il tetto di ciò che è estraibile.
- **Il caso a soli dati autentici resta binario.** Con due sole lineage, un modello addestrato solo sui reali risolve un problema a due classi; con più dataset di riferimento il compito sarebbe più severo e più vicino all'uso reale, dove l'archivio autentico è uno fra molti possibili.
- **Un solo seed.** Ogni configurazione è stata addestrata una volta. Per l'articolo servono ripetizioni con seed diversi e intervalli di confidenza, soprattutto sui punti a finestra piccola dove la varianza è maggiore.


## 13. Riproducibilità

Tutto il codice è nel repository **synth-attribution**. I dati e i checkpoint non sono versionati; i report in formato JSON e le figure sì. I moduli aggiunti in questa fase:


| file | funzione |
|---|---|
| src/models/backbones.py | registry dei backbone (ResNet, CLIP, DINOv2) con la propria spec di preprocessing |
| src/models/siamese.py | encoder siamese, freeze del backbone, testa configurabile, caricamento checkpoint |
| src/data/siamese_dataset.py | pairing per lineage, estrazione patch (flat / random / grid), augmentation forense |
| src/eval/eval_lineage.py | geometria, soglia calibrata con FAR/FRR, breakdown per generatore |
| src/eval/class_signatures.py | firme FFT/DCT/colore/residuo e pavimento handcrafted |
| src/eval/content_baseline.py | baseline zero-shot: quanto basta la sola semantica |
| src/eval/eval_gradcam.py | Grad-CAM media per classe e correlazione fra mappe di scarto |
| src/viz/gradcam.py | Grad-CAM per metric learning, su CNN e su ViT |
| src/eval/plot_patch_sweep.py | figura e tabella dello sweep |
| src/eval/pairing_stats.py | composizione delle coppie: quanta parte della supervisione lega dati reali e generati |
| scripts/normalize_pipeline.py | dataset con catena di ricampionamento identica per tutte le classi |
| scripts/show_patch_positions.py | figura di metodo sulle posizioni delle patch |


### Pipeline completa


```bash
# 1. manifest con i tre generatori in holdout
python -m src.data.build_manifest --config configs/dataset_lineage.yaml

# 2. dataset a ricampionamento normalizzato (controllo del confound)
python scripts/normalize_pipeline.py --src data/raw --dst data/raw_pm128 --via 128
python -m src.data.build_manifest --config configs/dataset_lineage_pm128.yaml

# 3. un punto dello sweep: deep + pavimento + zero-shot
python -m src.train.train_siamese --config configs/train_lineage_patch.yaml \
       --patch-size 64 --patch-policy grid --out runs/lineage_grid64_resnet18
python -m src.eval.eval_lineage --checkpoint runs/lineage_grid64_resnet18/best.pt \
       --manifest data/manifest_lineage_pm128.csv --patch-size 64 --patch-policy grid
python -m src.eval.class_signatures --manifest data/manifest_lineage_pm128.csv \
       --patch-size 64 --patch-policy grid
python -m src.eval.content_baseline --manifest data/manifest_lineage_pm128.csv \
       --patch-size 64 --patch-policy grid

# 4. figure, tabelle e questo documento
python -m src.eval.plot_patch_sweep --prefix grid --out report/grid_sweep
python -m src.eval.build_lineage_table --out report/lineage_summary.md
python scripts/build_report.py
```


## Appendice: indice degli esperimenti

Ogni riga corrisponde a una cartella in `report/` con il proprio `report.json`. Il **delta** è il margine del modello addestrato sul pavimento handcrafted calcolato sullo stesso regime.


| run | regime | AUC mai visti | acc. | SG3 | delta |
|---|---|---|---|---|---|
| `raw_clip_vit_l14` | immagine intera, dataset originale | 1.0000 | 100.0% | 100.0% | +0.0001 |
| `pm128_clip_vit_l14` | immagine intera, ricampionamento normalizzato | 1.0000 | 99.8% | 99.5% | +0.0042 |
| `pm128_resnet18` | immagine intera, normalizzato | 1.0000 | 99.9% | 99.8% | +0.0042 |
| `pm128_resnet18_hp` | immagine intera, normalizzato, front-end a residuo | 1.0000 | 100.0% | 100.0% | +0.0042 |
| `pm128_resnet50` | immagine intera, normalizzato | 1.0000 | 100.0% | 100.0% | +0.0042 |
| `pm128_clip_vit_b16` | immagine intera, normalizzato | 1.0000 | 100.0% | 100.0% | +0.0042 |
| `pm128_dinov2_vitb14` | immagine intera, normalizzato | 1.0000 | 99.9% | 100.0% | +0.0042 |
| `pm128_dinov2_vitl14` | immagine intera, normalizzato | 1.0000 | 99.8% | 99.8% | +0.0042 |
| `patch16_resnet18` | patch 16, selezione per varianza | 0.8841 | 77.0% | 86.9% | +0.0951 |
| `patch32_resnet18` | patch 32, selezione per varianza | 0.9819 | 90.2% | 96.5% | +0.1149 |
| `patch64_resnet18` | patch 64, selezione per varianza | 0.9990 | 98.4% | 98.9% | +0.0659 |
| `patch128_resnet18` | patch 128, selezione per varianza | 1.0000 | 99.8% | 99.5% | +0.0057 |
| `grid16_resnet18` | patch 16, griglia fissa | 0.9698 | 89.9% | 93.9% | +0.2091 |
| `grid32_resnet18` | patch 32, griglia fissa | 0.9977 | 97.5% | 99.0% | +0.1240 |
| `grid64_resnet18` | patch 64, griglia fissa | 1.0000 | 99.2% | 99.9% | +0.0436 |
| `grid128_resnet18` | patch 128, griglia fissa | 1.0000 | 100.0% | 99.9% | +0.0196 |
| `realonly256_resnet18` | SOLI REALI in training, immagine intera | 1.0000 | 100.0% | 100.0% | +0.0900 |
| `realonlygrid64_resnet18` | SOLI REALI in training, patch 64 griglia | 0.9991 | 98.4% | 99.9% | +0.1262 |
| `realonlygrid16_resnet18` | SOLI REALI in training, patch 16 griglia | 0.9446 | 84.7% | 89.2% | +0.2161 |

Riassunto tabellare completo, comprensivo dei pavimenti handcrafted per ogni famiglia di descrittori e dei baseline zero-shot: `report/lineage_summary.md`. Tabelle dello sweep: `report/grid_sweep/patch_sweep.md` e `report/patch_sweep/patch_sweep.md`.

