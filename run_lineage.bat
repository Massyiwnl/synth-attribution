@echo off
setlocal
REM ============================================================
REM  Fase 8 - Attribuzione dei DATI DI ADDESTRAMENTO
REM
REM  Confronto di backbone a parita' di protocollo, su DUE dataset:
REM    raw   = dataset originale (catene di resize diverse fra lineage)
REM    pm128 = dataset normalizzato (stessa catena per tutte le classi)
REM
REM  Il numero che conta e' il DELTA fra il modello deep e il pavimento
REM  handcrafted (src.eval.class_signatures), su entrambi i dataset.
REM
REM  Uso: dalla radice del repo, nel .venv:  run_lineage.bat
REM ============================================================
set EPOCHS=15

echo.
echo ===== 0. MANIFEST (holdout: attgan, gdwct, stylegan3) =====
python -m src.data.build_manifest --config configs\dataset_lineage.yaml

echo.
echo ===== 1. PAVIMENTO HANDCRAFTED (dataset originale) =====
python -m src.eval.class_signatures --manifest data\manifest_lineage.csv --out report\signatures_raw

REM ---------- confronto di backbone sul dataset originale ----------
for %%B in (resnet18 resnet50 clip_vit_b16 clip_vit_l14 dinov2_vitb14) do (
    echo.
    echo ===== TRAIN lineage - backbone %%B =====
    python -m src.train.train_siamese --config configs\train_lineage.yaml ^
        --backbone %%B --epochs %EPOCHS% --out runs\lineage_%%B
    python -m src.eval.eval_lineage --checkpoint runs\lineage_%%B\best.pt ^
        --manifest data\manifest_lineage.csv --out report\lineage_%%B
)

echo.
echo ===== 2. CONTROLLO DEL CONFOUND: dataset a pipeline normalizzata =====
REM se non esiste ancora, costruiscilo (circa 21000 immagini)
if not exist data\raw_pm128 (
    python scripts\normalize_pipeline.py --src data\raw --dst data\raw_pm128 --via 128
)
python -m src.data.build_manifest --config configs\dataset_lineage_pm128.yaml
python -m src.eval.class_signatures --manifest data\manifest_lineage_pm128.csv --out report\signatures_pm128

for %%B in (resnet18 clip_vit_l14) do (
    echo.
    echo ===== TRAIN lineage PM128 - backbone %%B =====
    python -m src.train.train_siamese --config configs\train_lineage.yaml ^
        --backbone %%B --manifest data\manifest_lineage_pm128.csv ^
        --epochs %EPOCHS% --out runs\lineage_pm128_%%B
    python -m src.eval.eval_lineage --checkpoint runs\lineage_pm128_%%B\best.pt ^
        --manifest data\manifest_lineage_pm128.csv --out report\lineage_pm128_%%B
)

echo.
echo ===== 3. TABELLA DI CONFRONTO =====
python -m src.eval.build_lineage_table --out report\lineage_summary.md

echo.
echo Fatto. Vedi report\lineage_summary.md
pause
