@echo off
setlocal
REM ============================================================
REM  Tabella grande (leave-one-generator-out) - per CMD Windows
REM  Uso: dalla radice del repo, nel .venv, digita:  run_matrix.bat
REM ============================================================
set EPOCHS=20

REM pulisco il CSV cumulativo per non mischiare run vecchie
del report\matrix\matrix_rows.csv 2>nul

echo.
echo ===== TRAIN celeba + stargan =====
python -m src.train.train_siamese --config configs\train.yaml --lineage-filter celeba --only-fake stargan --epochs %EPOCHS% --out runs\celeba__stargan
python -m src.eval.eval_matrix --checkpoint runs\celeba__stargan\best.pt --trained-fake stargan --out report\matrix\celeba__stargan

echo.
echo ===== TRAIN celeba + attgan =====
python -m src.train.train_siamese --config configs\train.yaml --lineage-filter celeba --only-fake attgan --epochs %EPOCHS% --out runs\celeba__attgan
python -m src.eval.eval_matrix --checkpoint runs\celeba__attgan\best.pt --trained-fake attgan --out report\matrix\celeba__attgan

echo.
echo ===== TRAIN celeba + gdwct =====
python -m src.train.train_siamese --config configs\train.yaml --lineage-filter celeba --only-fake gdwct --epochs %EPOCHS% --out runs\celeba__gdwct
python -m src.eval.eval_matrix --checkpoint runs\celeba__gdwct\best.pt --trained-fake gdwct --out report\matrix\celeba__gdwct

echo.
echo ===== TRAIN ffhq + stylegan2 =====
python -m src.train.train_siamese --config configs\train.yaml --lineage-filter ffhq --only-fake stylegan2 --epochs %EPOCHS% --out runs\ffhq__stylegan2
python -m src.eval.eval_matrix --checkpoint runs\ffhq__stylegan2\best.pt --trained-fake stylegan2 --out report\matrix\ffhq__stylegan2

echo.
echo ===== TRAIN ffhq + stylegan3 =====
python -m src.train.train_siamese --config configs\train.yaml --lineage-filter ffhq --only-fake stylegan3 --epochs %EPOCHS% --out runs\ffhq__stylegan3
python -m src.eval.eval_matrix --checkpoint runs\ffhq__stylegan3\best.pt --trained-fake stylegan3 --out report\matrix\ffhq__stylegan3

echo.
echo ===== ASSEMBLO LE MATRICI =====
python -m src.eval.build_matrix_table --csv report\matrix\matrix_rows.csv --out report\matrix

echo.
echo Fatto. Tabelle in report\matrix\matrix_tables.md
pause
