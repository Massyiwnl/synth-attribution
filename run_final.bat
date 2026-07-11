@echo off
setlocal
REM ============================================================
REM  Fase 3 - modello finale (tutto su tutti) + caratterizzazione
REM  Uso: dalla radice del repo, nel .venv:  run_final.bat
REM ============================================================
set EPOCHS=20

echo.
echo ===== TRAIN MODELLO FINALE (tutte le lineage, tutti i generatori) =====
python -m src.train.train_siamese --config configs\train.yaml --lineage-filter all --epochs %EPOCHS% --out runs\final_all

echo.
echo ===== CARATTERIZZAZIONE: PCA + attribuzione closed-set top-1 =====
python -m src.eval.eval_final --checkpoint runs\final_all\best.pt --out report\final

echo.
echo Fatto. Vedi report\final\  (confusion.png, pca_final.png, report.json)
pause
