# Modelo predictivo XGBoost — Buenaventura

## Preparar en Cursor (Windows)
1. Descomprime el ZIP y abre la carpeta en Cursor (File > Open Folder).
2. El Excel ya está dentro de `data/`. Verifica que se llame `DEMO DAY - UNIDAD I .xlsx`.
3. En Terminal > New Terminal ejecuta:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Si falla la activación en PowerShell, ejecuta ` .\.venv\Scripts\python.exe -m pip install -r requirements.txt` y antepone ` .\.venv\Scripts\python.exe` a los comandos siguientes.

## Fase 1: revisar el Excel
```powershell
python src/01_explorar_excel.py
```

## Fase 2: descargar tasa BCRP
```powershell
python src/02_descargar_tasa_bcrp.py
```
Fuente oficial: serie diaria PD12301MD, Tasa de Referencia de la Política Monetaria. API documentada en https://estadisticas.bcrp.gob.pe/estadisticas/series/ayuda/api. El script puede requerir ajuste si cambia el formato CSV de la API; no rellenes datos manualmente ni inventes valores.

## Fase 3: cargar EBITDA histórico verificado
Completa `data/ebitda_trimestral_plantilla.csv` con una fila por trimestre y EBITDA en USD millones. Agrega la URL del reporte oficial, el nombre del documento y la metodología. No copies EBITDA acumulado como si fuera de un trimestre:
- Q1 = acumulado a marzo.
- Q2 = acumulado a junio menos acumulado a marzo.
- Q3 = acumulado a septiembre menos acumulado a junio.
- Q4 = EBITDA anual menos acumulado a septiembre.
Solo aplica diferencias si los reportes usan la misma definición de EBITDA. No mezcles EBITDA ajustado y no ajustado. La base del Excel original sirve como contexto, pero sus ventas de 2025 no sustituyen una serie histórica de EBITDA trimestral.

Fuentes: https://buenaventura.com/inversionista/estados-financieros-2025/ y https://buenaventura.com/en/inversionista/reporte-trimestral-2025/

## Fase 4: unir las fuentes
```powershell
python src/03_preparar_base_modelo.py
```
Revisa `data/base_modelo_trimestral.csv` y confirma que los trimestres coincidan y que no falte EBITDA ni tasa.

## Fase 5: entrenar y evaluar
```powershell
python src/04_entrenar_xgboost.py
```
El script se detiene si no hay al menos 24 filas utilizables. Separa cronológicamente 80% entrenamiento y 20% prueba, y compara XGBoost con una referencia que predice el EBITDA del trimestre previo. Resultados: `outputs/predicciones_prueba.csv` y `outputs/evaluacion_modelo.txt`.

## Criterios metodológicos
- La tasa del BCRP no demuestra causalidad sobre EBITDA.
- Las variables deben estar disponibles en la fecha en que se haría la predicción; evitar fuga de información.
- MAE/RMSE bajos solo describen el error de esa prueba histórica; no garantizan predicciones futuras.
- Si EBITDA no está publicado de manera consistente, detenerse y acordar una definición replicable desde estados financieros antes de entrenar.
