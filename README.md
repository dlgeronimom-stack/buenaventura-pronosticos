# Buenaventura · Pronóstico de ingresos trimestrales

Aplicación experimental en Streamlit para explorar escenarios de ingresos trimestrales a partir de 30 observaciones históricas (2019Q1–2026Q2) y precios de metales publicados por el Banco Mundial. **No es una proyección oficial de Buenaventura ni una herramienta de inversión validada.**

## Contenido del proyecto

- `app_ventas.py`: aplicación Streamlit revisada.
- `data/base_ventas_buenaventura_2019Q1_2026Q2.csv`: serie de ingresos con fuentes por fila; la aplicación la lee, no la modifica.
- `assets/mining_header.jpg` y `assets/mining_sidebar.jpg`: recursos decorativos; no son fotografías oficiales de una operación minera.
- `requirements.txt`: dependencias para el entorno web.
- `AUDITORIA_FINAL.md`: resultado y límites de la revisión previa a publicación.

## Ejecutar localmente en Windows

Desde esta carpeta (o desde la terminal de VS Code/Cursor ubicada en ella):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app_ventas.py
```

La aplicación se abrirá en una dirección local de Streamlit, normalmente `http://localhost:8501`.

## Publicar en Streamlit Community Cloud

1. Crea un repositorio de GitHub y sube **todo el contenido de esta carpeta**, incluida la carpeta `data` y `assets`.
2. En Streamlit Community Cloud, crea una nueva aplicación y elige el repositorio, la rama y `app_ventas.py` como archivo principal.
3. Espera a que instale `requirements.txt` y prueba los controles en la URL pública.
4. No subas `.venv`, tokens, contraseñas ni archivos de otros proyectos.

## Requisito de conexión

La aplicación consulta la página de materias primas del Banco Mundial y descarga su archivo mensual XLSX para construir precios trimestrales. Por tanto, la instancia pública requiere conexión de salida disponible al sitio del Banco Mundial. Si la fuente cambia de URL o no está disponible, la carga de precios puede fallar; antes de presentar resultados, verifica que la app muestra correctamente el dato externo y su periodo. La actualización tiene caché temporal de 12 horas.

## Cómo leer el escenario del oro

El porcentaje cambia el supuesto de precio solo para los trimestres cuyo precio completo no está observado. El modelo XGBoost puede devolver el mismo ingreso para varios precios dentro de un rango; en ese caso, la app lo advierte y muestra en la tabla el efecto de cada trimestre frente al escenario de 0 %. No se fuerza una elasticidad artificial.

## Limitaciones que no se deben ocultar

- Solo hay 30 trimestres históricos.
- La variante XGBoost que se muestra como seleccionada se elige por el MAE recursivo de cuatro trimestres. En la validación a un trimestre, la referencia sencilla de último trimestre fue más precisa; la app mantiene visible la advertencia.
- Los pronósticos de cinco a ocho trimestres son extrapolaciones y no tienen evaluación específica al mismo horizonte.
- La banda `±RMSE` es ilustrativa, no es un intervalo de confianza.
- El índice ponderado es una aproximación analítica a partir de volúmenes y precios realizados de FY2025, no una desagregación auditada de ingresos por metal.
- Aunque la estructura de datos se verificó, no se afirma una certificación documental independiente 30/30 de cada cifra trimestral.
