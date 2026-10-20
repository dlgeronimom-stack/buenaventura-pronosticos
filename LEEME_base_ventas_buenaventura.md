# Base histórica de ingresos de Buenaventura

## Uso
`base_ventas_buenaventura_2019Q1_2026Q2.csv` es el único archivo local de datos que `app_ventas.py` necesita para cargar los ingresos históricos. Contiene 30 trimestres continuos entre 2019Q1 y 2026Q2, expresados en US$ millones.

## Trazabilidad
Cada fila tiene período, valor, método de obtención, documento, URL y nota. La columna `metodo_dato` distingue los datos publicados directamente de los comparativos oficiales. El archivo `base_ventas_buenaventura_2019Q1_2026Q2.xlsx` es un libro de auditoría y documentación (no es requerido en tiempo de ejecución):
- `Base_Ventas`: serie trimestral, fuente y notas por fila.
- `Control_Anual`: comparación con cifras oficiales anuales o semestrales; 2020 y 2023 conservan una diferencia de redondeo de US$0.1 millones.
- `Diccionario`: definición de los campos y sus usos.

## Corrección de trazabilidad
La fila 2026Q2 ahora enlaza el comunicado oficial de resultados 2Q26 presentado ante la SEC (Form 6-K), que declara Total Revenues de US$529.0 millones y 6M26 de US$1,153.6 millones:
https://www.sec.gov/Archives/edgar/data/1013131/000110465926088868/tm2621768d1_6k.htm

Como comprobación adicional, los estados financieros intermedios de Buenaventura presentan US$528.992 millones de `Total operating income` para 2Q26 y US$1,153.552 millones para 6M26, valores que redondean a los comunicados:
https://buenaventura.com/wp-content/uploads/2026/04/2Q-Buenaventura-Consolidated-2026.pdf

## Criterios y límites
- No se fuerza la suma trimestral para que coincida con la cifra anual. Las diferencias de US$0.1m en FY2020 y FY2023 están señaladas como redondeo para revisión, no como errores que deban corregirse sin más evidencia.
- El dato 2026 es parcial: hay Q1 y Q2, no un total anual de 2026.
- La serie contiene solo 30 observaciones; es una muestra corta para modelos complejos. Los resultados son experimentales y no son pronósticos oficiales de la compañía.
- Los precios de los metales se consultan por separado en tiempo de ejecución desde el Banco Mundial. Si cambia el XLSX fuente, también podrían variar las proyecciones. Para reproducir exactamente un pronóstico antiguo, hay que archivar la versión del libro del Banco Mundial utilizada.
