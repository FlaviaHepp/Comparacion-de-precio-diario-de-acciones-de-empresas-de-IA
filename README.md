# 📈Análisis y Simulación de Acciones de Empresas Líderes en IA

Este proyecto realiza un análisis financiero integral de las principales empresas tecnológicas vinculadas al desarrollo de Inteligencia Artificial, combinando análisis técnico, ingeniería de features, simulación de precios y visualización interactiva.

Las empresas analizadas son:

- Apple (AAPL)

- Amazon (AMZN)

- Google / Alphabet (GOOGL)

- Meta (META)

- Microsoft (MSFT)

- NVIDIA (NVDA)

- Tesla (TSLA)

🎯 Objetivo del proyecto

- Analizar la evolución histórica de precios (10 años).
- Calcular indicadores técnicos clave para trading y análisis cuantitativo.
- Simular trayectorias futuras del precio de las acciones mediante Monte Carlo (Geometric Brownian Motion).
- Explorar visualmente los datos con gráficos interactivos en Plotly.
- Generar un pipeline reproducible para análisis financiero en Python.

🧠 Metodología
1. Obtención de datos
- Descarga de datos históricos desde Yahoo Finance usando yfinance.
- Frecuencia diaria desde 2014 hasta la fecha actual.

2. Ingeniería de variables
- Se calculan, entre otros:
- ATR (Average True Range) – volatilidad
- RSI (7 y 14 días) – momentum
- SMA y EMA (50 y 100 días) – tendencias
- Log Returns y Percent Returns
- Rendimientos acumulados

3. Simulación de precios
- Modelo de Movimiento Browniano Geométrico
- Simulación de precios futuros a 1 año
- Múltiples trayectorias por activo
- Visualización interactiva de escenarios posibles

4. Análisis Exploratorio (EDA)
- Gráficos interactivos con menús desplegables
- Análisis de:
  - Precio alto, bajo y cierre
  - Volumen
  - Promedios, máximos y mínimos
  - Indicadores técnicos superpuestos

📊 Visualizaciones
- Gráficos interactivos con Plotly
- Menús dinámicos para seleccionar métricas
- Anotaciones automáticas de máximos, mínimos y promedios
- Estilo plotly_dark orientado a análisis financiero

🛠️ Tecnologías utilizadas
- Python
- pandas / numpy
- yfinance
- ta (Technical Analysis)
- matplotlib
- plotly
- scipy

📂 Estructura del proyecto
├── Análisis de acciones de las 7 empresas de la IA.py
├── AAPL1424.csv
├── AMZN1424.csv
├── GOOGL1424.csv
├── META1424.csv
├── MSFT1424.csv
├── NVDA1424.csv
├── TSLA1424.csv
└── README.md

▶️ Cómo ejecutar el proyecto

Clonar el repositorio

git clone https://github.com/tu_usuario/nombre_del_repo.git


Instalar dependencias

pip install pandas numpy yfinance ta plotly matplotlib scipy


Ejecutar el script

python "Análisis de acciones de las 7 empresas de la IA.py"

⚠️ Disclaimer

Este proyecto tiene fines educativos y analíticos.
No constituye asesoramiento financiero ni recomendaciones de inversión.

👤 Autor

Flavia Hepp
Data Science · Análisis Financiero · Machine Learning
