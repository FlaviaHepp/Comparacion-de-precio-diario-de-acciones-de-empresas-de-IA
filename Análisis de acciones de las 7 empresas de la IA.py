
#Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import yfinance as yf
import ta
import requests
from datetime import datetime
from scipy.stats import norm


##Definir una función para realizar simulación y trazado para cada empresa
def simulate_and_plot(data, company_name):
    #Preprocesar datos para la empresa
    data_ps = data.copy()
    data_ps.set_index('Date', inplace=True)
    data_prs = data_ps['Close']

    retornos_de_registro = np.log(1 + data_prs.pct_change())

    u = retornos_de_registro.mean()
    var = retornos_de_registro.var()
    stdev = retornos_de_registro.std()

    deriva = u - (0.5 * var)
    np.array(deriva)

    intervalos_t = 365
    iteraciones = 6

    devoluciones_diarias = np.exp(deriva + stdev * norm.ppf(np.random.rand(intervalos_t, iteraciones)))
    S0 = data_prs.iloc[-1]

    lista_de_precios = np.zeros_like(devoluciones_diarias)
    lista_de_precios[0] = S0

    for t in range(1, intervalos_t):
        lista_de_precios[t] = lista_de_precios[t-1] * devoluciones_diarias[t]

    datos = pd.date_range(start='2024-03-19', periods=intervalos_t)

    #Simulación de parcela para la empresa
    fig = go.Figure()
    for i in range(iteraciones):
        fig.add_trace(go.Scatter(x=datos, y=lista_de_precios[:, i], mode='lines', name=f'Simulation {i+1}'))

    #Agregar información de diseño
    fig.update_layout(title=f'{company_name} Simulación del precio de las acciones\n',
                       xaxis_title='Fecha\n',
                       yaxis_title='Precio (dólares americanos)\n',
                       hovermode='x',
                       template='plotly_dark')

    #mostrar la trama
    fig.show()


AAPL = pd.read_csv("AAPL1424.csv")
simulate_and_plot(AAPL, "AAPL")
print("\nAPPLE\n", AAPL)

AMZN = pd.read_csv("AMZN1424.csv")
simulate_and_plot(AMZN, "AMZN")
print("\nAMAZON\n", AMZN)

GOOGL = pd.read_csv("GOOGL1424.csv")
simulate_and_plot(GOOGL, "GOOGL")
print("\nGOOGLE\n", GOOGL)

META = pd.read_csv("META1424.csv")
simulate_and_plot(META, "META")
print("\nMETA\n", META)

MSFT = pd.read_csv("MSFT1424.csv")
simulate_and_plot(MSFT, "MSFT")
print("\nMICROSOFT\n", MSFT)

NVDA = pd.read_csv('NVDA1424.csv')
simulate_and_plot(NVDA, "NVDA")
print("\nNVIDIA\n", NVDA)

TSLA = pd.read_csv("TSLA1424.csv")
simulate_and_plot(TSLA, "TSLA")
print("\nTSLA\n", TSLA)

#Obtener la fecha de hoy
current_date = datetime.today().strftime('%Y-%m-%d')

##CSV's e ingeniería de funciones
#AAPL
AAPL= yf.download('AAPL', start='2014-01-02', end=current_date)
AAPL.reset_index(inplace=True)

AAPL['ATR (7 Days)'] = ta.volatility.average_true_range(high=AAPL['High'], low=AAPL['Low'], close=AAPL['Close'], window=7, fillna=True)
AAPL['RSI (7 Days)'] = ta.momentum.RSIIndicator(AAPL['Close'], window=7, fillna = True).rsi()
AAPL['RSI (14 Days)'] = ta.momentum.RSIIndicator(AAPL['Close'], window=14, fillna= True).rsi()
AAPL['EMA (50 Days)'] = ta.trend.ema_indicator(AAPL['Close'], window=50, fillna=True)
AAPL['EMA (100 Days)'] = ta.trend.ema_indicator(AAPL['Close'], window=100, fillna=True)
AAPL['SMA (50 Days)'] = ta.trend.sma_indicator(AAPL['Close'], window=50, fillna=True)
AAPL['SMA (100 Days)'] = ta.trend.sma_indicator(AAPL['Close'], window=100, fillna=True)

AAPL['Log Return'] = np.log(AAPL['Close'] / AAPL['Close'].shift(1)) #Calcular los retornos de registros diarios
AAPL['Percent Return'] = AAPL['Close'].pct_change() #Calcular el porcentaje de rendimiento diario
AAPL['Cumulative Log Return'] = AAPL['Log Return'].cumsum().ffill() #Calcular los rendimientos de registros acumulados
AAPL['Cumulative Percent Return'] = AAPL['Percent Return'].cumsum().ffill() #Calcular los rendimientos porcentuales acumulados

print(AAPL.isnull().sum())


#AMAZON
AMZN= yf.download('AMZN', start='2014-01-02', end=current_date)
AMZN.reset_index(inplace=True)

#Aplicar algunos análisis técnicos 
AMZN['ATR (7 Days)'] = ta.volatility.average_true_range(high=AMZN['High'], low=AMZN['Low'], close=AMZN['Close'], window=7, fillna=True)
AMZN['RSI (7 Days)'] = ta.momentum.RSIIndicator(AMZN['Close'], window=7, fillna = True).rsi()
AMZN['RSI (14 Days)'] = ta.momentum.RSIIndicator(AMZN['Close'], window=14, fillna= True).rsi()
AMZN['EMA (50 Days)'] = ta.trend.ema_indicator(AMZN['Close'], window=50, fillna=True)
AMZN['EMA (100 Days)'] = ta.trend.ema_indicator(AMZN['Close'], window=100, fillna=True)
AMZN['SMA (50 Days)'] = ta.trend.sma_indicator(AMZN['Close'], window=50, fillna=True)
AMZN['SMA (100 Days)'] = ta.trend.sma_indicator(AMZN['Close'], window=100, fillna=True)

AMZN['Log Return'] = np.log(AMZN['Close'] / AMZN['Close'].shift(1)) 
AMZN['Percent Return'] = AMZN['Close'].pct_change() 
AMZN['Cumulative Log Return'] = AMZN['Log Return'].cumsum().ffill() 
AMZN['Cumulative Percent Return'] = AMZN['Percent Return'].cumsum().ffill() 

print(AMZN.isnull().sum())


#GOOGL
GOOGL= yf.download('GOOGL', start='2014-01-02', end=current_date)
GOOGL.reset_index(inplace=True)

GOOGL['ATR (7 Days)'] = ta.volatility.average_true_range(high=GOOGL['High'], low=GOOGL['Low'], close=GOOGL['Close'], window=7, fillna=True)
GOOGL['RSI (7 Days)'] = ta.momentum.RSIIndicator(GOOGL['Close'], window=7, fillna = True).rsi()
GOOGL['RSI (14 Days)'] = ta.momentum.RSIIndicator(GOOGL['Close'], window=14, fillna= True).rsi()
GOOGL['EMA (50 Days)'] = ta.trend.ema_indicator(GOOGL['Close'], window=50, fillna=True)
GOOGL['EMA (100 Days)'] = ta.trend.ema_indicator(GOOGL['Close'], window=100, fillna=True)
GOOGL['SMA (50 Days)'] = ta.trend.sma_indicator(GOOGL['Close'], window=50, fillna=True)
GOOGL['SMA (100 Days)'] = ta.trend.sma_indicator(GOOGL['Close'], window=100, fillna=True)

GOOGL['Log Return'] = np.log(GOOGL['Close'] / GOOGL['Close'].shift(1)) 
GOOGL['Percent Return'] = GOOGL['Close'].pct_change() 
GOOGL['Cumulative Log Return'] = GOOGL['Log Return'].cumsum().ffill() 
GOOGL['Cumulative Percent Return'] = GOOGL['Percent Return'].cumsum().ffill() 

print(GOOGL.isnull().sum())


#META
META= yf.download('META', start='2014-01-02', end=current_date)
META.reset_index(inplace=True)

META['ATR (7 Days)'] = ta.volatility.average_true_range(high=META['High'], low=META['Low'], close=META['Close'], window=7, fillna=True)
META['RSI (7 Days)'] = ta.momentum.RSIIndicator(META['Close'], window=7, fillna = True).rsi()
META['RSI (14 Days)'] = ta.momentum.RSIIndicator(META['Close'], window=14, fillna= True).rsi()
META['EMA (50 Days)'] = ta.trend.ema_indicator(META['Close'], window=50, fillna=True)
META['EMA (100 Days)'] = ta.trend.ema_indicator(META['Close'], window=100, fillna=True)
META['SMA (50 Days)'] = ta.trend.sma_indicator(META['Close'], window=50, fillna=True)
META['SMA (100 Days)'] = ta.trend.sma_indicator(META['Close'], window=100, fillna=True)

META['Log Return'] = np.log(META['Close'] / META['Close'].shift(1)) 
META['Percent Return'] = META['Close'].pct_change() 
META['Cumulative Log Return'] = META['Log Return'].cumsum().ffill() 
META['Cumulative Percent Return'] = META['Percent Return'].cumsum().ffill() 

print(META.isnull().sum())


#MSFT
MSFT = yf.download('MSFT', start='2014-01-02', end=current_date)
MSFT.reset_index(inplace=True)

MSFT['ATR (7 Days)'] = ta.volatility.average_true_range(high=MSFT['High'], low=MSFT['Low'], close=MSFT['Close'], window=7, fillna=True)
MSFT['RSI (7 Days)'] = ta.momentum.RSIIndicator(MSFT['Close'], window=7, fillna = True).rsi()
MSFT['RSI (14 Days)'] = ta.momentum.RSIIndicator(MSFT['Close'], window=14, fillna= True).rsi()
MSFT['EMA (50 Days)'] = ta.trend.ema_indicator(MSFT['Close'], window=50, fillna=True)
MSFT['EMA (100 Days)'] = ta.trend.ema_indicator(MSFT['Close'], window=100, fillna=True)
MSFT['SMA (50 Days)'] = ta.trend.sma_indicator(MSFT['Close'], window=50, fillna=True)
MSFT['SMA (100 Days)'] = ta.trend.sma_indicator(MSFT['Close'], window=100, fillna=True)

MSFT['Log Return'] = np.log(MSFT['Close'] / MSFT['Close'].shift(1)) 
MSFT['Percent Return'] = MSFT['Close'].pct_change() 
MSFT['Cumulative Log Return'] = MSFT['Log Return'].cumsum().ffill() 
MSFT['Cumulative Percent Return'] = MSFT['Percent Return'].cumsum().ffill() 

print(MSFT.isnull().sum())


#NVIDIA
NVDA = yf.download('NVDA', start='2014-01-02', end=current_date)
NVDA.reset_index(inplace=True)

NVDA['ATR (7 Days)'] = ta.volatility.average_true_range(high=NVDA['High'], low=NVDA['Low'], close=NVDA['Close'], window=7, fillna=True)
NVDA['RSI (7 Days)'] = ta.momentum.RSIIndicator(NVDA['Close'], window=7, fillna = True).rsi()
NVDA['RSI (14 Days)'] = ta.momentum.RSIIndicator(NVDA['Close'], window=14, fillna= True).rsi()
NVDA['EMA (50 Days)'] = ta.trend.ema_indicator(NVDA['Close'], window=50, fillna=True)
NVDA['EMA (100 Days)'] = ta.trend.ema_indicator(NVDA['Close'], window=100, fillna=True)
NVDA['SMA (50 Days)'] = ta.trend.sma_indicator(NVDA['Close'], window=50, fillna=True)
NVDA['SMA (100 Days)'] = ta.trend.sma_indicator(NVDA['Close'], window=100, fillna=True)

NVDA['Log Return'] = np.log(NVDA['Close'] / NVDA['Close'].shift(1)) 
NVDA['Percent Return'] = NVDA['Close'].pct_change() 
NVDA['Cumulative Log Return'] = NVDA['Log Return'].cumsum().ffill() 
NVDA['Cumulative Percent Return'] = NVDA['Percent Return'].cumsum().ffill() 

print(NVDA.isnull().sum())


#TSLA
TSLA = yf.download('TSLA', start='2014-01-02', end=current_date)
TSLA.reset_index(inplace=True)

TSLA['ATR (7 Days)'] = ta.volatility.average_true_range(high=TSLA['High'], low=TSLA['Low'], close=TSLA['Close'], window=7, fillna=True)
TSLA['RSI (7 Days)'] = ta.momentum.RSIIndicator(TSLA['Close'], window=7, fillna = True).rsi()
TSLA['RSI (14 Days)'] = ta.momentum.RSIIndicator(TSLA['Close'], window=14, fillna= True).rsi()
TSLA['EMA (50 Days)'] = ta.trend.ema_indicator(TSLA['Close'], window=50, fillna=True)
TSLA['EMA (100 Days)'] = ta.trend.ema_indicator(TSLA['Close'], window=100, fillna=True)
TSLA['SMA (50 Days)'] = ta.trend.sma_indicator(TSLA['Close'], window=50, fillna=True)
TSLA['SMA (100 Days)'] = ta.trend.sma_indicator(TSLA['Close'], window=100, fillna=True)

TSLA['Log Return'] = np.log(TSLA['Close'] / TSLA['Close'].shift(1)) 
TSLA['Percent Return'] = TSLA['Close'].pct_change() 
TSLA['Cumulative Log Return'] = TSLA['Log Return'].cumsum().ffill() 
TSLA['Cumulative Percent Return'] = TSLA['Percent Return'].cumsum().ffill() 

print(TSLA.isnull().sum())


##EDA con menús desplegables
#NVDA
fig = go.Figure()

fig.add_trace(go.Scatter(x=list(NVDA.Date),y=list(NVDA.High),name="Alto\n",
    line=dict(color="#33CFA5"),mode="lines",showlegend=True,visible=True))

fig.add_trace(go.Scatter(x=list(NVDA.Date),y=[NVDA.High.mean()] * len(NVDA.Date), name="Alto promedio\n",
               visible=False, line=dict(color="#33CFA5", dash="dash")))

fig.add_trace(go.Scatter(x=list(NVDA.Date),y=list(NVDA.Low),
               name="Bajo\n", line=dict(color="#F06A6A")))

fig.add_trace(go.Scatter(x=list(NVDA.Date), y=[NVDA.Low.mean()] * len(NVDA.Date),name="Bajo promedio\n",
                         visible=False,line=dict(color="#F06A6A", dash="dash")))

fig.add_trace(go.Scatter(x=list(NVDA.Date),y=list(NVDA.Close),visible=False,
               name="Cerca\n", line=dict(color="#fa8825")))

fig.add_trace(go.Scatter(x=list(NVDA.Date),y=[NVDA.Close.mean()] * len(NVDA.Date),name="Cerrar Promedio\n",
                         visible=False,line=dict(color="#fa8825", dash="dash")))

fig.add_trace(go.Scatter(x=list(NVDA.Date),y=list(NVDA.Volume),visible=False,
               name="Volumen\n", line=dict(color="#002f43")))

fig.add_trace(go.Scatter(x=list(NVDA.Date),
               y=[NVDA.Volume.mean()] * len(NVDA.Date),name="Promedio de volumen\n",visible=False,
               line=dict(color="#002f43", dash="dash")))


#Agregar anotaciones y botones
high_annotations = [dict(x="2016-03-01",
                         y=NVDA.High.mean(),
                         xref="x", yref="y",
                         text="High Average:<br> %.3f" % NVDA.High.mean(),
                         ax=0, ay=-40),
                    dict(x=NVDA.Date[NVDA.High.idxmax()],
                         y=NVDA.High.max(),
                         xref="x", yref="y",
                         text="High Max:<br> %.3f" % NVDA.High.max(),
                         ax=-40, ay=-40),
                    dict(x=NVDA.Date[NVDA.High.idxmin()],
                         y=NVDA.High.min(),
                         xref="x", yref="y",
                         text="High Min:<br> %.3f" % NVDA.High.min(),
                         ax=-40, ay=-40) ]
low_annotations = [dict(x="2015-05-01",
                        y=NVDA.Low.mean(),
                        xref="x", yref="y",
                        text="Low Average:<br> %.3f" % NVDA.Low.mean(),
                        ax=0, ay=40),
                   dict(x=NVDA.Date[NVDA.Low.idxmin()],
                        y=NVDA.Low.min(),
                        xref="x", yref="y",
                        text="Low Min:<br> %.3f" % NVDA.Low.min(),
                        ax=0, ay=40),
                   dict(x=NVDA.Date[NVDA.Low.idxmax()],
                         y=NVDA.Low.max(),
                         xref="x", yref="y",
                         text="Low Max:<br> %.3f" % NVDA.Low.max(),
                         ax=-40, ay=-40)]
close_annotations = [dict(x="2015-05-01",
                        y=NVDA.Close.mean(),
                        xref="x", yref="y",
                        text="Close Average:<br> %.3f" % NVDA.Close.mean(),
                        ax=0, ay=40),
                   dict(x=NVDA.Date[NVDA.Close.idxmin()],
                        y=NVDA.Low.min(),
                        xref="x", yref="y",
                        text="Close Min:<br> %.3f" % NVDA.Close.min(),
                        ax=0, ay=40),
                   dict(x=NVDA.Date[NVDA.Close.idxmax()],
                         y=NVDA.Close.max(),
                         xref="x", yref="y",
                         text="Close Max:<br> %.3f" % NVDA.Close.max(),
                         ax=-40, ay=-40)]
volume_annotations = [dict(x="2015-05-01",
                        y=NVDA.Volume.mean(),
                        xref="x", yref="y",
                        text="Volume Average:<br> %.3f" % NVDA.Volume.mean(),
                        ax=0, ay=40),
                   dict(x=NVDA.Date[NVDA.Volume.idxmin()],
                        y=NVDA.Volume.min(),
                        xref="x", yref="y",
                        text="Volume Min:<br> %.3f" % NVDA.Volume.min(),
                        ax=0, ay=40),
                   dict(x=NVDA.Date[NVDA.Volume.idxmax()],
                         y=NVDA.Volume.max(),
                         xref="x", yref="y",
                         text="Volume Max:<br> %.3f" % NVDA.Volume.max(),
                         ax=-40, ay=-40)]

fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label="Descripción general",
                     method="update",
                     args=[{"visible": [True, False, True, False, False, False, False, False]},
                           {"title": "Descripción general de NVIDIA\n",
                            "annotations": []}]),
                dict(label="Alto",
                     method="update",
                     args=[{"visible": [True, True, False, False, False, False, False, False]},
                           {"title": "NVIDIA Alto (10 años)\n",
                            "annotations": high_annotations}]),
                dict(label="Bajo",
                     method="update",
                     args=[{"visible": [False, False, True, True, False, False, False, False]},
                           {"title": "NVIDIA bajo (10 años)\n",
                            "annotations": low_annotations}]),
                dict(label="Cerca",
                     method="update",
                     args=[{"visible": [False, False, False, False, True, True, False, False]},
                           {"title": "Cierre de NVIDIA (10 años)\n",
                            "annotations": close_annotations}]),
                dict(label="Volumen",
                     method="update",
                     args=[{"visible": [False, False, False, False, False, False, True, True]},
                           {"title": "Volumen de NVIDIA (10 años)\n",
                            "annotations": volume_annotations}]),
            ]),
        )
    ])

#Establecer título
fig.update_layout(title_text="Indicadores técnicos del precio de las acciones de NVIDIA Corp (10 años)\n", template='plotly_dark')


#AAPL
fig.show()

fig1 = go.Figure()

fig1.add_trace(go.Scatter(x=list(AAPL.Date),y=list(AAPL.Close),visible=True,
               name="Cerca\n", line=dict(color="#fa8825"), showlegend=True))

fig1.add_trace(go.Scatter(x=list(AAPL.Date),y=list(AAPL['RSI (7 Days)']),visible=False,
               name="Índice de fuerza relativa (7 días)\n",showlegend=True, line=dict(color="#6495ed")))

fig1.add_trace(go.Scatter(x=list(AAPL.Date),y=list(AAPL['ATR (7 Days)']),visible=False,
               name="Rango verdadero promedio (7 días)\n",showlegend=True,  line=dict(color="#008080")))

fig1.add_trace(go.Scatter(x=list(AAPL.Date),y=list(AAPL['SMA (50 Days)']),visible=False,
                name="Media móvil simple (50 días)\n",showlegend=True,line=dict(color="#4b1454")))

fig1.add_trace(go.Scatter(x=list(AAPL.Date),y=list(AAPL['SMA (100 Days)']),visible=False,
               name="Media móvil simple (100 días)\n", showlegend=True, line=dict(color="#a094a4")))

fig1.add_trace(go.Scatter(x=list(AAPL.Date),y=list(AAPL['EMA (50 Days)']),visible=False,
               name="Media móvil exponencial (50 días)\n",showlegend=True, line=dict(color="#04233f")))

fig1.add_trace(go.Scatter(x=list(AAPL.Date),y=list(AAPL['EMA (100 Days)']),visible=False,
               name="Media móvil exponencial (100 días)\n",showlegend=True, line=dict(color="#ecb08a")))

button_layer_1_height = 1.08

fig1.update_layout(
    updatemenus=[
        dict(
            buttons=list([
                dict(
                    args=[{"visible": [True, False, False, False, False, False, False]},
                          {"title": "Media móvil simple (SMA)\n"}],
                    label="SMA (solo cierre)\n",
                    method="update"
                ),
                dict(
                    args=[{"visible": [True, False, False, True, False, False, False]},
                          {"title": "Media móvil simple (SMA)\n"}],
                    label="AME (50 días)\n",
                    method="update"
                ),
                dict(
                    args=[{"visible": [True, False, False, False, True, False, False]},
                          {"title": "Índice de fuerza relativa (RSI)\n"}],
                    label="SMA (100 días)\n",
                    method="update"
                ),
                dict(
                    args=[{"visible": [True, False, False, True, True, False, False]},
                          {"title": "Media móvil exponencial (EMA)\n"}],
                    label="AME (Ambos)",
                    method="update"
                ),
            ]),
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=button_layer_1_height,
            yanchor="top"
        ),
        dict(
            buttons=list([
                dict(
                    args=[{"visible": [True, True, False, False, False, False, False]},
                          {"title": "Índice de fuerza relativa (RSI)\n"}],
                    label="RSI (7 días)\n",
                    method="update"
                ),
                dict(
                    args=[{"visible": [True, False, True, False, False, False, False]},
                          {"title": "Índice de fuerza relativa (RSI)\n"}],
                    label="ATR (7 días)\n",
                    method="update"
                ),
                dict(
                    args=[{"visible": [True, True, True, False, False, False, False]},
                          {"title": "Índice de fuerza relativa (RSI)\n"}],
                    label="RSI y ATR (ambos)",
                    method="update"
                ),
            ]),
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.37,
            xanchor="left",
            y=button_layer_1_height,
            yanchor="top"
        ),
        dict(
            buttons=list([
                dict(
                    args=[{"visible": [True, False, False, False, False, False, True]},
                          {"title": "Media móvil exponencial (EMA)\n"}],
                    label="EMA (100 días)\n",
                    method="update"
                ),
                dict(
                    args=[{"visible": [True, False, False, False, False, True, False]},
                          {"title": "Media móvil exponencial (EMA))\n"}],
                    label="EMA (50 días)\n",
                    method="update"
                ),
                dict(
                    args=[{"visible": [True, False, False, False, False, True, True]},
                          {"title": "Media móvil exponencial (EMA)\n"}],
                    label="EMA (Ambos)\n",
                    method="update"
                ),
            ]),
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.64,
            xanchor="left",
            y=button_layer_1_height,
            yanchor="top",
        ),
    ]
)

#Establecer título
fig.update_layout(title_text="Indicadores técnicos del precio de las acciones de AAPL Corp (10 años)\n", template='plotly_dark')
fig.show()


#MSFT
fig = go.Figure()

fig.add_trace(go.Scatter(x=list(MSFT.Date),y=list(MSFT.High),name="Alto\n",
    line=dict(color="#33CFA5"),mode="lines",showlegend=True,visible=True))

fig.add_trace(go.Scatter(x=list(MSFT.Date),y=[MSFT.High.mean()] * len(MSFT.Date), name="Alto promedio\n",
               visible=False, line=dict(color="#33CFA5", dash="dash")))

fig.add_trace(go.Scatter(x=list(MSFT.Date),y=list(MSFT.Low),
               name="Bajo\n", line=dict(color="#F06A6A")))

fig.add_trace(go.Scatter(x=list(MSFT.Date), y=[MSFT.Low.mean()] * len(MSFT.Date),name="Bajo promedio\n",
                         visible=False,line=dict(color="#F06A6A", dash="dash")))

fig.add_trace(go.Scatter(x=list(MSFT.Date),y=list(MSFT.Close),visible=False,
               name="Cerca\n", line=dict(color="#fa8825")))

fig.add_trace(go.Scatter(x=list(MSFT.Date),y=[MSFT.Close.mean()] * len(MSFT.Date),name="Cerrar Promedio\n",
                         visible=False,line=dict(color="#fa8825", dash="dash")))

fig.add_trace(go.Scatter(x=list(MSFT.Date),y=list(MSFT.Volume),visible=False,
               name="Volumen\n", line=dict(color="#002f43")))

fig.add_trace(go.Scatter(x=list(MSFT.Date),
               y=[MSFT.Volume.mean()] * len(MSFT.Date),name="Promedio de volumen\n",visible=False,
               line=dict(color="#002f43", dash="dash")))


#Agregar anotaciones y botones
high_annotations = [dict(x="2016-03-01",
                         y=MSFT.High.mean(),
                         xref="x", yref="y",
                         text="High Average:<br> %.3f" % MSFT.High.mean(),
                         ax=0, ay=-40),
                    dict(x=MSFT.Date[MSFT.High.idxmax()],
                         y=MSFT.High.max(),
                         xref="x", yref="y",
                         text="High Max:<br> %.3f" % MSFT.High.max(),
                         ax=-40, ay=-40),
                    dict(x=MSFT.Date[MSFT.High.idxmin()],
                         y=MSFT.High.min(),
                         xref="x", yref="y",
                         text="High Min:<br> %.3f" % MSFT.High.min(),
                         ax=-40, ay=-40) ]
low_annotations = [dict(x="2015-05-01",
                        y=MSFT.Low.mean(),
                        xref="x", yref="y",
                        text="Low Average:<br> %.3f" % MSFT.Low.mean(),
                        ax=0, ay=40),
                   dict(x=MSFT.Date[MSFT.Low.idxmin()],
                        y=MSFT.Low.min(),
                        xref="x", yref="y",
                        text="Low Min:<br> %.3f" % MSFT.Low.min(),
                        ax=0, ay=40),
                   dict(x=MSFT.Date[MSFT.Low.idxmax()],
                         y=MSFT.Low.max(),
                         xref="x", yref="y",
                         text="Low Max:<br> %.3f" % MSFT.Low.max(),
                         ax=-40, ay=-40)]
close_annotations = [dict(x="2015-05-01",
                        y=MSFT.Close.mean(),
                        xref="x", yref="y",
                        text="Close Average:<br> %.3f" % MSFT.Close.mean(),
                        ax=0, ay=40),
                   dict(x=MSFT.Date[MSFT.Close.idxmin()],
                        y=MSFT.Low.min(),
                        xref="x", yref="y",
                        text="Close Min:<br> %.3f" % MSFT.Close.min(),
                        ax=0, ay=40),
                   dict(x=MSFT.Date[MSFT.Close.idxmax()],
                         y=MSFT.Close.max(),
                         xref="x", yref="y",
                         text="Close Max:<br> %.3f" % MSFT.Close.max(),
                         ax=-40, ay=-40)]
volume_annotations = [dict(x="2015-05-01",
                        y=MSFT.Volume.mean(),
                        xref="x", yref="y",
                        text="Volume Average:<br> %.3f" % MSFT.Volume.mean(),
                        ax=0, ay=40),
                   dict(x=MSFT.Date[MSFT.Volume.idxmin()],
                        y=MSFT.Volume.min(),
                        xref="x", yref="y",
                        text="Volume Min:<br> %.3f" % MSFT.Volume.min(),
                        ax=0, ay=40),
                   dict(x=MSFT.Date[MSFT.Volume.idxmax()],
                         y=MSFT.Volume.max(),
                         xref="x", yref="y",
                         text="Volume Max:<br> %.3f" % MSFT.Volume.max(),
                         ax=-40, ay=-40)]

fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label="Descripción general",
                     method="update",
                     args=[{"visible": [True, False, True, False, False, False, False, False]},
                           {"title": "Descripción general de MSFT\n",
                            "annotations": []}]),
                dict(label="Alto",
                     method="update",
                     args=[{"visible": [True, True, False, False, False, False, False, False]},
                           {"title": "MSFT Alto (10 años)\n",
                            "annotations": high_annotations}]),
                dict(label="Bajo",
                     method="update",
                     args=[{"visible": [False, False, True, True, False, False, False, False]},
                           {"title": "MSFT bajo (10 años)\n",
                            "annotations": low_annotations}]),
                dict(label="Cerca",
                     method="update",
                     args=[{"visible": [False, False, False, False, True, True, False, False]},
                           {"title": "Cierre de MSFT (10 años)\n",
                            "annotations": close_annotations}]),
                dict(label="Volumen",
                     method="update",
                     args=[{"visible": [False, False, False, False, False, False, True, True]},
                           {"title": "Volumen de MSFT (10 años)\n",
                            "annotations": volume_annotations}]),
            ]),
        )
    ])

#Establecer título
fig.update_layout(title_text="Indicadores técnicos del precio de las acciones de MSFT Corp (10 años)\n", template='plotly_dark')
fig.show()


#TSLA
fig = go.Figure()

fig.add_trace(go.Scatter(x=list(TSLA.Date),y=list(TSLA.High),name="Alto\n",
    line=dict(color="#33CFA5"),mode="lines",showlegend=True,visible=True))

fig.add_trace(go.Scatter(x=list(TSLA.Date),y=[TSLA.High.mean()] * len(TSLA.Date), name="Alto promedio\n",
               visible=False, line=dict(color="#33CFA5", dash="dash")))

fig.add_trace(go.Scatter(x=list(TSLA.Date),y=list(TSLA.Low),
               name="Bajo\n", line=dict(color="#F06A6A")))

fig.add_trace(go.Scatter(x=list(TSLA.Date), y=[TSLA.Low.mean()] * len(TSLA.Date),name="Bajo promedio\n",
                         visible=False,line=dict(color="#F06A6A", dash="dash")))

fig.add_trace(go.Scatter(x=list(TSLA.Date),y=list(TSLA.Close),visible=False,
               name="Cerca\n", line=dict(color="#fa8825")))

fig.add_trace(go.Scatter(x=list(TSLA.Date),y=[TSLA.Close.mean()] * len(TSLA.Date),name="Cerrar Promedio\n",
                         visible=False,line=dict(color="#fa8825", dash="dash")))

fig.add_trace(go.Scatter(x=list(TSLA.Date),y=list(TSLA.Volume),visible=False,
               name="Volumen\n", line=dict(color="#002f43")))

fig.add_trace(go.Scatter(x=list(TSLA.Date),
               y=[TSLA.Volume.mean()] * len(TSLA.Date),name="Promedio de volumen\n",visible=False,
               line=dict(color="#002f43", dash="dash")))


#Agregar anotaciones y botones
high_annotations = [dict(x="2016-03-01",
                         y=TSLA.High.mean(),
                         xref="x", yref="y",
                         text="High Average:<br> %.3f" % TSLA.High.mean(),
                         ax=0, ay=-40),
                    dict(x=TSLA.Date[TSLA.High.idxmax()],
                         y=TSLA.High.max(),
                         xref="x", yref="y",
                         text="High Max:<br> %.3f" % TSLA.High.max(),
                         ax=-40, ay=-40),
                    dict(x=TSLA.Date[TSLA.High.idxmin()],
                         y=TSLA.High.min(),
                         xref="x", yref="y",
                         text="High Min:<br> %.3f" % TSLA.High.min(),
                         ax=-40, ay=-40) ]
low_annotations = [dict(x="2015-05-01",
                        y=TSLA.Low.mean(),
                        xref="x", yref="y",
                        text="Low Average:<br> %.3f" % TSLA.Low.mean(),
                        ax=0, ay=40),
                   dict(x=TSLA.Date[TSLA.Low.idxmin()],
                        y=TSLA.Low.min(),
                        xref="x", yref="y",
                        text="Low Min:<br> %.3f" % TSLA.Low.min(),
                        ax=0, ay=40),
                   dict(x=TSLA.Date[TSLA.Low.idxmax()],
                         y=TSLA.Low.max(),
                         xref="x", yref="y",
                         text="Low Max:<br> %.3f" % TSLA.Low.max(),
                         ax=-40, ay=-40)]
close_annotations = [dict(x="2015-05-01",
                        y=TSLA.Close.mean(),
                        xref="x", yref="y",
                        text="Close Average:<br> %.3f" % TSLA.Close.mean(),
                        ax=0, ay=40),
                   dict(x=TSLA.Date[TSLA.Close.idxmin()],
                        y=TSLA.Low.min(),
                        xref="x", yref="y",
                        text="Close Min:<br> %.3f" % TSLA.Close.min(),
                        ax=0, ay=40),
                   dict(x=TSLA.Date[TSLA.Close.idxmax()],
                         y=TSLA.Close.max(),
                         xref="x", yref="y",
                         text="Close Max:<br> %.3f" % TSLA.Close.max(),
                         ax=-40, ay=-40)]
volume_annotations = [dict(x="2015-05-01",
                        y=TSLA.Volume.mean(),
                        xref="x", yref="y",
                        text="Volume Average:<br> %.3f" % TSLA.Volume.mean(),
                        ax=0, ay=40),
                   dict(x=TSLA.Date[TSLA.Volume.idxmin()],
                        y=TSLA.Volume.min(),
                        xref="x", yref="y",
                        text="Volume Min:<br> %.3f" % TSLA.Volume.min(),
                        ax=0, ay=40),
                   dict(x=TSLA.Date[TSLA.Volume.idxmax()],
                         y=TSLA.Volume.max(),
                         xref="x", yref="y",
                         text="Volume Max:<br> %.3f" % TSLA.Volume.max(),
                         ax=-40, ay=-40)]

fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label="Descripción general",
                     method="update",
                     args=[{"visible": [True, False, True, False, False, False, False, False]},
                           {"title": "Descripción general de TSLA\n",
                            "annotations": []}]),
                dict(label="Alto",
                     method="update",
                     args=[{"visible": [True, True, False, False, False, False, False, False]},
                           {"title": "TSLA Alto (10 años)\n",
                            "annotations": high_annotations}]),
                dict(label="Bajo",
                     method="update",
                     args=[{"visible": [False, False, True, True, False, False, False, False]},
                           {"title": "TSLA bajo (10 años)\n",
                            "annotations": low_annotations}]),
                dict(label="Cerca",
                     method="update",
                     args=[{"visible": [False, False, False, False, True, True, False, False]},
                           {"title": "Cierre de TSLA (10 años)\n",
                            "annotations": close_annotations}]),
                dict(label="Volumen",
                     method="update",
                     args=[{"visible": [False, False, False, False, False, False, True, True]},
                           {"title": "Volumen de TSLA (10 años)\n",
                            "annotations": volume_annotations}]),
            ]),
        )
    ])

#Establecer título
fig.update_layout(title_text="Indicadores técnicos del precio de las acciones de TSLA Corp (10 años)\n", template='plotly_dark')
fig.show()


#AMZN
fig = go.Figure()

fig.add_trace(go.Scatter(x=list(AMZN.Date),y=list(AMZN.High),name="Alto\n",
    line=dict(color="#33CFA5"),mode="lines",showlegend=True,visible=True))

fig.add_trace(go.Scatter(x=list(AMZN.Date),y=[AMZN.High.mean()] * len(AMZN.Date), name="Alto promedio\n",
               visible=False, line=dict(color="#33CFA5", dash="dash")))

fig.add_trace(go.Scatter(x=list(AMZN.Date),y=list(AMZN.Low),
               name="Bajo\n", line=dict(color="#F06A6A")))

fig.add_trace(go.Scatter(x=list(AMZN.Date), y=[AMZN.Low.mean()] * len(AMZN.Date),name="Bajo promedio\n",
                         visible=False,line=dict(color="#F06A6A", dash="dash")))

fig.add_trace(go.Scatter(x=list(AMZN.Date),y=list(AMZN.Close),visible=False,
               name="Cerca\n", line=dict(color="#fa8825")))

fig.add_trace(go.Scatter(x=list(AMZN.Date),y=[AMZN.Close.mean()] * len(AMZN.Date),name="Cerrar Promedio\n",
                         visible=False,line=dict(color="#fa8825", dash="dash")))

fig.add_trace(go.Scatter(x=list(AMZN.Date),y=list(AMZN.Volume),visible=False,
               name="Volumen\n", line=dict(color="#002f43")))

fig.add_trace(go.Scatter(x=list(AMZN.Date), y=[AMZN.Volume.mean()] * len(AMZN.Date),name="Promedio de volumen\n",visible=False,
               line=dict(color="#002f43", dash="dash")))


#Agregar anotaciones y botones
high_annotations = [dict(x="2016-03-01",
                         y=AMZN.High.mean(),
                         xref="x", yref="y",
                         text="High Average:<br> %.3f" % AMZN.High.mean(),
                         ax=0, ay=-40),
                    dict(x=AMZN.Date[AMZN.High.idxmax()],
                         y=AMZN.High.max(),
                         xref="x", yref="y",
                         text="High Max:<br> %.3f" % AMZN.High.max(),
                         ax=-40, ay=-40),
                    dict(x=AMZN.Date[AMZN.High.idxmin()],
                         y=AMZN.High.min(),
                         xref="x", yref="y",
                         text="High Min:<br> %.3f" % AMZN.High.min(),
                         ax=-40, ay=-40) ]
low_annotations = [dict(x="2015-05-01",
                        y=AMZN.Low.mean(),
                        xref="x", yref="y",
                        text="Low Average:<br> %.3f" % AMZN.Low.mean(),
                        ax=0, ay=40),
                   dict(x=AMZN.Date[AMZN.Low.idxmin()],
                        y=AMZN.Low.min(),
                        xref="x", yref="y",
                        text="Low Min:<br> %.3f" % AMZN.Low.min(),
                        ax=0, ay=40),
                   dict(x=AMZN.Date[AMZN.Low.idxmax()],
                         y=AMZN.Low.max(),
                         xref="x", yref="y",
                         text="Low Max:<br> %.3f" % AMZN.Low.max(),
                         ax=-40, ay=-40)]
close_annotations = [dict(x="2015-05-01",
                        y=AMZN.Close.mean(),
                        xref="x", yref="y",
                        text="Close Average:<br> %.3f" % AMZN.Close.mean(),
                        ax=0, ay=40),
                   dict(x=AMZN.Date[AMZN.Close.idxmin()],
                        y=AMZN.Low.min(),
                        xref="x", yref="y",
                        text="Close Min:<br> %.3f" % AMZN.Close.min(),
                        ax=0, ay=40),
                   dict(x=AMZN.Date[AMZN.Close.idxmax()],
                         y=AMZN.Close.max(),
                         xref="x", yref="y",
                         text="Close Max:<br> %.3f" % AMZN.Close.max(),
                         ax=-40, ay=-40)]
volume_annotations = [dict(x="2015-05-01",
                        y=AMZN.Volume.mean(),
                        xref="x", yref="y",
                        text="Volume Average:<br> %.3f" % AMZN.Volume.mean(),
                        ax=0, ay=40),
                   dict(x=AMZN.Date[AMZN.Volume.idxmin()],
                        y=AMZN.Volume.min(),
                        xref="x", yref="y",
                        text="Volume Min:<br> %.3f" % AMZN.Volume.min(),
                        ax=0, ay=40),
                   dict(x=AMZN.Date[AMZN.Volume.idxmax()],
                         y=AMZN.Volume.max(),
                         xref="x", yref="y",
                         text="Volume Max:<br> %.3f" % AMZN.Volume.max(),
                         ax=-40, ay=-40)]

fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label="Descripción general",
                     method="update",
                     args=[{"visible": [True, False, True, False, False, False, False, False]},
                           {"title": "Descripción general de AMZN\n",
                            "annotations": []}]),
                dict(label="Alto",
                     method="update",
                     args=[{"visible": [True, True, False, False, False, False, False, False]},
                           {"title": "AMZN Alto (10 años)\n",
                            "annotations": high_annotations}]),
                dict(label="Bajo",
                     method="update",
                     args=[{"visible": [False, False, True, True, False, False, False, False]},
                           {"title": "AMZN bajo (10 años)\n",
                            "annotations": low_annotations}]),
                dict(label="Cerca",
                     method="update",
                     args=[{"visible": [False, False, False, False, True, True, False, False]},
                           {"title": "Cierre de AMZN (10 años)\n",
                            "annotations": close_annotations}]),
                dict(label="Volumen",
                     method="update",
                     args=[{"visible": [False, False, False, False, False, False, True, True]},
                           {"title": "Volumen de AMZN (10 años)\n",
                            "annotations": volume_annotations}]),
            ]),
        )
    ])

#Establecer título
fig.update_layout(title_text="Indicadores técnicos del precio de las acciones de AMZN Corp (10 años)\n", template='plotly_dark')
fig.show()


#GOOGL
fig = go.Figure()

fig.add_trace(go.Scatter(x=list(GOOGL.Date),y=list(GOOGL.High),name="Alto\n",
    line=dict(color="#33CFA5"),mode="lines",showlegend=True,visible=True))

fig.add_trace(go.Scatter(x=list(GOOGL.Date),y=[GOOGL.High.mean()] * len(GOOGL.Date), name="Alto promedio\n",
               visible=False, line=dict(color="#33CFA5", dash="dash")))

fig.add_trace(go.Scatter(x=list(GOOGL.Date),y=list(GOOGL.Low),
               name="Bajo\n", line=dict(color="#F06A6A")))

fig.add_trace(go.Scatter(x=list(GOOGL.Date), y=[GOOGL.Low.mean()] * len(GOOGL.Date),name="Bajo promedio\n",
                         visible=False,line=dict(color="#F06A6A", dash="dash")))

fig.add_trace(go.Scatter(x=list(GOOGL.Date),y=list(GOOGL.Close),visible=False,
               name="Cerca\n", line=dict(color="#fa8825")))

fig.add_trace(go.Scatter(x=list(GOOGL.Date),y=[GOOGL.Close.mean()] * len(GOOGL.Date),name="Cerrar Promedio\n",
                         visible=False,line=dict(color="#fa8825", dash="dash")))

fig.add_trace(go.Scatter(x=list(GOOGL.Date),y=list(GOOGL.Volume),visible=False,
               name="Volumen\n", line=dict(color="#002f43")))

fig.add_trace(go.Scatter(x=list(GOOGL.Date),
               y=[GOOGL.Volume.mean()] * len(GOOGL.Date),name="Promedio de volumen\n",visible=False,
               line=dict(color="#002f43", dash="dash")))


#Agregar anotaciones y botones
high_annotations = [dict(x="2016-03-01",
                         y=GOOGL.High.mean(),
                         xref="x", yref="y",
                         text="High Average:<br> %.3f" % GOOGL.High.mean(),
                         ax=0, ay=-40),
                    dict(x=GOOGL.Date[GOOGL.High.idxmax()],
                         y=GOOGL.High.max(),
                         xref="x", yref="y",
                         text="High Max:<br> %.3f" % GOOGL.High.max(),
                         ax=-40, ay=-40),
                    dict(x=GOOGL.Date[GOOGL.High.idxmin()],
                         y=GOOGL.High.min(),
                         xref="x", yref="y",
                         text="High Min:<br> %.3f" % GOOGL.High.min(),
                         ax=-40, ay=-40) ]
low_annotations = [dict(x="2015-05-01",
                        y=GOOGL.Low.mean(),
                        xref="x", yref="y",
                        text="Low Average:<br> %.3f" % GOOGL.Low.mean(),
                        ax=0, ay=40),
                   dict(x=GOOGL.Date[GOOGL.Low.idxmin()],
                        y=GOOGL.Low.min(),
                        xref="x", yref="y",
                        text="Low Min:<br> %.3f" % GOOGL.Low.min(),
                        ax=0, ay=40),
                   dict(x=GOOGL.Date[GOOGL.Low.idxmax()],
                         y=GOOGL.Low.max(),
                         xref="x", yref="y",
                         text="Low Max:<br> %.3f" % GOOGL.Low.max(),
                         ax=-40, ay=-40)]
close_annotations = [dict(x="2015-05-01",
                        y=GOOGL.Close.mean(),
                        xref="x", yref="y",
                        text="Close Average:<br> %.3f" % GOOGL.Close.mean(),
                        ax=0, ay=40),
                   dict(x=GOOGL.Date[GOOGL.Close.idxmin()],
                        y=GOOGL.Low.min(),
                        xref="x", yref="y",
                        text="Close Min:<br> %.3f" % GOOGL.Close.min(),
                        ax=0, ay=40),
                   dict(x=GOOGL.Date[GOOGL.Close.idxmax()],
                         y=GOOGL.Close.max(),
                         xref="x", yref="y",
                         text="Close Max:<br> %.3f" % GOOGL.Close.max(),
                         ax=-40, ay=-40)]
volume_annotations = [dict(x="2015-05-01",
                        y=GOOGL.Volume.mean(),
                        xref="x", yref="y",
                        text="Volume Average:<br> %.3f" % GOOGL.Volume.mean(),
                        ax=0, ay=40),
                   dict(x=GOOGL.Date[GOOGL.Volume.idxmin()],
                        y=GOOGL.Volume.min(),
                        xref="x", yref="y",
                        text="Volume Min:<br> %.3f" % GOOGL.Volume.min(),
                        ax=0, ay=40),
                   dict(x=GOOGL.Date[GOOGL.Volume.idxmax()],
                         y=GOOGL.Volume.max(),
                         xref="x", yref="y",
                         text="Volume Max:<br> %.3f" % GOOGL.Volume.max(),
                         ax=-40, ay=-40)]

fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label="Descripción general",
                     method="update",
                     args=[{"visible": [True, False, True, False, False, False, False, False]},
                           {"title": "Descripción general de GOOGL\n",
                            "annotations": []}]),
                dict(label="Alto",
                     method="update",
                     args=[{"visible": [True, True, False, False, False, False, False, False]},
                           {"title": "GOOGL Alto (10 años)\n",
                            "annotations": high_annotations}]),
                dict(label="Bajo",
                     method="update",
                     args=[{"visible": [False, False, True, True, False, False, False, False]},
                           {"title": "GOOGL bajo (10 años)\n",
                            "annotations": low_annotations}]),
                dict(label="Cerca",
                     method="update",
                     args=[{"visible": [False, False, False, False, True, True, False, False]},
                           {"title": "Cierre de GOOGL (10 años)\n",
                            "annotations": close_annotations}]),
                dict(label="Volumen",
                     method="update",
                     args=[{"visible": [False, False, False, False, False, False, True, True]},
                           {"title": "Volumen de GOOGL (10 años)\n",
                            "annotations": volume_annotations}]),
            ]),
        )
    ])

#Establecer título
fig.update_layout(title_text="Indicadores técnicos del precio de las acciones de GOOGL Corp (10 años)\n", template='plotly_dark')
fig.show()



#META
fig = go.Figure()

fig.add_trace(go.Scatter(x=list(META.Date),y=list(META.High),name="Alto\n",
    line=dict(color="#33CFA5"),mode="lines",showlegend=True,visible=True))

fig.add_trace(go.Scatter(x=list(META.Date),y=[META.High.mean()] * len(META.Date), name="Alto promedio\n",
               visible=False, line=dict(color="#33CFA5", dash="dash")))

fig.add_trace(go.Scatter(x=list(META.Date),y=list(META.Low),
               name="Bajo\n", line=dict(color="#F06A6A")))

fig.add_trace(go.Scatter(x=list(META.Date), y=[META.Low.mean()] * len(META.Date),name="Bajo promedio\n",
                         visible=False,line=dict(color="#F06A6A", dash="dash")))

fig.add_trace(go.Scatter(x=list(META.Date),y=list(META.Close),visible=False,
               name="Cerca\n", line=dict(color="#fa8825")))

fig.add_trace(go.Scatter(x=list(META.Date),y=[META.Close.mean()] * len(META.Date),name="Cerrar Promedio\n",
                         visible=False,line=dict(color="#fa8825", dash="dash")))

fig.add_trace(go.Scatter(x=list(META.Date),y=list(META.Volume),visible=False,
               name="Volumen\n", line=dict(color="#002f43")))

fig.add_trace(go.Scatter(x=list(META.Date),
               y=[META.Volume.mean()] * len(META.Date),name="Promedio de volumen\n",visible=False,
               line=dict(color="#002f43", dash="dash")))


#Agregar anotaciones y botones
high_annotations = [dict(x="2016-03-01",
                         y=META.High.mean(),
                         xref="x", yref="y",
                         text="High Average:<br> %.3f" % META.High.mean(),
                         ax=0, ay=-40),
                    dict(x=META.Date[META.High.idxmax()],
                         y=META.High.max(),
                         xref="x", yref="y",
                         text="High Max:<br> %.3f" % META.High.max(),
                         ax=-40, ay=-40),
                    dict(x=META.Date[META.High.idxmin()],
                         y=META.High.min(),
                         xref="x", yref="y",
                         text="High Min:<br> %.3f" % META.High.min(),
                         ax=-40, ay=-40) ]
low_annotations = [dict(x="2015-05-01",
                        y=META.Low.mean(),
                        xref="x", yref="y",
                        text="Low Average:<br> %.3f" % META.Low.mean(),
                        ax=0, ay=40),
                   dict(x=META.Date[META.Low.idxmin()],
                        y=META.Low.min(),
                        xref="x", yref="y",
                        text="Low Min:<br> %.3f" % META.Low.min(),
                        ax=0, ay=40),
                   dict(x=META.Date[META.Low.idxmax()],
                         y=META.Low.max(),
                         xref="x", yref="y",
                         text="Low Max:<br> %.3f" % META.Low.max(),
                         ax=-40, ay=-40)]
close_annotations = [dict(x="2015-05-01",
                        y=META.Close.mean(),
                        xref="x", yref="y",
                        text="Close Average:<br> %.3f" % META.Close.mean(),
                        ax=0, ay=40),
                   dict(x=META.Date[META.Close.idxmin()],
                        y=META.Low.min(),
                        xref="x", yref="y",
                        text="Close Min:<br> %.3f" % META.Close.min(),
                        ax=0, ay=40),
                   dict(x=META.Date[META.Close.idxmax()],
                         y=META.Close.max(),
                         xref="x", yref="y",
                         text="Close Max:<br> %.3f" % META.Close.max(),
                         ax=-40, ay=-40)]
volume_annotations = [dict(x="2015-05-01",
                        y=META.Volume.mean(),
                        xref="x", yref="y",
                        text="Volume Average:<br> %.3f" % META.Volume.mean(),
                        ax=0, ay=40),
                   dict(x=META.Date[META.Volume.idxmin()],
                        y=META.Volume.min(),
                        xref="x", yref="y",
                        text="Volume Min:<br> %.3f" % META.Volume.min(),
                        ax=0, ay=40),
                   dict(x=META.Date[META.Volume.idxmax()],
                         y=META.Volume.max(),
                         xref="x", yref="y",
                         text="Volume Max:<br> %.3f" % META.Volume.max(),
                         ax=-40, ay=-40)]

fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label="Descripción general",
                     method="update",
                     args=[{"visible": [True, False, True, False, False, False, False, False]},
                           {"title": "Descripción general de META\n",
                            "annotations": []}]),
                dict(label="Alto",
                     method="update",
                     args=[{"visible": [True, True, False, False, False, False, False, False]},
                           {"title": "META Alto (10 años)\n",
                            "annotations": high_annotations}]),
                dict(label="Bajo",
                     method="update",
                     args=[{"visible": [False, False, True, True, False, False, False, False]},
                           {"title": "META bajo (10 años)\n",
                            "annotations": low_annotations}]),
                dict(label="Cerca",
                     method="update",
                     args=[{"visible": [False, False, False, False, True, True, False, False]},
                           {"title": "Cierre de META (10 años)\n",
                            "annotations": close_annotations}]),
                dict(label="Volumen",
                     method="update",
                     args=[{"visible": [False, False, False, False, False, False, True, True]},
                           {"title": "Volumen de META (10 años)\n",
                            "annotations": volume_annotations}]),
            ]),
        )
    ])

#Establecer título
fig.update_layout(title_text="Indicadores técnicos del precio de las acciones de META Corp (10 años)\n", template='plotly_dark')
fig.show()