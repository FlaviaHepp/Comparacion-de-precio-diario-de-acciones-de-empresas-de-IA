
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


#Definir una función para realizar simulación y trazado para cada empresa
def simulate_and_plot(data, company_name):
    #Preprocesar datos para la empresa
    data_ps = data.copy()
    data_ps.set_index('Date', inplace=True)
    data_prs = data_ps['Close']

    log_returns = np.log(1 + data_prs.pct_change())

    u = log_returns.mean()
    var = log_returns.var()
    stdev = log_returns.std()

    drift = u - (0.5 * var)
    np.array(drift)

    t_intervals = 365
    iterations = 6

    daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(t_intervals, iterations)))
    S0 = data_prs.iloc[-1]

    prices_list = np.zeros_like(daily_returns)
    prices_list[0] = S0

    for t in range(1, t_intervals):
        prices_list[t] = prices_list[t-1] * daily_returns[t]

    dates = pd.date_range(start='2024-03-19', periods=t_intervals)

    #Simulación de parcela para la empresa
    fig = go.Figure()
    for i in range(iterations):
        fig.add_trace(go.Scatter(x=dates, y=prices_list[:, i], mode='lines', name=f'Simulation {i+1}'))

    #Agregar información de diseño
    fig.update_layout(title=f'{company_name} Simulación del precio de las acciones\n',
                       xaxis_title='Fecha\n',
                       yaxis_title='Precio (dólares americanos)\n',
                       hovermode='x',
                       template='plotly_dark')

    #mostrar la trama
    fig.show()

AMZN = pd.read_csv("AMZN1424.csv")
simulate_and_plot(AMZN, "AMZN")
print("\nAMAZON\n", AMZN)

GOOGL = pd.read_csv("GOOGL1424.csv")
simulate_and_plot(GOOGL, "GOOGL")
print("\nGOOGLE\n", GOOGL)

AAPL = pd.read_csv("AAPL1424.csv")
simulate_and_plot(AAPL, "AAPL")
print("\nAPPLE\n", AAPL)

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
print("\nTESLA\n", TSLA)

#Obtener la fecha de hoy
current_date = datetime.today().strftime('%Y-%m-%d')


#CSV's e ingeniería de funciones
#AMAZON
AMZN= yf.download('AMZN', start='2014-01-02', end=current_date)
AMZN.reset_index(inplace=True)

#Aplicar algunos análisis técnicos a
AMZN['ATR (7 Days)'] = ta.volatility.average_true_range(high=AMZN['High'], low=AMZN['Low'], close=AMZN['Close'], window=7, fillna=True)
AMZN['RSI (7 Days)'] = ta.momentum.RSIIndicator(AMZN['Close'], window=7, fillna = True).rsi()
AMZN['RSI (14 Days)'] = ta.momentum.RSIIndicator(AMZN['Close'], window=14, fillna= True).rsi()
AMZN['EMA (50 Days)'] = ta.trend.ema_indicator(AMZN['Close'], window=50, fillna=True)
AMZN['EMA (100 Days)'] = ta.trend.ema_indicator(AMZN['Close'], window=100, fillna=True)
AMZN['SMA (50 Days)'] = ta.trend.sma_indicator(AMZN['Close'], window=50, fillna=True)
AMZN['SMA (100 Days)'] = ta.trend.sma_indicator(AMZN['Close'], window=100, fillna=True)

AMZN['Log Return'] = np.log(AMZN['Close'] / AMZN['Close'].shift(1)) #Calcular los retornos de registros diarios
AMZN['Percent Return'] = AMZN['Close'].pct_change() #Calcular el porcentaje de rendimiento diario
AMZN['Cumulative Log Return'] = AMZN['Log Return'].cumsum().ffill() #Calcular los rendimientos de registros acumulados
AMZN['Cumulative Percent Return'] = AMZN['Percent Return'].cumsum().ffill() #Calcular los rendimientos porcentuales acumulados

AMZN.isnull().sum()


#GOOGLE ALPHABET
GOOGL= yf.download('GOOGL', start='2014-01-02', end=current_date)
GOOGL.reset_index(inplace=True)

GOOGL['ATR (7 Days)'] = ta.volatility.average_true_range(high=GOOGL['High'], low=GOOGL['Low'], close=GOOGL['Close'], window=7, fillna=True)
GOOGL['RSI (7 Days)'] = ta.momentum.RSIIndicator(GOOGL['Close'], window=7, fillna = True).rsi()
GOOGL['RSI (14 Days)'] = ta.momentum.RSIIndicator(GOOGL['Close'], window=14, fillna= True).rsi()
GOOGL['EMA (50 Days)'] = ta.trend.ema_indicator(GOOGL['Close'], window=50, fillna=True)
GOOGL['EMA (100 Days)'] = ta.trend.ema_indicator(GOOGL['Close'], window=100, fillna=True)
GOOGL['SMA (50 Days)'] = ta.trend.sma_indicator(GOOGL['Close'], window=50, fillna=True)
GOOGL['SMA (100 Days)'] = ta.trend.sma_indicator(GOOGL['Close'], window=100, fillna=True)

GOOGL['Log Return'] = np.log(GOOGL['Close'] / GOOGL['Close'].shift(1)) #Calcular los retornos de registros diarios
GOOGL['Percent Return'] = GOOGL['Close'].pct_change() #Calcular el porcentaje de rendimiento diario
GOOGL['Cumulative Log Return'] = GOOGL['Log Return'].cumsum().ffill() #Calcular los rendimientos de registros acumulados
GOOGL['Cumulative Percent Return'] = GOOGL['Percent Return'].cumsum().ffill() #Calcular los rendimientos porcentuales acumulados


#APPLE
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

META['Log Return'] = np.log(META['Close'] / META['Close'].shift(1)) #Calcular los retornos de registros diarios
META['Percent Return'] = META['Close'].pct_change() #Calcular el porcentaje de rendimiento diario
META['Cumulative Log Return'] = META['Log Return'].cumsum().ffill() #Calcular los rendimientos de registros acumulados
META['Cumulative Percent Return'] = META['Percent Return'].cumsum().ffill() #Calcular los rendimientos porcentuales acumulados


#MICROSOFT
MSFT = yf.download('MSFT', start='2014-01-02', end=current_date)
MSFT.reset_index(inplace=True)

MSFT['ATR (7 Days)'] = ta.volatility.average_true_range(high=MSFT['High'], low=MSFT['Low'], close=MSFT['Close'], window=7, fillna=True)
MSFT['RSI (7 Days)'] = ta.momentum.RSIIndicator(MSFT['Close'], window=7, fillna = True).rsi()
MSFT['RSI (14 Days)'] = ta.momentum.RSIIndicator(MSFT['Close'], window=14, fillna= True).rsi()
MSFT['EMA (50 Days)'] = ta.trend.ema_indicator(MSFT['Close'], window=50, fillna=True)
MSFT['EMA (100 Days)'] = ta.trend.ema_indicator(MSFT['Close'], window=100, fillna=True)
MSFT['SMA (50 Days)'] = ta.trend.sma_indicator(MSFT['Close'], window=50, fillna=True)
MSFT['SMA (100 Days)'] = ta.trend.sma_indicator(MSFT['Close'], window=100, fillna=True)

MSFT['Log Return'] = np.log(MSFT['Close'] / MSFT['Close'].shift(1)) #Calcular los retornos de registros diarios
MSFT['Percent Return'] = MSFT['Close'].pct_change() #Calcular el porcentaje de rendimiento diario
MSFT['Cumulative Log Return'] = MSFT['Log Return'].cumsum().ffill() #Calcular los rendimientos de registros acumulados
MSFT['Cumulative Percent Return'] = MSFT['Percent Return'].cumsum().ffill() #Calcular los rendimientos porcentuales acumulados


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

NVDA['Log Return'] = np.log(NVDA['Close'] / NVDA['Close'].shift(1)) #Calcular los retornos de registros diarios
NVDA['Percent Return'] = NVDA['Close'].pct_change() #Calcular el porcentaje de rendimiento diario
NVDA['Cumulative Log Return'] = NVDA['Log Return'].cumsum().ffill() #Calcular los rendimientos de registros acumulados
NVDA['Cumulative Percent Return'] = NVDA['Percent Return'].cumsum().ffill() #Calcular los rendimientos porcentuales acumulados


#TESLA
TSLA = yf.download('TSLA', start='2014-01-02', end=current_date)
TSLA.reset_index(inplace=True)

TSLA['ATR (7 Days)'] = ta.volatility.average_true_range(high=TSLA['High'], low=TSLA['Low'], close=TSLA['Close'], window=7, fillna=True)
TSLA['RSI (7 Days)'] = ta.momentum.RSIIndicator(TSLA['Close'], window=7, fillna = True).rsi()
TSLA['RSI (14 Days)'] = ta.momentum.RSIIndicator(TSLA['Close'], window=14, fillna= True).rsi()
TSLA['EMA (50 Days)'] = ta.trend.ema_indicator(TSLA['Close'], window=50, fillna=True)
TSLA['EMA (100 Days)'] = ta.trend.ema_indicator(TSLA['Close'], window=100, fillna=True)
TSLA['SMA (50 Days)'] = ta.trend.sma_indicator(TSLA['Close'], window=50, fillna=True)
TSLA['SMA (100 Days)'] = ta.trend.sma_indicator(TSLA['Close'], window=100, fillna=True)

TSLA['Log Return'] = np.log(TSLA['Close'] / TSLA['Close'].shift(1)) #Calcular los retornos de registros diarios
TSLA['Percent Return'] = TSLA['Close'].pct_change() #Calcular el porcentaje de rendimiento diario
TSLA['Cumulative Log Return'] = TSLA['Log Return'].cumsum().ffill() #Calcular los rendimientos de registros acumulados
TSLA['Cumulative Percent Return'] = TSLA['Percent Return'].cumsum().ffill() #Calcular los rendimientos porcentuales acumulados


#EDA con menús desplegables
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

fig1.update_layout(
    title_text="Indicadores técnicos del precio de las acciones de Apple (10 años)\n", template='plotly_dark',

    annotations=[
        dict(text="SMA", x=0, xref="paper", y=1.06, yref="paper",
             align="left", showarrow=False),
        dict(text="RSI", x=0.3, xref="paper", y=1.07,
             yref="paper", showarrow=False),
        dict(text="EMA", x=0.54, xref="paper", y=1.06, yref="paper",
             showarrow=False)
    ])

fig1.show()

fig2 = go.Figure()

fig2.add_trace(go.Candlestick(x=list(AMZN['Date']), open=list(AMZN['Open']),
               high=list(AMZN['High']),low=list(AMZN['Low']),name='Candlesticks',
              close=list(AMZN['Close']),showlegend=True,visible=False,))


fig2.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label="Candelero",
                     method="update",
                     args=[{"visible": [True]},
                           {"title": "Gráfico de velas Amazon 10 años Indicadores técnicos\n",
                            "annotations": []}]),
            ])
        )
    ]
)

#Establecer título
fig2.update_layout(title_text="Gráfico de velas Amazon 10 años Indicadores técnicos\n", template='plotly_dark')
fig2.show()
