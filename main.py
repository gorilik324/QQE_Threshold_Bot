import MetaTrader5 as mt
import pandas as pd
import talib as ta
import numpy as np
import math
from datetime import datetime
import warnings 
warnings.filterwarnings('ignore')

def nz(source, replacement):

    if source == np.nan:
        result = 0
    else:
        result = replacement

    return result

def ma(type, src, len):

    result = 0.0

    if type == "SMA": # Simple
        result = ta.SMA(src, len)
    elif type == "EMA": # Exponentail
        result = ta.EMA(src, len)
    elif type == "DEMA": # Double Exponential
        e = ta.EMA(src, len)
        result = 2 * e - ta.EMA(e, len)
    elif type == "TEMA": # Triple Exponential
        e = ta.EMA(src, len)
        result = 3 * (e - ta.EMA(e, len)) + ta.EMA(ta.EMA(e, len), len)
    elif type == "WMA": # Weighted
        result = ta.WMA(src, len)
    elif type == "VWMA": # Volume Weighted
        # result = ta.VWMA(src, len)
        print("Not Available Right Now")
    elif type == "SMMA": # Smoothed
        w = ta.WMA(src, len)
        # result := na(w[1]) ? sma(src, len) : (w[1] * (len - 1) + src) / len
        print("Not Available Right Now")
    elif type == "HMA": # Hull
        # result = ta.WMA(2 * ta.WMA(src, len / 2) - ta.WMA(src, len), round(math.sqrt(len)))
        print("Not Available Right Now")
    elif type == "LSMA": # Least Squares
        # result = linreg(src, len, lsma_offset)
        print("Not Available Right Now")
    elif type == "ALMA": # Arnaud Legoux
        # result = alma(src, len, alma_offset, alma_sigma)
        print("Not Available Right Now")
    elif type == "PEMA":
        # Copyright (c) 2010-present, Bruno Pio
        # Copyright (c) 2019-present, Alex Orekhov (everget)
        # Pentuple Exponential Moving Average script may be freely distributed under the MIT license.
        ema1 = ta.EMA(src, len)
        ema2 = ta.EMA(ema1, len)
        ema3 = ta.EMA(ema2, len)
        ema4 = ta.EMA(ema3, len)
        ema5 = ta.EMA(ema4, len)
        ema6 = ta.EMA(ema5, len)
        ema7 = ta.EMA(ema6, len)
        ema8 = ta.EMA(ema7, len)
        pema = 8 * ema1 - 28 * ema2 + 56 * ema3 - 70 * ema4 + 56 * ema5 - 28 * ema6 + 8 * ema7 - ema8
        result = pema

    return result

def calc_rsi():

    src = df['close'] # RSI Source
    Wilders_Period = (RSI_Period * 2) - 1

    Rsi = ta.RSI(src, RSI_Period)
    df['Rsi'] = Rsi

    RsiMa = ma(ma_type,  df['Rsi'], SF)
    df['RsiMa'] = RsiMa

    df['AtrRsi'] = np.nan

    for curr in range(0, len(df.index)):

        AtrRsi = abs(df.iloc[curr - 1].RsiMa - df.iloc[curr].RsiMa)
        df['AtrRsi'][curr] = AtrRsi

    MaAtrRsi = ma(ma_type, df['AtrRsi'], Wilders_Period)
    df['MaAtrRsi'] = MaAtrRsi

    dar = ma(ma_type, df['MaAtrRsi'], Wilders_Period)
    df['dar'] = dar

    for curr in range(0, len(df.index)):

        df['dar'][curr] = df['dar'][curr] * QQE

    df['longband'] = 0.0
    df['shortband'] = 0.0
    df['trend'] = np.nan

    DeltaFastAtrRsi = df['dar']
    RSIndex = df['RsiMa']

    df['newshortband'] = np.nan
    df['newlongband'] = np.nan
    df['FastAtrRsiTL'] = np.nan

    for curr in range(0, len(df.index)):

        newshortband = df['RsiMa'][curr] + df['dar'][curr]
        df['newshortband'][curr] = newshortband

        newlongband = df['RsiMa'][curr] - df['dar'][curr]
        df['newlongband'][curr] = newlongband

    for curr in range(1, len(df.index)):

        if df['RsiMa'][curr - 1] > df['longband'][curr - 1] and df['RsiMa'][curr] > df['longband'][curr - 1]:
            longband = max(df['longband'][curr - 1], df['newlongband'][curr])
            df['longband'][curr] = longband
        else:
            longband = df['newlongband'][curr]
            df['longband'][curr] = longband

        if df['RsiMa'][curr - 1] < df['shortband'][curr - 1] and df['RsiMa'][curr] < df['shortband'][curr - 1]:
            shortband = min(df['shortband'][curr - 1], df['newshortband'][curr])
            df['shortband'][curr] = shortband
        else:
            shortband = df['newshortband'][curr]
            df['shortband'][curr] = shortband

    for curr in range(1, len(df.index)):

        cross_1 = df['longband'][curr - 1] > df['RsiMa'][curr]

        if df['RsiMa'][curr] > df['shortband'][curr - 1]:
            trend = 1
        elif cross_1:
            trend = -1
        else:
            trend = nz(df['trend'][curr - 1], 1)

        df['trend'][curr] = trend

        if trend == 1:
            FastAtrRsiTL = df['longband'][curr]
        else:
            FastAtrRsiTL = df['shortband'][curr]

        df['FastAtrRsiTL'][curr] = FastAtrRsiTL

def qqe_crosses():

    df['QQExlong'] = 0
    df['QQExshort'] = 0
    df['QQEzlong'] = 0
    df['QQEzshort'] = 0
    df['QQEclong'] = 0
    df['QQEcshort'] = 0

    for curr in range(1, len(df.index)):

        # Find all the QQE Crosses

        if df['FastAtrRsiTL'][curr] < df['dar'][curr]:
            df['QQExlong'][curr] += 1
        if df['FastAtrRsiTL'][curr] > df['dar'][curr]:
            df['QQExshort'][curr] += 1

        # Zero cross

        if df['dar'][curr] >= 50:
            df['QQEzlong'] += 1
        if df['dar'][curr] < 50:
            df['QQEzshort'] += 1

        # Thresh Hold channel Crosses give the BUY/SELL alerts.
        
        if df['dar'][curr] > (50 + ThreshHold):
            df['QQEclong'][curr] += 1
        if df['dar'][curr] < (50 - ThreshHold):
            df['QQEcshort'][curr] += 1

    df['QQE_XC_Over_Channel'] = np.nan
    df['QQE_XC_Under_Channel'] = np.nan
    df['QQE_XQ_Cross_Over'] = np.nan
    df['QQE_XQ_Cross_Under'] = np.nan
    df['QQE_XZ_Zero_Cross_Over'] = np.nan
    df['QQE_XZ_Zero_Cross_Under'] = np.nan
    df['QQE_Direction'] = np.nan

    for curr in range(1, len(df.index)):

        # QQE exit from Thresh Hold Channel

        if df['QQEclong'][curr] == 1:
            df['QQE_XC_Over_Channel'][curr] = df['RsiMa'][curr] - 50
        if df['QQEcshort'][curr] == 1:
            df['QQE_XC_Under_Channel'][curr] = df['RsiMa'][curr] - 50

        # QQE crosses

        if df['QQExlong'][curr] == 1:
            df['QQE_XQ_Cross_Over'][curr] = df['FastAtrRsiTL'][curr - 1] - 50
        if df['QQExshort'][curr] == 1:
            df['QQE_XQ_Cross_Under'][curr] = df['FastAtrRsiTL'][curr - 1] - 50

        # Signal crosses zero line

        if df['QQEzlong'][curr] == 1:
            df['QQE_XZ_Zero_Cross_Over'][curr] = df['RsiMa'][curr] - 50
        if df['QQEzshort'][curr] == 1:
            df['QQE_XZ_Zero_Cross_Under'][curr] = df['RsiMa'][curr] - 50
        
        # Direction

        if df['RsiMa'][curr] - 50 > ThreshHold:
            df['QQE_Direction'][curr] = "buy"
        elif df['RsiMa'][curr] - 50 < 0 - ThreshHold:
            df['QQE_Direction'][curr] = "sell"
        else:
            df['QQE_Direction'][curr] = "NoPosition"
    

SYMBOL = "XAUUSD"
TIMEFRAME = mt.TIMEFRAME_M30

RSI_Period = 14 # RSI Length
SF = 5 # Smoothing
QQE = 4.238 # Fast QQE Factor
ThreshHold = 10 # Thresh-Hold

ma_type = "EMA" # Options=["ALMA", "EMA", "DEMA", "TEMA", "WMA", "VWMA", "SMA", "SMMA", "HMA", "LSMA", "PEMA"]
lsma_offset = 0 # * Least Squares (LSMA) Only - Offset Value
alma_offset = 0.85 # * Arnaud Legoux (ALMA) Only - Offset Value
alma_sigma = 6 # * Arnaud Legoux (ALMA) Only - Sigma Value

# sQQEx = False # Show Smooth RSI, QQE Signal crosses
# sQQEz = False # Show Smooth RSI Zero crosses
# sQQEc = False # Show Smooth RSI Thresh Hold Channel Exits
# inpDrawBars = True #Color bars?

mt.initialize()

# while True:
bars_data = bars = mt.copy_rates_range(SYMBOL,TIMEFRAME,datetime(2020,1,19),datetime.now())
df = pd.DataFrame(bars_data)

calc_rsi()
qqe_crosses()

df.to_csv("QQE_ThreshHold.csv")