from binance.client import Client
import numpy as np
import pandas as pd
import time
import numba
import matplotlib

start_time = time.time()

client = Client("example1",
                "example2")

# klines = client.get_historical_klines("ETHUSDT", Client.KLINE_INTERVAL_4HOUR, "16 days ago UTC")
klines = client.get_historical_klines("ETHUSDT", Client.KLINE_INTERVAL_1DAY, "300 days ago UTC")

# candlestick_data = ["Open time", "Open", "High", "Low", "Close", "Volume"]
candlestick_data = ["Open", "High", "Low", "Close"]

dt1 = pd.DataFrame([x[1:5] for x in klines], columns=candlestick_data)
dt1["DateTime"] = [x[0] for x in klines]

pd.set_option("display.max_columns", None)
pd.set_option('display.max_rows', None)

dt1 = dt1.astype('float64')
dt1.DateTime = pd.to_numeric(dt1["DateTime"])
dt1.DateTime = pd.to_datetime(dt1.DateTime, unit="ms")
dt1 = dt1.set_index("DateTime")

def get_klines_1day(symbol_x):
    klines = client.get_historical_klines(symbol_x, Client.KLINE_INTERVAL_1DAY, "300 days ago UTC")

    candlestick_data = ["Open", "High", "Low", "Close"]

    dt1 = pd.DataFrame([x[1:5] for x in klines], columns=candlestick_data)
    dt1["DateTime"] = [x[0] for x in klines]

    pd.set_option("display.max_columns", None)
    pd.set_option('display.max_rows', None)

    dt1 = dt1.astype('float64')
    dt1.DateTime = pd.to_numeric(dt1["DateTime"])
    dt1.DateTime = pd.to_datetime(dt1.DateTime, unit="ms")
    dt1 = dt1.set_index("DateTime")

    return dt1

def ichimoku(dfCrypto):
    df_ichimoku = dfCrypto

    # Tenkan_sen / Conversion Line
    nine_period_high = df_ichimoku["High"].rolling(window=9).max()
    nine_period_low = df_ichimoku["Low"].rolling(window=9).min()
    df_ichimoku["Conversion_line"] = (nine_period_high + nine_period_low) / 2

    # Kijun-sen / Base Line
    period26_high = df_ichimoku["High"].rolling(window=26).max()
    period26_low = df_ichimoku["Low"].rolling(window=26).min()
    df_ichimoku["Base_line"] = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A)
    df_ichimoku["Leading_span_A"] = (df_ichimoku["Conversion_line"] + df_ichimoku["Base_line"]) / 2

    # Senkou Span B (Leading Span B)
    period52_high = df_ichimoku["High"].rolling(window=52).max()
    period52_low = df_ichimoku["Low"].rolling(window=52).min()
    df_ichimoku["Leading_span_B"] = (period52_high + period52_low) / 2

    # Chikou Span (Lagging Span)
    df_ichimoku["Lagging_span"] = df_ichimoku["Close"].shift(-26)

    return df_ichimoku


def macd(dfCrypto):
    df_macd = dfCrypto
    # EMA
    df_macd["EMA_12"] = df_macd['Close'].ewm(span=12, min_periods=0, adjust=False, ignore_na=False).mean()
    df_macd["EMA_26"] = df_macd['Close'].ewm(span=26, min_periods=0, adjust=False, ignore_na=False).mean()

    # MACD Line
    df_macd["MACD_Line"] = df_macd["EMA_12"] - df_macd["EMA_26"]

    # Signal Line
    df_macd["Signal_Line"] = df_macd['MACD_Line'].ewm(span=9, min_periods=0, adjust=False, ignore_na=False).mean()

    # MACD Histogram
    df_macd["MACD_Histogram"] = df_macd["MACD_Line"] - df_macd["Signal_Line"]

    return df_macd


def macd_add_only(dfCrypto):
    # EMA
    dfCrypto["EMA_12"] = dfCrypto['Close'].ewm(span=12, min_periods=0, adjust=False, ignore_na=False).mean()
    dfCrypto["EMA_26"] = dfCrypto['Close'].ewm(span=26, min_periods=0, adjust=False, ignore_na=False).mean()

    # MACD Line
    dfCrypto["MACD_Line"] = dfCrypto["EMA_12"] - dfCrypto["EMA_26"]

    # Signal Line
    dfCrypto["Signal_Line"] = dfCrypto['MACD_Line'].ewm(span=9, min_periods=0, adjust=False, ignore_na=False).mean()

    # MACD Histogram
    dfCrypto["MACD_Histogram"] = dfCrypto["MACD_Line"] - dfCrypto["Signal_Line"]

    return dfCrypto


def ta_signals(dfCrypto):
    dt = ichimoku(dfCrypto)

    dt = macd_add_only(dt)

    dt.loc[
        (dt["Leading_span_A"] > dt["Leading_span_B"]) & (dt["Leading_span_A"].shift(1) < dt["Leading_span_B"].shift(1)) \
        & (dt["Close"] > dt["Leading_span_A"]), 'Green_cloud'] \
        = True

    dt.loc[
        (dt["MACD_Line"] > 0) & (dt["MACD_Line"].shift(1) <= 0) & (dt["Signal_Line"] > 0) & (dt["Signal_Line"].shift(1) <= 0)\
        ,'MACD_above_zero'] \
        = True

    return dt[["Green_cloud", "MACD_above_zero"]].iloc[-1]

def get_sym_signals(symbol_x):
    dfSymbol = get_klines_1day(symbol_x)
    print(symbol_x)
    dfSymbol = ta_signals(dfSymbol)
    return dfSymbol



def getPairs():
    allPairs = client.get_exchange_info()

    testd = pd.DataFrame(allPairs['symbols'])

    mask1 = testd['status'].str.contains('TRADING')
    testd = testd[mask1]

    mask2 = testd['symbol'].str.endswith('USDT')

    testd = testd[mask2]

    # removing BULL BEAR UP DOWN
    testd = testd[~testd.symbol.str.contains("BULL", na=False)]
    testd = testd[~testd.symbol.str.contains("BEAR", na=False)]
    testd = testd[~testd.symbol.str.contains("UP", na=False)]
    testd = testd[~testd.symbol.str.contains("DOWN", na=False)]
    testd = testd[~testd.symbol.str.contains("BCC", na=False)]


    # testd = testd['symbol'] becomes series because of this
    testd.drop(testd.columns.difference(['symbol']), 1, inplace=True)

    testd = testd.iloc[:-10]
    return testd


df2 = getPairs()
df2[["Green_cloud", "MACD_above_zero"]] = df2.apply(lambda x: get_sym_signals(x.symbol), axis=1)

print(df2)


# #Tenkan_sen / Conversion Line
# nine_period_high = dt1["High"].rolling(window=9).max()
# nine_period_low = dt1["Low"].rolling(window=9).min()
# dt1["Conversion_line"] = (nine_period_high + nine_period_low) / 2
#
# #Kijun-sen / Base Line
# period26_high = dt1["High"].rolling(window=26).max()
# period26_low = dt1["Low"].rolling(window=26).min()
# dt1["Base_line"] = (period26_high + period26_low) / 2
#
# #Senkou Span A (Leading Span A)
# dt1["Leading_span_A"] = (dt1["Conversion_line"] + dt1["Base_line"]) / 2
#
# #Senkou Span B (Leading Span B)
# period52_high = dt1["High"].rolling(window=52).max()
# period52_low = dt1["Low"].rolling(window=52).min()
# dt1["Leading_span_B"] = (period52_high + period52_low) / 2
#
# #Chikou Span (Lagging Span)
# dt1["Lagging_span"] = dt1["Close"].shift(-26)


# #EMA
# dt1["EMA_12"] = dt1['Close'].ewm(span=12, min_periods=0, adjust=False, ignore_na=False).mean()
# dt1["EMA_26"] = dt1['Close'].ewm(span=26, min_periods=0, adjust=False, ignore_na=False).mean()
#
# #MACD Line
# dt1["MACD_Line"] = dt1["EMA_12"] - dt1["EMA_26"]
#
# #Signal Line
# dt1["Signal_Line"] =dt1['MACD_Line'].ewm(span=9, min_periods=0, adjust=False, ignore_na=False).mean()
#
# #MACD Histogram
# dt1["MACD_Histogram"] = dt1["MACD_Line"] - dt1["Signal_Line"]
