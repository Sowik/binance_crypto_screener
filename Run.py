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
klines = client.get_historical_klines("ETHUSDT", Client.KLINE_INTERVAL_4HOUR, "16 days ago UTC")

# candlestick_data = ["Open time", "Open", "High", "Low", "Close", "Volume"]
candlestick_data = ["Close"]

dt1 = pd.DataFrame([x[4:5] for x in klines], columns=candlestick_data)

pd.set_option("display.max_columns", None)
pd.set_option('display.max_rows', None)

dt1 = dt1.astype('float64')

n = 14


def getKlines4h(pairName):
    klines = client.get_historical_klines(pairName, Client.KLINE_INTERVAL_4HOUR, "16 days ago UTC")

    candlestick_data = ["Close"]

    dt1 = pd.DataFrame([x[4:5] for x in klines], columns=candlestick_data)

    dt1 = dt1.astype('float64')
    return dt1


def getKlines1h(pairName):
    klines = client.get_historical_klines(pairName, Client.KLINE_INTERVAL_1HOUR, "16 days ago UTC")

    candlestick_data = ["Close"]

    dt1 = pd.DataFrame([x[4:5] for x in klines], columns=candlestick_data)

    dt1 = dt1.astype('float64')
    return dt1


def getRsiForEachPair(pairName):
    dfSymbol = getKlines1h(pairName)
    dfSymRsi = rsi(dfSymbol)
    # df['rsi'] = dfSymRsi['rsi_14'].iloc[-1]
    if (dfSymRsi['rsi_14'].iloc[-1] < 31):
        print(pairName + " " + str(dfSymRsi['rsi_14'].iloc[-1]))
    return dfSymRsi['rsi_14'].iloc[-1]


# running moving average for rsi using numpy(no loops)
def rma(x, n, y0):
    a = (n - 1) / n
    ak = a ** np.arange(len(x) - 1, -1, -1)
    result = np.empty(n, dtype="float64")
    result = np.r_[np.full(n, np.nan), y0, np.cumsum(ak * x) / ak / n + y0 * a ** np.arange(1, len(x) + 1)]
    return result
    # return np.r_[np.full(n, np.nan), y0, np.cumsum(ak * x) / ak / n + y0 * a ** np.arange(1, len(x) + 1)]


def rsi(dfCrypto):
    df_rsi = dfCrypto
    df_rsi['Change'] = df_rsi['Close'].diff()
    df_rsi["Gain"] = df_rsi.Change.mask(df_rsi.Change < 0, 0.0)
    df_rsi["Loss"] = -df_rsi.Change.mask(df_rsi.Change > 0, -0.0)
    df_rsi['Avg_gain'] = rma(df_rsi.Gain[n + 1:].to_numpy(), n, np.nansum(df_rsi.Gain.to_numpy()[:n + 1]) / n)
    df_rsi['Avg_loss'] = rma(df_rsi.Loss[n + 1:].to_numpy(), n, np.nansum(df_rsi.Loss.to_numpy()[:n + 1]) / n)
    df_rsi['rs'] = df_rsi.Avg_gain / df_rsi.Avg_loss
    df_rsi['rsi_14'] = 100 - (100 / (1 + df_rsi.rs))

    # print(df_rsi.round(2))

    return df_rsi


def rsi_price_change(dfCrypto):
    df_rsi = dfCrypto
    df_rsi['Change'] = df_rsi['Close'].diff()
    df_rsi["Gain"] = df_rsi.Change.mask(df_rsi.Change < 0, 0.0)
    df_rsi["Loss"] = -df_rsi.Change.mask(df_rsi.Change > 0, -0.0)
    df_rsi['Avg_gain'] = rma(df_rsi.Gain[n + 1:].to_numpy(), n, np.nansum(df_rsi.Gain.to_numpy()[:n + 1]) / n)
    df_rsi['Avg_loss'] = rma(df_rsi.Loss[n + 1:].to_numpy(), n, np.nansum(df_rsi.Loss.to_numpy()[:n + 1]) / n)
    df_rsi['rs'] = df_rsi.Avg_gain / df_rsi.Avg_loss
    df_rsi['rsi_14'] = 100 - (100 / (1 + df_rsi.rs))
    #calculate change after 4 hours
    # df_rsi.loc[df_rsi['rsi_14'] <= 30, 'change4h'] = df_rsi.Close.pct_change(periods=-1)*100
    df_rsi.loc[df_rsi['rsi_14'] <= 30, 'change4h'] = ((df_rsi.Close.shift(-1)-df_rsi.Close)/df_rsi.Close)*100
    df_rsi.loc[df_rsi['rsi_14'] <= 30, 'change12h'] = ((df_rsi.Close.shift(-3) - df_rsi.Close) / df_rsi.Close) * 100
    df_rsi.loc[df_rsi['rsi_14'] <= 30, 'change24h'] = ((df_rsi.Close.shift(-6) - df_rsi.Close) / df_rsi.Close) * 100
    df_rsi.loc[df_rsi['rsi_14'] <= 30, 'change48h'] = ((df_rsi.Close.shift(-12) - df_rsi.Close) / df_rsi.Close) * 100

    # print(df_rsi.round(2))

    return df_rsi


def getPairs():
    allPairs = client.get_exchange_info()

    testd = pd.DataFrame(allPairs['symbols'])

    # mask = testd['symbol'].str.contains('USDT')
    mask = testd['symbol'].str.endswith('USDT')

    testd = testd[mask]

    # removing BULL BEAR UP DOWN
    testd = testd[~testd.symbol.str.contains("BULL", na=False)]
    testd = testd[~testd.symbol.str.contains("BEAR", na=False)]
    testd = testd[~testd.symbol.str.contains("UP", na=False)]
    testd = testd[~testd.symbol.str.contains("DOWN", na=False)]

    # testd = testd['symbol'] becomes series because of this
    testd.drop(testd.columns.difference(['symbol']), 1, inplace=True)
    print(type(testd))

    testd = testd.iloc[:-5]
    print(testd)
    return testd


# df = rsi(dt1)
# df = rsi_price_change(dt1)  # price change after rsi lower than 30
# print(df)

# df.plot()
# df.to_excel("output.xlsx")

df2 = getPairs()
df2['rsi'] = df2.apply(lambda x: getRsiForEachPair(x.symbol), axis=1)

print(df2)

print("--- %s seconds ---" % (time.time() - start_time))
