import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# Load list of tickers
def loadTickers():
    f = open('./tickerList.txt', 'r')
    tickers = []
    for line in f.readlines():
        tickers.append(line.replace('\n', ''))
    f.close()
    return tickers

# Plot historical data
def plotHistData(ticker, data):
    plt.figure()
    plt.plot(data)
    print('IN PLOT HIST DATA')
    plt.title(ticker)

# Save date, close, return for tickers
def saveTickerCloses(ticker, show=False):
    histData =  yf.Ticker(ticker).history(period='max', interval='1d')
    if('Adj Close' in histData.columns):
        print('FOUND ADJ CLOSE: ', ticker)
    for col in histData.columns:
        if col != 'Close' and col != 'Adj Close':
            histData = histData.drop(columns=col)
    histData['Closem1'] = histData['Close'].shift(1)
    histData['Returns'] = histData['Close'] / histData['Closem1'] - 1
    histData = histData.drop(columns='Closem1')
    histData.to_csv(f'./historyData/{ticker}')

    if show:
        print("PLOTTING")
        plotHistData(ticker, histData["Close"].values)


# Load ticker data
def loadTickerData(ticker):
    f = open(f'historyData/{ticker}')
    dates = []
    prices = []
    returns = []
    for line in f.readlines()[2:]:
        date, price, returnVal = line.split(',')
        dates.append(date)
        prices.append(float(price))
        returns.append(float(returnVal))
    f.close()
    return dates, prices, returns

# Get mean and standard deviation of ticker
def getMeanStdevOfStock(ticker):
    dates, prices = loadTickerData(ticker)
    returns = []
    for i in range(len(prices)-1):
        pass

# Get correlation of two data series
def correlation(data1, data2):
    if len(data1) < len(data2):
        data2 = data2[-len(data1):]
    elif len(data2) < len(data1):
        data1 = data1[-len(data2):]
    return np.corrcoef(data1,data2)[1,0]

# Get covariance of two data series
def covariance(data1, data2):
    if len(data1) < len(data2):
        data2 = data2[-len(data1):]
    elif len(data2) < len(data1):
        data1 = data1[-len(data2):]
    return np.cov(data1,data2)[1,0]