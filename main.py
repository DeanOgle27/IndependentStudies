import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

def loadTickers():
    f = open('./tickerList.txt', 'r')
    tickers = []
    for line in f.readlines():
        tickers.append(line.replace('\n', ''))
    f.close()
    return tickers

def plotHistData(ticker, data):
    plt.figure()
    plt.plot(data)
    print('IN PLOT HIST DATA')
    plt.title(ticker)



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

def getMeanStdevOfStock(ticker):
    dates, prices = loadTickerData(ticker)
    returns = []
    for i in range(len(prices)-1):
        pass

def correlation(data1, data2):
    if len(data1) < len(data2):
        data2 = data2[-len(data1):]
    elif len(data2) < len(data1):
        data1 = data1[-len(data2):]
    return np.corrcoef(data1,data2)[1,0]

def covariance(data1, data2):
    if len(data1) < len(data2):
        data2 = data2[-len(data1):]
    elif len(data2) < len(data1):
        data1 = data1[-len(data2):]
    return np.cov(data1,data2)[1,0]

def solveOptimalRisk(covmat, rets, retBar):
    def f(xbar):
        return np.matmul(np.matmul(xbar, covmat), xbar)

    # Sum to 1 constraint
    conSum = {'type': 'eq',
        'fun': lambda x: np.sum(x)-1
    }
    
    # Positive constraint
    conPos = {
        'type': 'ineq',
        'fun': lambda x: np.sum(np.where(x >= 0, 0, x))
    }

    # Expected return constraint
    conRet = {
        'type': 'eq',
        'fun': lambda x: np.matmul(x.T, rets)[0] - retBar
    }

    x0 = np.ones(rets.shape[0]) / rets.shape[0]

    res = scipy.optimize.minimize(f, x0, method='SLSQP', constraints=[conSum, conRet])
    return res


def plotData(variance, returns, tickers):
    plt.plot(variance, returns, '+')
    plt.xlabel('Variance')
    plt.ylabel('Returns')
    plt.title('Avg daily returns vs Variance')
    for var, ret, ticker in zip(variance, returns, tickers):
        plt.annotate(ticker, (var, ret))
    plt.show()

if __name__ == '__main__':
    # Load covariance, returns, tickers
    covMat = np.load('covmat.npy')
    returns = np.load('returns.npy')
    tickers = loadTickers()
    variance = np.diag(covMat)
    #plotData(variance, returns, tickers)

    res = solveOptimalRisk(covMat, returns, np.mean(returns) * 1.18)
    print('Res: ', res)