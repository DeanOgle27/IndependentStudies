import yfinance as yf
import numpy as np
import os 

def loadTickers():
    f = open('./tickerList.txt', 'r')
    tickers = []
    for line in f.readlines():
        tickers.append(line.replace('\n', ''))
    f.close()
    return tickers

# Save date, close, return for ticker
def saveTickerCloses(ticker, show=False, dropRows=0):
    histData = yf.Ticker(ticker).history(period='max', interval='1d')
    if('Adj Close' in histData.columns):
        print('FOUND ADJ CLOSE: ', ticker)
    for col in histData.columns:
        if col != 'Close' and col != 'Adj Close':
            histData = histData.drop(columns=col)
    histData['Closem1'] = histData['Close'].shift(1)
    histData['Returns'] = histData['Close'] / histData['Closem1'] - 1
    histData = histData.drop(columns='Closem1')
    if dropRows > 0:
        histData.drop(histData.tail(dropRows).index,inplace=True)
    histData.to_csv(f'./historyDataRaw/{ticker}')

def loadTickerData(ticker):
    f = open(f'historyDataRaw/{ticker}')
    dates = []
    prices = []
    returns = []
    for line in f.readlines()[2:]:
        try:
            date, price, returnVal = line.split(',')
            dates.append(date)
            prices.append(float(price))
            returns.append(float(returnVal))
        except:
            #print(f'{ticker} error; line: ', line)
            pass
    f.close()
    return dates, prices, returns

def getRetBar(data):
    return np.array([np.mean(data, axis=0)]).T

def getCovMat(data):
    nAssets = data.shape[1]
    covMat = np.zeros((nAssets,nAssets))
    for i in range(nAssets):
        for j in range(nAssets):
            row1 = data[:,i]
            row2 = data[:,j]
            covVal = np.cov(row1, row2)[1, 0]
            covMat[i][j] = covVal
    return covMat

    
def splitHistoricalData(tickers):
    # Initialize variables
    prior5Prices = None
    prior5Rets = None
    last5Price = None
    last5Rets = None

    prior5testPrices = None
    prior5testRets = None
    last5testPrice = None
    last5testRets = None

    prior5trainPrices = None
    prior5trainRets = None
    last5trainPrice = None
    last5trainRets = None

    prior5Tickers = []
    last5Tickers = []

    # Iterate over tickers
    for ticker in tickers:
        dates, prices, returns = loadTickerData(ticker)
        numPts = len(dates)

        if numPts > 2517 and dates[-2517] == '2012-03-30': # Ten years
            print(f'{ticker}\t{dates[-2517]}\t{dates[-1764]}\t{dates[-1259]}\t{dates[-504]}')
            p5prices = np.array([prices[-2517:-1259]]).T
            p5rets = np.array([returns[-2517:-1259]]).T
            l5prices = np.array([prices[-1259:]]).T
            l5rets = np.array([returns[-1259:]]).T

            p5testprices = np.array([prices[-1764:-1259]]).T
            p5testrets = np.array([returns[-1764:-1259]]).T
            l5testprices = np.array([prices[-504:]]).T
            l5testrets = np.array([returns[-504:]]).T

            p5trainprices = np.array([prices[-2517:-1764]]).T
            p5trainrets = np.array([returns[-2517:-1764]]).T
            l5trainprices = np.array([prices[-1259:-504]]).T
            l5trainrets = np.array([returns[-1259:-504]]).T

            # Make Ticker list
            prior5Tickers.append(ticker)
            last5Tickers.append(ticker)

            # For first iteration
            if(prior5Prices is None):
                prior5Prices = p5prices
                prior5Rets = p5rets
                last5Price = l5prices
                last5Rets = l5rets

                prior5testPrices = p5testprices
                prior5testRets = p5testrets
                last5testPrice = l5testprices
                last5testRets = l5testrets

                prior5trainPrices = p5trainprices
                prior5trainRets = p5trainrets
                last5trainPrice = l5trainprices
                last5trainRets = l5trainrets
            else:
                prior5Prices = np.hstack((prior5Prices, p5prices))
                prior5Rets = np.hstack((prior5Rets, p5rets))
                last5Price = np.hstack((last5Price, l5prices))
                last5Rets = np.hstack((last5Rets, l5rets))

                prior5testPrices = np.hstack((prior5testPrices, p5testprices))
                prior5testRets = np.hstack((prior5testRets, p5testrets))
                last5testPrice = np.hstack((last5testPrice, l5testprices))
                last5testRets = np.hstack((last5testRets, l5testrets))

                prior5trainPrices = np.hstack((prior5trainPrices, p5trainprices))
                prior5trainRets = np.hstack((prior5trainRets, p5trainrets))
                last5trainPrice = np.hstack((last5trainPrice, l5trainprices))
                last5trainRets = np.hstack((last5trainRets, l5trainrets))

        elif numPts > 1259 and dates[-1259] == '2017-03-31': # Five years
            print(f'{ticker}\t{"    NA    "}\t{dates[-1259]}\t{dates[-504]}')
            l5prices = np.array([prices[-1259:]]).T
            l5rets = np.array([returns[-1259:]]).T
            l5testprices = np.array([prices[-504:]]).T
            l5testrets = np.array([returns[-504:]]).T
            l5trainprices = np.array([prices[-1259:-504]]).T
            l5trainrets = np.array([returns[-1259:-504]]).T

            last5Price = np.hstack((last5Price, l5prices))
            last5Rets = np.hstack((last5Rets, l5rets))
            last5testPrice = np.hstack((last5testPrice, l5testprices))
            last5testRets = np.hstack((last5testRets, l5testrets))
            last5trainPrice = np.hstack((last5trainPrice, l5trainprices))
            last5trainRets = np.hstack((last5trainRets, l5trainrets))

            prior5Tickers.append(ticker)
            last5Tickers.append(ticker)

        else:
            print(f'{ticker}\tOmitted')
    return
    # Save them here
    f = open('prior5Tickers.txt', 'w')
    for p5Ticker in prior5Tickers:
        f.write(f'{p5Ticker}\n')
    f.close()

    f = open('last5Tickers.txt', 'w')
    for l5Ticker in last5Tickers:
        f.write(f'{l5Ticker}\n')
    f.close()

    np.save('prior5Prices', prior5Prices)
    np.save('prior5Returns', prior5Rets)
    np.save('last5Prices', last5Price)
    np.save('last5rets', last5Rets)



    np.save('prior5testPrices', prior5testPrices)
    np.save('prior5testReturns', prior5testRets)
    np.save('last5testPrices', last5testPrice)
    np.save('last5testrets', last5testRets)

    np.save('prior5trainPrices', prior5trainPrices)
    np.save('prior5trainReturns', prior5trainRets)
    np.save('last5trainPrices', last5trainPrice)
    np.save('last5trainrets', last5trainRets)

    # Get and save covariances and retbars
    np.save('covp5rets', getCovMat(prior5Rets))
    np.save('covl5rets', getCovMat(last5Rets))
    np.save('covp5test', getCovMat(prior5testRets))
    np.save('covp5train', getCovMat(prior5trainRets))
    np.save('covl5test', getCovMat(last5testRets))
    np.save('covl5train', getCovMat(last5trainRets))

    np.save('retsp5rets', getRetBar(prior5Rets))
    np.save('retsl5rets', getRetBar(last5Rets))
    np.save('retsp5test', getRetBar(prior5testRets))
    np.save('retsp5train', getRetBar(prior5trainRets))
    np.save('retsl5test', getRetBar(last5testRets))
    np.save('retsl5train', getRetBar(last5trainRets))

def saveETFReturns(ETFS):
    for ticker in ETFS:
        dates, prices, returns = loadTickerData(ticker)
        numPts = len(dates)

        if numPts > 2517 and dates[-2517] == '2012-03-30': # Ten years
            print(f'{ticker}\t{dates[-2517]}\t{dates[-1764]}\t{dates[-1259]}\t{dates[-504]}')
            p5prices = np.array([prices[-2517:-1259]]).T
            l5prices = np.array([prices[-1259:]]).T
            p5testprices = np.array([prices[-1764:-1259]]).T
            l5testprices = np.array([prices[-504:]]).T
            p5trainprices = np.array([prices[-2517:-1764]]).T
            l5trainprices = np.array([prices[-1259:-504]]).T
            np.save(f'{ticker}_p5', p5prices)
            np.save(f'{ticker}_l5', l5prices)
            np.save(f'{ticker}_p5test', p5testprices)
            np.save(f'{ticker}_l5test', l5testprices)
            np.save(f'{ticker}_p5train', p5trainprices)
            np.save(f'{ticker}_l5train', l5trainprices)
        else:
            print(f'{ticker}\tOmitted')

def loadAllDatas(printInfo=True):
    dataObj = {}
    for fname in os.listdir():
        if(fname[-3:] == 'npy'):
            dataObj[fname.split('.')[0]] = np.load(fname, allow_pickle=True)
            if printInfo:
                print(f'File: {fname}, Shape: {dataObj[fname.split(".")[0]].shape}')
    return dataObj

if __name__ == '__main__':
    # saveTickerCloses('SPY')
    # saveTickerCloses('QQQ')

    #tickers = loadTickers()

    #splitHistoricalData(tickers)
    # loadAllDatas()

    saveETFReturns(['SPY', 'QQQ'])
