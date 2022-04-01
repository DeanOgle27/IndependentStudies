import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pandas
import yfinance as yf
from collections import Counter

# Minimum number of samples in ticker data for test-train split
#   Desired: test last 2 years, train prior 3 years; 5 years total
f = open('pcaRes.csv', 'w')

def runPCA(data, pcadim, printStats=False, printCorrels=False):
    pca = PCA(n_components=pcadim)
    reduced = pca.fit_transform(data)  # 1259 x pcadim
    print(reduced.shape)

    reconstructed = pca.inverse_transform(reduced)  # 1259 x 74

    correls = []
    for i in range(reconstructed.shape[1]):
        correlation = np.corrcoef(reconstructed[:, i], data[:, i])[
            0, 1]  # Also works with [1,0]
        if(printCorrels == True):  # Doing this instead of if(printCorrels) because I have PTSD from JavaScript
            print(f'Ticker: {i} Correl: {correlation:.3f}')
        correls.append(correlation)
    correls = np.array(correls)
    if(printStats == True):
        print('Correlation Stats: ')
        print('\tMean Correl:   ', np.mean(correls))
        print('\tMedian Correl: ', np.median(correls))
        print('\tMin Correl:    ', np.min(correls))
        print('\tMax Correl:    ', np.max(correls))
        print('\tStdv Correl:   ', np.std(correls))
    return pca, correls


def plotPCAAnalysis(xs, y1s, y2s, y3s, y1slabel, y2slabel, y3slabel, title=''):
    # plt.figure()
    plt.plot(xs, y1s, label=y1slabel)
    plt.plot(xs, y2s, label=y2slabel)
    plt.plot(xs, y3s, label=y3slabel)
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Original data correlation with reconstructed data')
    plt.legend()
    plt.title(title)


def runPCAAnalysis(data):
    dims = []
    mins = []
    maxs = []
    medians = []

    for dim in range(data.shape[1]):
        _, correls = runPCA(data, dim)
        mins.append(np.min(correls))
        maxs.append(np.max(correls))
        medians.append(np.median(correls))
        dims.append(dim)
        print('Analysis done with dim: ', dim)
    plotPCAAnalysis(dims, mins, maxs, medians,
                    'Auto Min Cors', 'Auto Max Cors', 'Auto Med Cors')


def runPCAAnalysisTestTrainSplit(data, splitIndex):
    dims = []
    mins = []
    maxs = []
    medians = []

    tests = data[-splitIndex:]
    trains = data[0:-splitIndex]

    for dim in range(data.shape[1]):
        pca, _ = runPCA(trains, dim)
        testTf = pca.transform(tests)
        testRecon = pca.inverse_transform(testTf)
        correls = []
        for i in range(testRecon.shape[1]):
            correlation = np.corrcoef(testRecon[:, i], tests[:, i])[
                0, 1]  # Also works with [1,0]
            correls.append(correlation)
        mins.append(np.min(correls))
        maxs.append(np.max(correls))
        medians.append(np.median(correls))
        dims.append(dim)
        print('Analysis done with dim: ', dim)
    plotPCAAnalysis(dims, mins, maxs, medians,
                    'Split Min Cors', 'Split Max Cors', 'Split Med Cors', title='Reconstructed PCA Correlation (Test/Train Split) vs Dim')

def saveTickerCloses(ticker):
    histData = yf.Ticker(ticker).history(period='5y', interval='1d')
    if('Adj Close' in histData.columns):
        print('FOUND ADJ CLOSE: ', ticker)
    for col in histData.columns:
        if col != 'Close' and col != 'Adj Close':
            histData = histData.drop(columns=col)
    histData['Closem1'] = histData['Close'].shift(1)
    histData['Returns'] = histData['Close'] / histData['Closem1'] - 1
    histData = histData.drop(columns='Closem1')
    histData.to_csv(f'./historyDataNew/{ticker}')

def loadHistoryData(ticker):
    f = open(f'historyDataNew/{ticker}')
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
            print('LINE: ', line)
            print('TICKER: ', ticker)

    f.close()
    return dates, prices, returns

def loadStockInfo():
    f = open('TickerInfo.txt')
    dataObj = {}
    for line in f.readlines():
        info = line.split(',')
        dates, prices, returns = loadHistoryData(info[0])
        dataObj[info[0]] = {
            'sector': info[1],
            'industry': info[2],
            'marketCap': float(info[3]),
            'dividendYield': float(info[4]),
            'ticker': info[0],
            'prices': np.array(prices),
            'returns': np.array(returns)
        }
    f.close()
    return dataObj

def getParamFromObj(param, dataObj):
    paramList = []
    for key in dataObj:
        paramList.append(dataObj[key][param])
    return paramList


def marketCapHistogram(dataObj):
    marketCaps = getParamFromObj('marketCap', dataObj)
    marketCaps.sort()
    nbins = 30
    print('33: ', np.percentile(marketCaps, 33))
    print('67: ', np.percentile(marketCaps, 67))
    plt.figure()
    plt.hist(marketCaps[:40], nbins)


def divYieldHist(dataObj):
    divs = getParamFromObj('dividendYield', dataObj)
    divs.sort()
    startIdx = 0
    while divs[startIdx] == 0:
        startIdx += 1
    divs = divs[startIdx:]
    print(len(divs))
    print(np.median(divs))
    nbins = 30
    plt.figure()
    plt.hist(divs, nbins)

def getReturnStructure(dataObj):
    returnsData = None
    for key in dataObj:
        if returnsData is None:
            returnsData = dataObj[key]['returns']
        else:
            returnsData = np.vstack((returnsData, dataObj[key]['returns']))
    return returnsData.T

def getReturnsByDividends(dataObj, cutoffs):
    numSeries = [None] * (len(cutoffs)+1)
    for key in dataObj:
        div = dataObj[key]['dividendYield']
        idx = 0
        while idx < len(cutoffs) and div > cutoffs[idx]:
            idx += 1
        if(numSeries[idx] is None):
            numSeries[idx] = dataObj[key]['returns']
        else:
            numSeries[idx] = np.vstack((numSeries[idx], dataObj[key]['returns']))
    divsDict = {}
    for i in range(len(numSeries)):
        label = f'Divs {i}'
        if i == 0:
            label = f'Divs <= {cutoffs[0]}'
        elif i < len(cutoffs):
            label = f'{cutoffs[i-1]} < Divs <= {cutoffs[i]}'
        else:
            label = f'{cutoffs[-1]} < Divs'
        print(label, ': ', numSeries[i].shape)
        divsDict[label] = np.mean(numSeries[i], axis=0)
    return divsDict

def getReturnsByMarketCap(dataObj, cutoffs):
    numSeries = [None] * (len(cutoffs)+1)
    for key in dataObj:
        mCap = dataObj[key]['marketCap']
        idx = 0
        while idx < len(cutoffs) and mCap > cutoffs[idx]:
            idx += 1
        if(numSeries[idx] is None):
            numSeries[idx] = dataObj[key]['returns']
        else:
            numSeries[idx] = np.vstack((numSeries[idx], dataObj[key]['returns']))
    mCapDict = {}
    for i in range(len(numSeries)):
        label = f'{i}'
        if i == 0:
            label = f'Market Cap <= {cutoffs[0]}'
        elif i < len(cutoffs):
            label = f'{cutoffs[i-1]} < Market Cap <= {cutoffs[i]}'
        else:
            label = f'{cutoffs[-1]} < Market Cap'
        print(label, ': ', numSeries[i].shape)
        mCapDict[label] = np.mean(numSeries[i], axis=0)
    return mCapDict

def getReturnsByIndustry(dataObj):
    indsSeries = {}
    for key in dataObj:
        industry = dataObj[key]['industry']
        if industry in indsSeries:
            indsSeries[industry] = np.vstack((indsSeries[industry], dataObj[key]['returns']))
        else:
            indsSeries[industry] = dataObj[key]['returns']
    for key in indsSeries:
        print(key, ': ', indsSeries[key].shape)
        if(len(indsSeries[key].shape) == 2):
            indsSeries[key] = np.mean(indsSeries[key], axis=0)
    return indsSeries

def plotReturnsByIndustry(indsObj):
    for key in indsObj:
        print('Key: ', key, ' Shape: ', indsObj[key].shape)
        plt.plot(indsObj[key], label=key)
    plt.legend()

def getETFs(ETFS):
    dataDict = {}
    for ETF in ETFS:
        dates, prices, returns = loadHistoryData(ETF)
        dataDict[ETF] = np.array(returns)
    return dataDict

def getSummary(indicatorsDict, timeSeriesData):
    dim = 10
    pca, _ = runPCA(timeSeriesData, dim)
    testTf = pca.transform(timeSeriesData)
    infoDict = {}
    correlMat = None
    for key in indicatorsDict:
        indicatorData = indicatorsDict[key]
        correls = []
        for i in range(dim):
            dimData = testTf[:,i]
            correls.append(np.corrcoef(dimData, indicatorsDict[key])[0][1])
        infoDict[key] = correls
        if correlMat is None:
            correlMat = correls
        else:
            correlMat = np.vstack((correlMat, correls))
        print('\n\nKey: ', key, ' Correls: ', ','.join(f'{correl:.4f}' for correl in correls))
        f.write(f'{key}\n{",".join(f"{correl:.4f}" for correl in correls)}\n')
    return np.mean(correlMat, axis=0)

def getCrossCorrels(indicatorsDict):
    numIndicators = len(indicatorsDict.keys())
    keys = list(indicatorsDict.keys())
    CCs = np.zeros((numIndicators, numIndicators))
    for i in range(numIndicators):
        for j in range(i, numIndicators):
            correl = np.corrcoef(indicatorsDict[keys[i]], indicatorsDict[keys[j]])[0][1]
            CCs[i][j] = correl
            CCs[j][i] = correl
    return CCs



if __name__ == '__main__':
    dataObj = loadStockInfo()
    retData = getReturnStructure(dataObj)

    marketCapHistogram(dataObj)
    divYieldHist(dataObj)

    indicatorsDict = {}
    retIndustries = getReturnsByIndustry(dataObj)
    indicatorsDict.update(retIndustries)

    retsByCap = getReturnsByMarketCap(dataObj, [22764591452.16, 59327422218.240005])
    indicatorsDict.update(retsByCap)
    
    retsByDiv = getReturnsByDividends(dataObj, [0, 0.01452329575])
    indicatorsDict.update(retsByDiv)
    
    retsByEtf = getETFs(['SPY', 'QQQ'])
    indicatorsDict.update(retsByEtf)

    indicatorsDict.update({
        'Average': np.mean(retData, axis=1)
    })

    correlAvgs = getSummary(indicatorsDict, retData)
    #print("Correl Avgs: ", correlAvgs)

    ccs = getCrossCorrels(indicatorsDict)
    #print('CCS: ', ccs)
    f.close()
    #plt.show()
