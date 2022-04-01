import Methods
import numpy as np
import matplotlib.pyplot as plt

NUM_PTS = 200
'''
    Fit markowitz curve on prior5
    Fit markowitz curve on trainp5 and trainl5 with graph
    Evaluate markowitz curve on testp5 and testl5 with graph
'''
def loadAllTickers():
    f = open('./last5Tickers.txt', 'r')
    l5Tickers = []
    for line in f.readlines():
       l5Tickers.append(line.replace('\n', ''))
    f.close()
    f = open('./prior5Tickers.txt', 'r')
    p5Tickers = []
    for line in f.readlines():
        p5Tickers.append(line.replace('\n', ''))
    f.close()
    return l5Tickers, p5Tickers

def getTimeSeriesData(weights, testData):
    #Bit of a misnomer, its just so that the prices all start at 1
    normalizedTestData = testData / testData[0]
    return np.matmul(normalizedTestData, weights) + (1 - np.sum(weights))

def getMinimumRiskIndex(dataList):
    index = 0
    minRiskVal = 1 # As far as we're concerned, this is essentially infinity
    for i in range(len(dataList)):
        if dataList[i]['risk'] < minRiskVal:
            minRiskVal = dataList[i]['risk']
            index = i
    return index

def runTestTrainSplit(trainCovMat, trainRets, testCovMat, testRets, tickers, testTimeSeries, plot=False, testDateStr='', trainDateStr="", prior5=False):
    canShort = False
    canCash = False
    desiredRets = np.linspace(max(np.min(trainRets), 0), np.max(trainRets), NUM_PTS)

    trainOptimalCurves = []
    minimumReturnIdx = 0
    maximumReturnIdx = np.argmax(desiredRets > np.percentile(trainRets, 90))

    for i in range(4):
        canShort = i >= 2
        canCash = (i % 2 == 1)
        dataListTrain = Methods.getCurveCvx(trainRets, trainCovMat, desiredRets, hasCash=canCash, shortsAllowed=canShort, gamma=0.02)
        trainOptimalCurves.append({
            'label': f'Shorts: {canShort}, Hold Cash: {canCash}',
            'dataList': dataListTrain,
            'marker': '.'
        })
        if((not canShort) and (not canCash)):
            minimumReturnIdx = getMinimumRiskIndex(dataListTrain)
    # Show train data results
    if plot:
        Methods.plotOptimalCurves(trainOptimalCurves, np.diag(trainCovMat), trainRets, tickers, title=f'Trained Curve on Train Data ({trainDateStr})', rainbow=False)

    testOptimalCurves = []
    for curve in trainOptimalCurves:
        testDataList = []
        for point in curve['dataList']:
            x = point['weights']
            testDataList.append({
                'weights': np.array(x),
                'risk': np.matmul(x.T, np.matmul(testCovMat, x))[0][0],
                'success': True,
                'ret': np.matmul(testRets.T, x)[0][0]
            })
        testOptimalCurves.append({
            'label': curve['label'],
            'dataList': testDataList,
            'marker': '-'
        })


    # Plot result on test tata
    if plot:
        Methods.plotOptimalCurves(testOptimalCurves, np.diag(testCovMat), testRets, tickers, title=f'Trained Curve on Test Data ({testDateStr})', rainbow=False)
        Methods.plotOptimalCurves(testOptimalCurves, np.diag(testCovMat), testRets, tickers, title=f'Trained Curve on Test Data ({testDateStr})', rainbow=True)


    # Use train data to get time series of returns
    percentiles = [0, 25, 50, 75]
    overLapPercentile = 25
    overlapPlotDatas = []
    summaryData = []
    ETFS = ['SPY', 'QQQ']
    ETF_STYLES = ['--', ':'] #https://matplotlib.org/2.1.2/api/_as_gen/matplotlib.pyplot.plot.html
    plotColors = ['red', 'cyan', 'green', 'indigo'] #https://matplotlib.org/stable/gallery/color/named_colors.html
    print(f'MINRETI: {minimumReturnIdx}, MAXRETI: {maximumReturnIdx}')
    for curve in trainOptimalCurves:
        plotDatas = []
        for percentile in percentiles:
            weightsIndex = int(minimumReturnIdx + percentile*maximumReturnIdx/100) # TODO: Update this
            weights = curve['dataList'][weightsIndex]['weights']
            tSeriesData = getTimeSeriesData(weights, testTimeSeries)
            plotDatas.append({
                'data': tSeriesData,
                'label': f'{curve["label"]}, Perc: {percentile}',
                'color': plotColors[len(plotDatas)]
            })
            if percentile == overLapPercentile:
                overlapPlotDatas.append({
                    'data': tSeriesData,
                    'label': f'{curve["label"]}, Perc: {percentile}',
                    'color': plotColors[len(overlapPlotDatas)]
                })
        for ETF in ETFS:
            plotDatas.append({
                'data': getNormalizedETFData(ETF, prior5=prior5),
                'label': f'{ETF}',
                'lineStyle': ETF_STYLES[ETFS.index(ETF)]
            })
        plotTimeSeriesData(plotDatas, title=f'Portfolio Value vs Trading Day on {testDateStr}')
        #printTimeSeriesSummary(plotDatas)
        summaryData += getSummaryData(plotDatas, prior5, getNormalizedETFData('SPY', prior5=prior5), getNormalizedETFData('QQQ', prior5=prior5), getETFS=False)
    for ETF in ETFS:
        overlapPlotDatas.append({
            'data': getNormalizedETFData(ETF, prior5=prior5),
            'label': f'{ETF}',
            'lineStyle': ETF_STYLES[ETFS.index(ETF)]
        })
    summaryData = getSummaryData(overlapPlotDatas, prior5, getNormalizedETFData('SPY', prior5=prior5), getNormalizedETFData('QQQ', prior5=prior5)) + summaryData
    plotTimeSeriesData(overlapPlotDatas, title=f'Portfolio Value vs Trading Day on {testDateStr}')
    #printTimeSeriesSummary(overlapPlotDatas)
    saveSummaryData(summaryData, prior5)
    

def saveSummaryData(summaryData, prior5):
    dataSetTag = 'l5'
    if prior5:
        dataSetTag = 'p5'
    f = open(f'summaryData/data_{dataSetTag}.csv', 'w')
    f.write('Cash,Short,Percentile,Variance Percentage,Annual Average Return Percentage,Total Return Percentage,SPY Correlation,QQQ Correlation\n')
    for dat in summaryData:
        f.write(f"{dat['cash'][0]},{dat['short'][0]},{dat['perc']},{dat['var']:.3f},{dat['avgRet']:.1f},{dat['totRet']:.1f},{dat['spycor']:.2f},{dat['qqqcor']:.2f}\n")
    f.close()

def getSummaryData(plotDatas, prior5, SPY, QQQ, getETFS=True):
    dataSetLabel = 'last5'
    if prior5:
        dataSetLabel = 'prior5'
    sumDat = []
    for plotData in plotDatas:
        data = plotData['data']
        label = plotData['label']

        lastPoint = data[-1][0]
        daynP1s = data[:,0][1:]
        dayns = data[:,0][:-1]
        returns = (daynP1s / dayns - 1)
        variancePercent = np.var(returns*100)
        totalReturn = (lastPoint - 1) * 100
        avgAnnualReturn = (lastPoint ** (252/len(data)) - 1) * 100

        infoLabel = f'{label}_{dataSetLabel}' #For QQQ and SPY
        # For not QQQ and SPY
        if(len(label) > 10):
            if not getETFS:
                info = label.replace(':', ',').replace(' ','').split(',')
                # Parse label info to get cash, short, and percentage info
                sumDat.append({
                    'cash': info[3],
                    'short': info[1],
                    'perc': info[-1].replace('Summary',''),
                    'var': variancePercent,
                    'avgRet': avgAnnualReturn,
                    'totRet': totalReturn,
                    'spycor': np.corrcoef(SPY[:,0],data[:,0])[0][1],
                    'qqqcor': np.corrcoef(QQQ[:,0],data[:,0])[0][1]
                })
        elif getETFS:
            sumDat.append({
                'cash': label,
                'short': label,
                'perc': label,
                'var': variancePercent,
                'avgRet': avgAnnualReturn,
                'totRet': totalReturn,
                'spycor': np.corrcoef(SPY[:,0],data[:,0])[0][1],
                'qqqcor': np.corrcoef(QQQ[:,0],data[:,0])[0][1]
            })
    return sumDat

def printTimeSeriesSummary(plotDatas):
    for plotData in plotDatas:
        data = plotData['data']
        label = plotData['label']
        lastPoint = data[-1][0]
        daynP1s = data[:,0][1:]
        dayns = data[:,0][:-1]
        returns = (daynP1s / dayns - 1)
        variancePercent = np.var(returns) *100
        totalReturn = (lastPoint - 1) * 100
        avgAnnualReturn = (lastPoint ** (252/len(data)) - 1) * 100
        print(f'{label} Summary\n\tDaily Return Variance: {variancePercent}%\n\tOverall Return: {totalReturn:.5f}\n\tDays: {len(data)}\n\tAverage annual percent return: {avgAnnualReturn:.5f}%')

def plotTimeSeriesData(plotDatas, title='Portfolio value vs Time'):
    plt.figure()
    plt.title(title)
    for pDat in plotDatas:
        if 'color' in pDat:
            plt.plot(pDat['data'], label=pDat['label'], color=pDat['color'])
        elif 'lineStyle' in pDat:
            plt.plot(pDat['data'], pDat['lineStyle'], label=pDat['label'], color='grey')
        else:
            plt.plot(pDat['data'], label=pDat['label'])
    plt.xlabel('Trading Day')
    plt.ylabel('Day 1-Normalized Portfolio Value')
    plt.legend()

def getNormalizedETFData(ETF, prior5=False, test=True):
    dataSetLabel = 'l5'
    if prior5:
        dataSetLabel = 'p5'
    sublabel = 'test'
    if not test:
        sublabel = 'train'
    data = np.load(f'{ETF}_{dataSetLabel}{sublabel}.npy')
    return data / data[0]

# Plot markowitz curve with colors
def runPriorTestTrainSplit():
    l5Tickers, p5Tickers = loadAllTickers()
    data = Methods.loadAllDatas(printInfo=True)
    covMat = data['covp5rets']
    retVect = data['retsp5rets']
    trainCovMat = data['covp5train']
    trainRets = data['retsp5train']
    testCovMat = data['covp5test']
    testRets = data['retsp5test']
    prior5TestPrices = data['prior5testPrices']

    runTestTrainSplit(trainCovMat, trainRets, testCovMat, testRets, p5Tickers, prior5TestPrices, testDateStr='3/31/2015 – 3/31/2017', trainDateStr="3/30/2012 – 3/31/2015", prior5=True)

def runLatestTestTrainSplit():
    l5Tickers, p5Tickers = loadAllTickers()
    data = Methods.loadAllDatas(printInfo=True)
    covMat = data['covl5rets']
    retVect = data['retsl5rets']
    trainCovMat = data['covl5train']
    trainRets = data['retsl5train']
    testCovMat = data['covl5test']
    testRets = data['retsl5test']
    last5TestPrices = data['last5testPrices']

    runTestTrainSplit(trainCovMat, trainRets, testCovMat, testRets, l5Tickers, last5TestPrices, testDateStr='4/1/2020 – 3/30/2022', trainDateStr="3/31/2017 – 3/31/2020", prior5=False)

def mainMethod():
    runPriorTestTrainSplit()
    runLatestTestTrainSplit()
    #plt.show()

    ### </TEST TRAIN SPLIT ON PRIOR>

    
    #getCurveCvx(retVect, covMat, desiredRets, hasCash=True, shortsAllowed=False, gamma=0.02)

if __name__ == '__main__':
    mainMethod()

    # 
