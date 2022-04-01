import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import time
import methods
#import Markowitz.methods as methods
import matplotlib.cm as cm



# Done for return object in solveOptimalRisk
class Object(object):
    pass
FPT_ERR_TOL = 10**-12

# <Helper and Debug Functions>
def printPortfolio(tickers, weights):
    pass

# </Helper and Debug Functions>
def loadTickers():
    f = open('./tickerList.txt', 'r')
    tickers = []
    for line in f.readlines():
        tickers.append(line.replace('\n', ''))
    f.close()
    return tickers
def plotData(variance, returns, tickers):
    plt.plot(variance, returns, '+')
    plt.xlabel('Variance')
    plt.ylabel('Returns')
    plt.title('Avg daily returns vs Variance')
    for var, ret, ticker in zip(variance, returns, tickers):
        plt.annotate(ticker, (var, ret))
    plt.show()

def dataListToXsYsFaileds(dataList):
    xs = []
    ys = []
    failedXs = []
    failedYs = []
    for i in range(len(dataList)):
        dataObj = dataList[i]
        if(dataObj['success']):
            xs.append(dataObj['risk'])
            ys.append(dataObj['ret'])
        else:
            failedXs.append(dataObj['risk'])
            failedYs.append(dataObj['ret'])
    return xs, ys, failedXs, failedYs

def colorMapper():
    pass
def plotOptimalCurve(dataList, variance, returns, tickers, title=None, saveFig=False, saveFigName=None):
    xs, ys, failedXs, failedYs = dataListToXsYsFaileds(dataList)

    plt.figure()
    plt.plot(variance, returns, '+', label='Tickers')
    plt.xlabel('Variance')
    plt.ylabel('Returns')
    if title is None:
        plt.title('Avg daily returns vs Variance')
    else:
        plt.title(title)
    if((failedXs is not None) and (failedYs is not None)):
        plt.plot(failedXs, failedYs, 'r.', label='Failed Points')
    for var, ret, ticker in zip(variance, returns, tickers):
        plt.annotate(ticker, (var, ret))
    plt.plot(xs, ys, 'g.', label='Solved Points')
    plt.legend()
    if saveFig:
        #https://stackoverflow.com/questions/10041627/how-to-make-savefig-save-image-for-maximized-window-instead-of-default-size
        figure = plt.gcf() # get current figure
        figure.set_size_inches(16, 8)
        plt.xlim((0,0.003))
        plt.ylim((0.0002,0.003))
        if saveFigName is None:
            plt.savefig('newPlotImages/res.png', dpi=200)
        else:
            plt.savefig(f'newPlotImages/{saveFigName}', dpi=200)
    else:
        pass
        #plt.show()


def plotOptimalCurves(dataObjs, variance, returns, tickers, title=None):
    plt.figure()
    nPts = len(variance)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=nPts, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Greys_r)

    # To get color: mapper.to_rgba(v

    for dataObj in dataObjs:
        label = dataObj['label']
        dataList = dataObj['dataList']
        xs, ys, failedXs, failedYs = dataListToXsYsFaileds(dataList)
        if title is None:
            plt.title('Avg daily returns vs Variance')
        else:
            plt.title(title)
        if((failedXs is not None) and (failedYs is not None) and len(failedXs) > 0):
            plt.plot(failedXs, failedYs, dataObj['marker'], label=f'Failed Points {label}')
        if('marker' in dataObj):
            plt.plot(xs, ys, dataObj['marker'], label=f'Frontier: {label}')
        else:
            plt.plot(xs, ys, '.', label=f'Frontier: {label}')


    plt.plot(variance, returns, '+', label='Tickers')
    plt.xlabel('Variance')
    plt.ylabel('Returns')
    for var, ret, ticker in zip(variance, returns, tickers):
        plt.annotate(ticker, (var, ret))
    plt.legend()

def plotMonteCarlo():
    data = np.load('frontier_5.npy')
    xs5, ys5 = data[:,0], data[:,1]
    plt.plot(xs5, ys5, label='Monte Carlo N=5')
    plt.legend()
    
def runSimWithNStocks(nStocks, method='slsqp'):
    print('Running Sim With N Stocks: ', nStocks)
    covMat = np.load('covmat.npy')
    returns = np.load('returns.npy')
    tickers = loadTickers()
    variance = np.diag(covMat)

    reducedCovMat = covMat[0:nStocks, 0:nStocks]
    reducedReturns = returns[0:nStocks]
    reducedStdevs = np.sqrt(np.diag(reducedCovMat))

    cvxDataList = methods.get_cvx_comparison_objects(reducedCovMat, reducedReturns)
    optimalCurves = [
        {
            'label': f'CVX frontier',
            'dataList': cvxDataList,
            'marker': '+'
        }
    ]

    rets = np.linspace(np.min(returns), np.max(returns), 200)
    gamma = 0.02

    markers = ['*', '+', 'D', '.']
    for i in range(4):
        canShort = (i >= 2)
        canCash = ((i % 2) == 0)
        dataList = methods.getCurveCvx(rets, hasCash=canCash, shortsAllowed=canShort, gamma=gamma)
        optimalCurves.append({
            'label': f'Shorts: {canShort}, Hold Cash: {canCash}',
            'dataList': dataList,
            'marker': markers[i]
        })

    plotOptimalCurves(optimalCurves, variance, returns, tickers)
    plotMonteCarlo()
    plt.show()


def testSim():
    runSimWithNStocks(75, method='slsqp')
    


if __name__ == '__main__':
    testSim()



    # for i in range(15):
    #     runSimWithNStocks(5*(i+1), method='slsqp')



    # First: Auto-encoder and Z's
    # Classical is using models (fama-french for example), drawback is linearity
    # (Decent locally, but not for large deviations)
    # Compare to standard factor model (as in deep portfolio paper)
    # Predictions could be more accurate using autoencoder
