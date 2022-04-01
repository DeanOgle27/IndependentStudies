import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import time
from Markowitz.methods import get_cvx_opt_fixedRet
# Done for return object in solveOptimalRisk
class Object(object):
    pass
FPT_ERR_TOL = 10**-12

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
    histData = yf.Ticker(ticker).history(period='max', interval='1d')
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
    return np.corrcoef(data1, data2)[1, 0]


def covariance(data1, data2):
    if len(data1) < len(data2):
        data2 = data2[-len(data1):]
    elif len(data2) < len(data1):
        data1 = data1[-len(data2):]
    return np.cov(data1, data2)[1, 0]


def getX0(rets, retBar):
    maxRet = np.max(rets)
    minRet = np.min(rets)
    maxWeight = (retBar - minRet) / (maxRet - minRet)
    minWeight = (maxRet - retBar) / (maxRet - minRet)
    weights_init = np.zeros(rets.shape[0])
    weights_init[np.argmax(rets)] = maxWeight
    weights_init[np.argmin(rets)] = minWeight
    return weights_init

def testGetX0Smart(returns):
    returns = returns[:,0]
    numPts = 30
    for i in np.linspace(np.min(returns), np.max(returns), numPts):
        x0Weights = getX0Smart(returns, i)
        print(x0Weights.shape)
        print(returns.shape)
        if(np.abs(i-np.matmul(returns, x0Weights)) > 10**-6):
            print(f"FAILED; RetBar: {i}, Smart Weights Ret: {np.matmul(returns, x0Weights)}")
        else:
            print(f"SUCCES; RetBar: {i}, Smart Weights Ret: {np.matmul(returns, x0Weights)}")


# TODO: Test out running onconstrained optimization with a quadratic sum(weights) penalty, then
#       scale the unconstrained weigts to sum to 1 or 1-tolerance with a return of ret +/- tolerance
def getX0Smart(rets, retBar, Debug=False):
    rets = rets[:,0]
    weights = np.ones(rets.shape[0]) / rets.shape[0]
    sortedRets = np.sort(rets)

    # In case where desired ret is ret bar, returns max or min weighted at 1
    if(retBar == np.max(rets)):
        return getX0(rets, retBar)
    if(retBar == np.min(rets)):
        return getX0(rets, retBar)

    # Uniform is less than retBar
    if np.matmul(weights, sortedRets) < retBar:
        splitIndex = np.argmax(sortedRets >= retBar)
        counter = 0
        curRet = np.matmul(weights, sortedRets)
        while (curRet + FPT_ERR_TOL) < retBar:
            delta = retBar - curRet
            addCounter = (len(weights)-1) - (counter %(len(weights)-splitIndex))
            maxDiff = (sortedRets[addCounter] - sortedRets[counter]) * weights[counter]
            if(maxDiff < delta):  # Transfer weight from counter to addCounter
                weights[addCounter] = weights[addCounter] + weights[counter]
                weights[counter] = 0 
            else:
                weightShift = delta / (sortedRets[addCounter] - sortedRets[counter])
                weights[addCounter] = weights[addCounter] + weightShift
                weights[counter] = weights[counter] - weightShift
            curRet = np.matmul(weights, sortedRets)
            if Debug: print(f'AC: {addCounter}, C: {counter}, RB: {retBar*10000}, RET: {curRet*10000}, MD: {maxDiff*10000}, Delt: {delta*10000}, Weights: {weights}')
            counter += 1

    # Uniform is more than retBar
    elif np.matmul(weights, sortedRets) > retBar:
        splitIndex = np.argmax(sortedRets >= retBar)
        counter = 0
        curRet = np.matmul(weights, sortedRets)
        while curRet > (retBar + FPT_ERR_TOL):
            delta = curRet - retBar
            addCounter = counter % splitIndex
            removeCounter = (len(weights) - 1) - counter
            maxDiff = (sortedRets[removeCounter] - sortedRets[addCounter]) * weights[removeCounter]
            if(maxDiff < delta):  # Transfer weight from counter to addCounter
                weights[addCounter] = weights[addCounter] + weights[removeCounter]
                weights[removeCounter] = 0 
            else:
                weightShift = delta / (sortedRets[removeCounter] - sortedRets[addCounter])
                weights[addCounter] = weights[addCounter] + weightShift
                weights[removeCounter] = weights[removeCounter] - weightShift
            curRet = np.matmul(weights, sortedRets)
            if Debug: print(f'AC: {addCounter}, RC: {removeCounter}, RET: {curRet:.3f}, MD: {maxDiff:.3f}, Delt: {delta:.3f}, Weights: {weights}')
            counter += 1

    else:
        return weights

    # Remap sorted rets' weights to unsorted rets' weights
    unsortedWeights = np.zeros(rets.shape[0])
    for i in range(len(rets)):
        unsortedWeights[i] = weights[np.argwhere(sortedRets==rets[i])[0][0]]
    return unsortedWeights


def solveOptimalRisk(covmat, rets, retBar, method='slsqp', startingPoint=None):
    def f(xbar):
        return np.matmul(np.matmul(xbar, covmat), xbar)
    def jac(xbar):
        return 2*np.matmul(xbar, covmat)
    def hess(xbar):
        return 2*covmat

    weights_init = None
    if type(startingPoint) == type(None):
        weights_init = getX0Smart(rets, retBar)
    else:
        print('USING STARTING POINT')
        weights_init = startingPoint

    # Handles edge cases
    if(retBar == np.min(rets) or retBar == np.max(rets)):
        retObj = Object()
        retObj.success = True
        retObj.fun = f(weights_init)
        return retObj

    # No shorting allowed, no position size greater than portfolio
    bounds = scipy.optimize.Bounds([0] * len(rets), [1] * len(rets))

    # SLSQP method
    if(method == 'slsqp' or method == 'SLSQP'):
        # Sum to 1 constraint
        sumToOne = {'type': 'eq',
                'fun': lambda x: 1-np.sum(x),
                'jac': lambda x: -1*np.ones(len(x))
                }

        # Expected return constraint
        sufficientReturn = {
            'type': 'eq',
            'fun': lambda x: np.matmul(x.T, rets)[0] - retBar,
            'jac': lambda x: rets[:,0]
        }

        #https://docs.scipy.org/doc/scipy/tutorial/optimize.html#sequential-least-squares-programming-slsqp-algorithm-method-slsqp
        result = scipy.optimize.minimize(f, weights_init, constraints=[sumToOne, sufficientReturn], jac=jac, method='slsqp', bounds=bounds)
        return result

    # Trust-constr method
    elif(method == 'trust-constr' or method=='TRUST-CONSTR' or method=='TRUSTCONSTR' or method=='trustconstr'):
        linearSumToOne = scipy.optimize.LinearConstraint([1]*len(rets),1,1) # Sum of positions has to be 1
        linearReturns = scipy.optimize.LinearConstraint(rets[:,0],retBar,np.inf) # Return must be greater than the retBar

        #https://docs.scipy.org/doc/scipy/tutorial/optimize.html#trust-region-constrained-algorithm-method-trust-constr
        result = scipy.optimize.minimize(f, weights_init, method='trust-constr', jac=jac, hess=hess,constraints=[linearSumToOne, linearReturns], bounds=bounds)
        return result

    elif(method == 'cvx' or method == 'CVX'):
        x = np.array(get_cvx_opt_fixedRet(covmat, rets, retBar))[:,0]
        print('X: ', x.shape)
        retObj = Object()
        retObj.success = True
        retObj.fun = f(x)
        return retObj
    else:
        print('ERROR: solveOptimalRisk: UNSUPPORTED METHOD ', method)

def plotData(variance, returns, tickers):
    plt.plot(variance, returns, '+')
    plt.xlabel('Variance')
    plt.ylabel('Returns')
    plt.title('Avg daily returns vs Variance')
    for var, ret, ticker in zip(variance, returns, tickers):
        plt.annotate(ticker, (var, ret))
    plt.show()


def getCurve(covMat, returns, method='slsqp'):
    xs = []
    ys = []
    failedXs = []
    failedYs = []
    counter = 0
    numPts = 200
    startingPoint = None
    for i in np.linspace(np.min(returns), np.max(returns), numPts):
        res = solveOptimalRisk(covMat, returns, i, method=method, startingPoint=startingPoint)
        if(hasattr(res, 'x')):
            startingPoint=res.x
        else:
            startingPoint=None
        counter += 1
        if(res.success):
            xs.append(res.fun)
            ys.append(i)
        else:
            failedXs.append(res.fun)
            failedYs.append(i)
        print(f'{100 * counter / numPts : .2f}%; Success: {res.success} done')
    return xs, ys, failedXs, failedYs


def plotOptimalCurve(xs, ys, variance, returns, tickers, failedXs=None, failedYs=None, title=None, saveFig=False, saveFigName=None):
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

    xs, ys, failedXs, failedYs = getCurve(reducedCovMat, reducedReturns, method=method)
    plotOptimalCurve(xs, ys, variance[0:nStocks], returns[0:nStocks], tickers[0:nStocks], failedXs=failedXs, failedYs=failedYs, title=f'Num Stocks: {nStocks}, Method: {method}', saveFig=False, saveFigName=f'{method}_{nStocks}.png')
    plotMonteCarlo()
    plt.show()

if __name__ == '__main__':

    nStocks = 75
    # Load covariance, returns, tickers
    covMat = np.load('covmat.npy')
    returns = np.load('returns.npy')
    tickers = loadTickers()
    variance = np.diag(covMat)

    runSimWithNStocks(75, method='slsqp')
    # for i in range(15):
    #     runSimWithNStocks(5*(i+1), method='slsqp')



    # First: Auto-encoder and Z's
    # Classical is using models (fama-french for example), drawback is linearity
    # (Decent locally, but not for large deviations)
    # Compare to standard factor model (as in deep portfolio paper)
    # Predictions could be more accurate using autoencoder
