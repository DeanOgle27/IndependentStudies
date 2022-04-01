import matplotlib.pyplot as plt
import numpy as np
from methods import pltFigure, getAxes
from scipy.spatial import ConvexHull

SIM_POINTS = 1000000
NPTS = [3,5,7,10]
def loadTickers():
    f = open('./tickerList.txt', 'r')
    tickers = []
    for line in f.readlines():
        tickers.append(line.replace('\n', ''))
    f.close()
    return tickers

'''
covMat: NxN covariance matrix
returns: Nx1 returns vector
numStocks: Maximum number of stocks in portfolio
'''
def runMonteCarlo(covMat, returns, numStocks):
    x = np.zeros(returns.shape[0])
    for i in range(numStocks):
        x[np.random.randint(returns.shape[0])] = np.random.random()
    x = x / np.sum(x)
    # Random n-stock 'portfolio'
    ret = np.matmul(x, returns)[0]
    risk = np.matmul(np.matmul(x, covMat), x)
    return ret, risk


def monteCarloFrontier(covMat, returns, numStocks):
    xs = []
    ys = []
    for i in range(SIM_POINTS):
        ret, risk = runMonteCarlo(covMat, returns, numStocks)
        xs.append(risk)
        ys.append(ret)
    return xs, ys


def plotOptimalCurveMonteCarlo(xs, ys, variance, returns, tickers, titleVal, ax, withHull=True, withAnnotations=False):
    ax.plot(xs, ys, '.')
    ax.plot(variance, returns, '+')
    ax.set_xlabel('Variance')
    ax.set_ylabel('Returns')
    ax.set_title(f'Avg daily returns vs Variance {titleVal}')
    outerXs, outerYs = getConvexHull(xs, ys)
    if withHull:
        ax.plot(outerXs, outerYs, 'r')
    if withAnnotations:
        for var, ret, ticker in zip(variance, returns, tickers):
            ax.annotate(ticker, (var, ret))
    return outerXs, outerYs

def plotConvexHulls(data, saveData=True):
    # Initialize plot
    pltFigure()
    for info in data:
        xs = info['data'][0]
        ys = info['data'][1]
        label = info['label']
        plt.plot(xs, ys, label=f'N Stocks: {label}')
        if saveData:
            npData = np.vstack((xs,ys)).T
            np.save(f'frontier_{label}', npData)
    plt.title('Markowitz Frontiers for Random Portfolios')
    plt.legend()
    plt.show()

def monteCarloSim(covMat, returns, variance, tickers, axes):
    xs1, ys1 = monteCarloFrontier(covMat, returns, NPTS[0])
    xs2, ys2 = monteCarloFrontier(covMat, returns, NPTS[1])
    xs3, ys3 = monteCarloFrontier(covMat, returns, NPTS[2])
    xs4, ys4 = monteCarloFrontier(covMat, returns, NPTS[3])
    opt1 = plotOptimalCurveMonteCarlo(xs1, ys1, variance, returns, tickers, str(NPTS[0]), axes[0])
    opt2 = plotOptimalCurveMonteCarlo(xs2, ys2, variance, returns, tickers, str(NPTS[1]), axes[1])
    opt3 = plotOptimalCurveMonteCarlo(xs3, ys3, variance, returns, tickers, str(NPTS[2]), axes[2])
    opt4 = plotOptimalCurveMonteCarlo(xs4, ys4, variance, returns, tickers, str(NPTS[3]), axes[3])

    frontiers = [
        {
            'data': opt1,
            'label': str(NPTS[0])
        },
                {
            'data': opt2,
            'label': str(NPTS[1])
        },
                {
            'data': opt3,
            'label': str(NPTS[2])
        },
                {
            'data': opt4,
            'label': str(NPTS[3])
        }
    ]
    plotConvexHulls(frontiers)

def getConvexHull(xs, ys):
    data = np.vstack((xs,ys)).T
    cHull = ConvexHull(data)
    newXs = []
    newYs = []
    for pt in cHull.vertices:
        newXs.append(xs[pt])
        newYs.append(ys[pt])

    startIndex = np.argmin(newYs)
    # Starts at bottom point, goes clockwise
    # until it falls from the top point
    optXs = []
    optYs = []
    numPts = len(newXs)
    priorY = 0
    for i in range(numPts):
        idx = (startIndex - i) % numPts
        x = newXs[idx]
        y = newYs[idx]
        if(y < priorY):
            break
        else:
            optXs.append(x)
            optYs.append(y)
            priorY = y
    return optXs, optYs

if __name__ == '__main__':
    # Load covariance, returns, tickers
    covMat = np.load('covmat.npy')
    returns = np.load('returns.npy')
    tickers = loadTickers()
    variance = np.diag(covMat)
    #plotData(variance, returns, tickers)

    axes = getAxes()
    reqAnnualReturn = 0.05

    # res = solveOptimalRisk(covMat, returns, np.mean(returns))

    # xs, ys = getCurve(covMat, returns)
    # plotOptimalCurve(xs, ys, variance, returns, tickers)

    monteCarloSim(covMat, returns, variance, tickers, axes)
    
    plt.show()

    # Plot curve of portfolio possible returns vs variance (markowitz portfolio curve)
    # Considering future projections

    # Dimension reduction with auto-encoder and de-coder
    # Z-variables (dim(Z) <= dim(W))

    # First: Auto-encoder and Z's
    # Classical is using models (fama-french for example), drawback is linearity
    # (Decent locally, but not for large deviations)
    # Compare to standard factor model (as in deep portfolio paper)
    # Predictions could be more accurate using autoencoder
