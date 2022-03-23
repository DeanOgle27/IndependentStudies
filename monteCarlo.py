import matplotlib.pyplot as plt
import numpy as np


def loadTickers():
    f = open('./tickerList.txt', 'r')
    tickers = []
    for line in f.readlines():
        tickers.append(line.replace('\n', ''))
    f.close()
    return tickers


def runMonteCarlo(covMat, returns, numStocks):
    x = np.zeros(returns.shape[0])
    for i in range(numStocks):
        x[np.random.randint(returns.shape[0])] = np.random.random()
    x = x / np.sum(x)

    # Random 5-stock 'portfolio'
    ret = np.matmul(x, returns)[0]
    risk = np.matmul(np.matmul(x, covMat), x)
    return ret, risk


def monteCarloFrontier(covMat, returns, numStocks):
    numPoints = 10000
    xs = []
    ys = []
    for i in range(numPoints):
        ret, risk = runMonteCarlo(covMat, returns, numStocks)
        xs.append(risk)
        ys.append(ret)
    return xs, ys


def plotOptimalCurveMonteCarlo(xs, ys, variance, returns, tickers, titleVal):
    plt.plot(xs, ys, '.')
    plt.plot(variance, returns, '+')
    plt.xlabel('Variance')
    plt.ylabel('Returns')
    plt.title(f'Avg daily returns vs Variance: {titleVal}')
    for var, ret, ticker in zip(variance, returns, tickers):
        plt.annotate(ticker, (var, ret))
    plt.show()


def monteCarloSim(covMat, returns, variance, tickers):
    xs10, ys10 = monteCarloFrontier(covMat, returns, 10)
    xs5, ys5 = monteCarloFrontier(covMat, returns, 5)
    xs3, ys3 = monteCarloFrontier(covMat, returns, 3)
    plotOptimalCurveMonteCarlo(xs3, ys3, variance, returns, tickers, '3')
    plotOptimalCurveMonteCarlo(xs5, ys5, variance, returns, tickers, '5')
    plotOptimalCurveMonteCarlo(xs10, ys10, variance, returns, tickers, '10')


if __name__ == '__main__':
    # Load covariance, returns, tickers
    covMat = np.load('covmat.npy')
    returns = np.load('returns.npy')
    tickers = loadTickers()
    variance = np.diag(covMat)
    #plotData(variance, returns, tickers)

    reqAnnualReturn = 0.05

    # res = solveOptimalRisk(covMat, returns, np.mean(returns))

    # xs, ys = getCurve(covMat, returns)
    # plotOptimalCurve(xs, ys, variance, returns, tickers)

    monteCarloSim(covMat, returns, variance, tickers)
    # Plot curve of portfolio possible returns vs variance (markowitz portfolio curve)
    # Considering future projections

    # Dimension reduction with auto-encoder and de-coder
    # Z-variables (dim(Z) <= dim(W))

    # First: Auto-encoder and Z's
    # Classical is using models (fama-french for example), drawback is linearity
    # (Decent locally, but not for large deviations)
    # Compare to standard factor model (as in deep portfolio paper)
    # Predictions could be more accurate using autoencoder
