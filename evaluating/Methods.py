
import cvxopt as opt
from cvxopt import blas, solvers
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import os
import matplotlib.cm as cm
import matplotlib
# Source: https://plotly.com/python/v3/ipython-notebooks/markowitz-portfolio-optimization/

NUM_ITERS = 1000
def scipyOptimize(covmat, rets, retBar, startingPoint, gamma, method='slsqp'):
    weights_init = np.array(startingPoint)
    def f(xbar):
        return np.matmul(np.matmul(xbar, covmat), xbar)
    def jac(xbar):
        return 2*np.matmul(xbar, covmat)
    def hess(xbar):
        return 2*covmat

    # No shorting allowed, no position size greater than portfolio
    bounds = scipy.optimize.Bounds([-gamma] * len(rets), [1] * len(rets))

    # SLSQP method
    if(method == 'slsqp' or method == 'SLSQP'):
        # Sum to 1 constraint
        sumToOne = {'type': 'ineq',
                'fun': lambda x: 1-np.sum(x),
                'jac': lambda x: -1*np.ones(len(x))
                }

        # Expected return constraint
        sufficientReturn = {
            'type': 'ineq',
            'fun': lambda x: np.matmul(x.T, rets)[0] - retBar,
            'jac': lambda x: rets[:,0]
        }

        #https://docs.scipy.org/doc/scipy/tutorial/optimize.html#sequential-least-squares-programming-slsqp-algorithm-method-slsqp
        result = scipy.optimize.minimize(f, weights_init, constraints=[sumToOne, sufficientReturn], jac=jac, method='slsqp', bounds=bounds)
        return result

    # Trust-constr method
    elif(method == 'trust-constr' or method=='TRUST-CONSTR' or method=='TRUSTCONSTR' or method=='trustconstr'):
        linearSumToOne = scipy.optimize.LinearConstraint([1]*len(rets),0,1) # Sum of positions has to be between 0 and 1
        linearReturns = scipy.optimize.LinearConstraint(rets[:,0],retBar,np.inf) # Return must be greater than the retBar

        #https://docs.scipy.org/doc/scipy/tutorial/optimize.html#trust-region-constrained-algorithm-method-trust-constr
        result = scipy.optimize.minimize(f, weights_init, method='trust-constr', jac=jac, hess=hess,constraints=[linearSumToOne, linearReturns], bounds=bounds)
        return result

    else:
        print('ERROR: solveOptimalRisk: UNSUPPORTED METHOD ', method)
    
class Object(object):
    pass

def optimal_portfolio_cvx(covMat, rets):
    n = len(rets)
    N = 20
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(covMat)
    print('S: ', S)
    pbar = opt.matrix(rets)
    print('Pbar: ', pbar)
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n)*0) # Disables positive constraint
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = []
    failedMus = []
    for mu in mus:
        try:
            portfolio = solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
            portfolios.append(portfolio)
        except:
            failedMus.append(mu)
    print('Successes: ', len(portfolios))
    print('Fails: ', len(failedMus))
    print('Portfolio: ', np.array(portfolios[0]))
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks

def plot_optimal_portfolio_cvx(covMat, rets, stockStds, stockMeans):
    returns, risks = get_cvx_comparison(covMat, rets)
    fig = plt.figure()
    plt.plot(np.power(stockStds,2), stockMeans, 'o')
    plt.ylabel('mean')
    plt.xlabel('variance')
    plt.plot(np.power(risks,2), returns, 'y-o')
    plt.show()

def get_cvx_comparison_objects(covMat, rets):
    n = len(rets)
    N = NUM_ITERS
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(covMat)
    pbar = opt.matrix(rets)
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = []
    failedMus = []
    for mu in mus:
        try:
            # x*(muS)*x + x*(-pbar)
            portfolio = solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
            portfolios.append(portfolio)
        except:
            failedMus.append(mu)
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    objList = []
    for x in portfolios:
        objList.append({
            'weights': np.array(x),
            'risk': blas.dot(x, S*x),
            'success': True,
            'ret': blas.dot(pbar, x),
        })
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    return objList

'''
covMat: nxn covariance matrix
rets: nx1 returns vector
retBar: desired return
hasCash: True if investor is allowed to hold cash
shortsAllowed: True if investor is allowed to short stocks
'''
def get_cvx_opt_fixedRet(covMat, rets, retBar, hasCash=True, shortsAllowed=True, gamma=0.02):
    n = len(rets)
    N = 100
    
    # Convert to cvxopt matrices
    S = opt.matrix(covMat)
    pbar = opt.matrix(rets*0)
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(np.vstack((np.ones(n), rets.reshape(1,n))))
    b = opt.matrix(np.array([[1],[retBar]]))

    if((hasCash) and (not shortsAllowed)):
        h = np.zeros((n+1,1))
        h[n][0] = 1
        G = opt.matrix(np.vstack((-np.eye(n), np.ones(n))))   # negative n x n identity matrix
        h = opt.matrix(h)
        A = opt.matrix(rets.reshape(1,n))
        b = opt.matrix(retBar)

    if(hasCash and shortsAllowed):
        h = np.ones((n+1,1)) * gamma
        h[n][0] = 1
        G = opt.matrix(np.vstack((-np.eye(n), np.ones(n))))   # negative n x n identity matrix
        h = opt.matrix(h)
        A = opt.matrix(rets.reshape(1,n))
        b = opt.matrix(retBar)

    if((not hasCash) and shortsAllowed):
        G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
        h = opt.matrix(gamma, (n ,1))
        A = opt.matrix(np.vstack((np.ones(n), rets.reshape(1,n))))
        b = opt.matrix(np.array([[1],[retBar]]))
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = []
    failedMus = []
    portfolio = solvers.qp(S, -pbar, G, h, A, b)
    return portfolio

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

def plotOptimalCurves(dataObjs, variance, returns, tickers, title=None, rainbow=False):
    plt.figure()
    nPts = len(variance)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.rainbow)

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
        if(rainbow):
            index = 0
            for x, y in zip(xs, ys):
                plt.scatter(x, y, color=mapper.to_rgba((0.4 * index / nPts)))
                index += 1
        else:
            plt.plot(xs, ys, dataObj['marker'], label=f'Frontier: {label}')



    plt.plot(variance, returns, '+', label='Tickers')
    plt.xlabel('Variance')
    plt.ylabel('Returns')
    for var, ret, ticker in zip(variance, returns, tickers):
        plt.annotate(ticker, (var, ret))
    plt.legend()



def printPortfolioSummary(portfolio, rets, covMat):
    p = np.array(portfolio)
    ret = np.matmul(rets.T, p)[0][0]
    variance = np.matmul(np.matmul(p.T, covMat), p)[0][0]
    #print('Portfolio: ', portfolio)
    print('\n\nReturn: ', ret)
    print('Variance: ', variance)
    print('Maximum Weight: ', np.max(portfolio))
    print('Minimum Weight: ', np.min(portfolio))
    print("Sum weights: ", np.sum(portfolio))
    print('Max Ret: ', np.max(rets))

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
    numPoints = 1000
    xs = []
    ys = []
    for i in range(numPoints):
        ret, risk = runMonteCarlo(covMat, returns, numStocks)
        xs.append(risk)
        ys.append(ret)
    return xs, ys

def plotMonteCarloFrontier(covMat, returns, numStocks):
    xs, ys = monteCarloFrontier(covMat, returns, numStocks)
    plt.plot(xs, ys, '+')
    plt.show()

def pltFigure():
    plt.figure()
    plt.xlim([0, 0.003])
    plt.ylim([0, 0.003])

def getAxes():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    axes = [ax1, ax2, ax3, ax4]
    for ax in axes:
        ax.set_xlim(0, 0.003)
        ax.set_ylim(0, 0.003)
    return axes


def get_global_minimum(covMat):
    n = covMat.shape[0]
    N = 100

    # Convert to cvxopt matrices
    S = opt.matrix(covMat)
    pbar = opt.matrix(np.zeros(n))
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    portfolio = solvers.qp(S, pbar, G, h, A, b)['x'] 
    return portfolio

def getCurveCvx(retVect, covMat, desiredRets, hasCash=True, shortsAllowed=False, gamma=0.02):
    objList = []
    for ret in desiredRets:
        res = get_cvx_opt_fixedRet(covMat, retVect, ret, hasCash=hasCash, shortsAllowed=shortsAllowed, gamma=gamma)
        x = res['x']
        success = res['gap'] < (10**-6)
        objList.append({
        'weights': np.array(x),
        'risk': np.matmul(np.matmul(x.T, covMat),x)[0][0],
        'success': success,
        'ret': np.matmul(retVect.T, x)[0][0],
        })
    return objList


def loadAllDatas(printInfo=False):
    dataObj = {}
    fnames = list(os.listdir())
    fnames.sort()
    for fname in fnames:
        if(fname[-3:] == 'npy'):
            dataObj[fname.split('.')[0]] = np.load(fname, allow_pickle=True)
            if printInfo:
                print(f'File: {fname}, Shape: {dataObj[fname.split(".")[0]].shape}')
    return dataObj

if __name__ == '__main__':
    xs = []
    ys = []
    for obj in res:
        xs.append(obj['risk'])
        ys.append(obj['ret'])
        print(xs)
        print(ys)
    plt.plot(xs, ys)
    plt.show()