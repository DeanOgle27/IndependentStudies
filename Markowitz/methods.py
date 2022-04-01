import cvxopt as opt
from cvxopt import blas, solvers
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
# Source: https://plotly.com/python/v3/ipython-notebooks/markowitz-portfolio-optimization/

COVMAT = np.load('covmat.npy')
RETS = np.load('returns.npy')
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
    A = opt.matrix(np.vstack((np.ones(n), rets.reshape(1,75))))
    b = opt.matrix(np.array([[1],[retBar]]))

    if((hasCash) and (not shortsAllowed)):
        h = np.zeros((n+1,1))
        h[n][0] = 1
        G = opt.matrix(np.vstack((-np.eye(n), np.ones(n))))   # negative n x n identity matrix
        h = opt.matrix(h)
        A = opt.matrix(rets.reshape(1,75))
        b = opt.matrix(retBar)

    if(hasCash and shortsAllowed):
        h = np.ones((n+1,1)) * gamma
        h[n][0] = 1
        G = opt.matrix(np.vstack((-np.eye(n), np.ones(n))))   # negative n x n identity matrix
        h = opt.matrix(h)
        A = opt.matrix(rets.reshape(1,75))
        b = opt.matrix(retBar)

    if((not hasCash) and shortsAllowed):
        G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
        h = opt.matrix(gamma, (n ,1))
        A = opt.matrix(np.vstack((np.ones(n), rets.reshape(1,75))))
        b = opt.matrix(np.array([[1],[retBar]]))
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = []
    failedMus = []
    portfolio = solvers.qp(S, -pbar, G, h, A, b)
    return portfolio

def muGenerator(N):
    return [10**(5.0 * t/N - 1.0) for t in range(N)]

def get_cvx_comparison_objects_shorts(covMat, rets, gamma):
    n = len(rets)
    N = 1000
    mus = muGenerator(N)

    # Convert to cvxopt matrices
    S = opt.matrix(covMat)
    pbar = opt.matrix(rets)
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(gamma, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    failedMus = []

    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    objList = []

    for mu in mus:
        try:
            # x*(muS)*x + x*(-pbar)
            res = solvers.qp(S, -pbar/mu, G, h, A, b)
            if res['gap'] >= (10**-6):
                print('FAILED: ', mu)
                ret = blas.dot(pbar, res['x'])
                res2 = scipyOptimize(covMat, rets, ret, res['x'], gamma, method='slsqp')
                #print('RES 2: ', res2)
                x = res2.x
                objList.append({
                'weights': np.array(x),
                'risk': res2.fun,
                'success': False,
                'ret': ret,
                })
            else:
                objList.append({
                    'weights': np.array(res['x']),
                    'risk': blas.dot(res['x'], S*res['x']),
                    'success': res['gap'] < (10**-6),
                    'ret': blas.dot(pbar, res['x']),
                })


        except Exception as e:
            print('Failed: ', mu, ' Message: ', e)
            failedMus.append(mu)

    
    return objList

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

def getCurveCvx(rets, hasCash=True, shortsAllowed=False, gamma=0.02):
    objList = []
    for ret in rets:
        res = get_cvx_opt_fixedRet(COVMAT, RETS, ret, hasCash=hasCash, shortsAllowed=shortsAllowed, gamma=gamma)
        x = res['x']
        success = res['gap'] < (10**-6)
        objList.append({
        'weights': np.array(x),
        'risk': np.matmul(np.matmul(x.T, COVMAT),x)[0][0],
        'success': success,
        'ret': np.matmul(RETS.T, x)[0][0],
        })
    return objList
'''
    - Add other restrictions (can only change n% of portfolio at a time)
    - Shorts/no shorts
    - Factorization (finding eigenvalues)
        - Dimensionality reduction performance
            - Look at post-reduction analysis to see if there is meaning to
                the dimensionality reduction
            - Hopefully dimensions are uncorrelated
            - Probably more open ended
        - Standard indicators?
    - Classification of the stocks

    - Start writing background, methods, little bit of literature, etc.
    - Overview of classical result and then extensions
'''

if __name__ == '__main__':
    res = getCurveCvx(np.linspace(np.min(RETS), np.max(RETS), NUM_ITERS))
    xs = []
    ys = []
    for obj in res:
        xs.append(obj['risk'])
        ys.append(obj['ret'])
        print(xs)
        print(ys)
    plt.plot(xs, ys)
    plt.show()
