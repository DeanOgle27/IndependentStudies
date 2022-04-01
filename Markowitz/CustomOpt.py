import numpy as np
import matplotlib.pyplot as plt
import methods

RETS = np.load('returns.npy')
COVMAT = np.load('covmat.npy')

def fX(x):
    return np.matmul(np.matmul(x, COVMAT), x.T)

def getReturns(x):
    return np.matmul(x, rets)

def gradX(x):
    return np.matmul(x, COVMAT)

def gradAlongConstraint(grad, constr):
    proj = (np.dot(grad, constr) / np.dot(constr, constr)) * constr
    return grad - proj

'''
COVMAT is a nxn matrix
x0 is 1xn
'''
def getGlobalMin():
    stepSize = 100

    x = np.ones(COVMAT.shape[1]) / COVMAT.shape[1]
    constr = np.ones(COVMAT.shape[1])
    ys = []

    for i in range(100):
        stepSize = stepSize * 1.2
        gx = gradX(x)
        gxConstr = gradAlongConstraint(gx, constr)
        newX = x - stepSize * gxConstr
        while(fX(newX) > fX(x)):
            stepSize =  stepSize / 2
            newX = x - stepSize * gxConstr
            print('Reducing Step Size')
        x = newX
        ys.append(fX(x))
    
    plt.plot(ys)
    return x



if __name__ == '__main__':
    #gMin = getGlobalMin()
    #plt.show()
    gMin = np.array(methods.get_global_minimum(COVMAT)).T
    print('Point: ', gMin.shape, ' Variance: ', fX(gMin))


