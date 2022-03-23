import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Minimum number of samples in ticker data for test-train split
#   Desired: test last 2 years, train prior 3 years; 5 years total


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
