import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
import pca

# Minimum number of samples in ticker data for test-train split
#   Desired: test last 2 years, train prior 3 years; 5 years total
MIN_SAMPLES = 1259
SPLIT_INDEX = 505
PCA_DIM = 10


def loadTickers():
    f = open('./tickerList.txt', 'r')
    tickers = []
    for line in f.readlines():
        tickers.append(line.replace('\n', ''))
    f.close()
    return tickers


if __name__ == '__main__':

    data5Year = np.load('5Year.npy')
    # print(data5Year.shape)
    #plt.figure('PCA Analysis')
    # pca.runPCAAnalysis(data5Year)
    #pca.runPCAAnalysisTestTrainSplit(data5Year, SPLIT_INDEX)
    # plt.show()
    pca.runPCA(data5Year, 15, printStats=True, printCorrels=True)
    # Create numpy array from dictionary

    # Dimension reduction with auto-encoder and de-coder
    # Z-variables (dim(Z) <= dim(W))

    # First: Auto-encoder and Z's
    # Classical is using models (fama-french for example), drawback is linearity
    # (Decent locally, but not for large deviations)
    # Compare to standard factor model (as in deep portfolio paper)
    # Predictions could be more accurate using autoencoder
