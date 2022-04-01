import yfinance as yf

def loadTickers():
    f = open('./tickerList.txt', 'r')
    tickers = []
    for line in f.readlines():
        tickers.append(line.replace('\n', ''))
    f.close()
    return tickers

if __name__ == '__main__':
    tickers = loadTickers()
    tickers = ['ZBRA']
    f = open('TickerInfo.txt', 'a')
    for ticker in tickers:
        data = yf.Ticker(ticker).info
        sector = data['sector']
        industry = data['industry']
        marketCap = data['marketCap']
        dividendYield = data['trailingAnnualDividendYield']
        f.write(f'{ticker},{sector},{industry},{marketCap},{dividendYield}\n')
        print('Done with: ', ticker)
    f.close()
    