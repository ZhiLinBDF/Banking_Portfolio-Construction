import numpy as np
import pandas as pd
from pandas_datareader import data
from matplotlib import pyplot as plt
df = data.DataReader(['GOOGL', 'COST','NKE','NOG','MCB','TMO','KIM','MSFT','JBHT','AEE'], 'yahoo', start='2020/10/20', end='2021/10/20')
df= df['Adj Close']

cov_matrix=df.pct_change().apply(lambda x: np.log(1+x)).cov() #covariance matrix
print(cov_matrix)

corr_matrix = df.pct_change().apply(lambda x: np.log(1+x)).corr() #correlation matrix
print(corr_matrix)

# Yearly returns for individual companies
ind_er = df.resample('Y').last().pct_change().mean()

#average weighted portfolio
w = [0.1, 0.1,0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1]
port_return = (w*ind_er).sum()
print(port_return)

#252 trading days
ann_sd = df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(252))

assets = pd.concat([ind_er, ann_sd], axis=1) # Creating a table for visualising returns and volatility of assets
assets.columns = ['Returns', 'Volatility']
print(assets)

p_ret = [] 
p_vol = [] 
p_weights = [] 

num_assets = len(df.columns)
num_portfolios = 100000

for portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)#random floats
    weights = weights/np.sum(weights)
    p_weights.append(weights)
    returns = np.dot(weights, ind_er) # Returns are the product of individual expected returns of asset and its 
                                      # weights 
    p_ret.append(returns)
    var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
    sd = np.sqrt(var) # Daily standard deviation
    ann_sd = sd*np.sqrt(250) # Annual standard deviation = volatility
    p_vol.append(ann_sd)

data = {'Returns':p_ret, 'Volatility':p_vol}

for counter, symbol in enumerate(df.columns.tolist()):
    #print(counter, symbol)
    data[symbol+' weight'] = [w[counter] for w in p_weights]

portfolios  = pd.DataFrame(data)
portfolios.plot.scatter(x='Volatility', y='Returns', marker='o', s=10, alpha=0.3, grid=True, figsize=[10,10])

#minimum volatility Portfolio
min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]                           
print(min_vol_port)

# Finding the optimal portfolio
rf = 0.0166 # risk free rate
optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]
print(optimal_risky_port)
# Plotting optimal portfolio
plt.subplots(figsize=(10, 10))
plt.xlabel('volatility')
plt.ylabel('return')
plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)
plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=500)
