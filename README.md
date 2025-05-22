# Portfolio-Optimization-using-LLMs-and-RL

This is the repository for the final project of the course Deep Learning from the master's degree in Computer Science at the Athens University of Economics and Business.

This project implements the following:

1) data_prep.py


Uses open source stock data and the FNSPID dataset, which is a dataset containing news summaries. Merges these to ensure sufficient data for a long enough time horizon.
We then select a portfolio of 5-8 stocks, ideally with negative correlation between them to ensure portfolio diversification, manage risk while maintaining returns.

2) sentiment.py

   
Using FinBERT, a sentiment score is calculated for the news summary of each company, creating column "company_sentiment". This is left empty for now if there are no news available for that specific day. 
Also, companies are grouped per sector. For example, supposing Microsoft, a tech company is in the portfolio. Then, 3 companies in the Tech sector act as proxies for the entire Tech sector, for example these could be Apple, IBM and NVIDIA. The average sentiment score for these 3 companies is used to create a new feature, "sector_sentiment". Is left empty if there are no news available that particular day for these specific companies. 

3) portfolio_trading_environment.py


Using the FinRL repository as the base, this script creates a FinRL compatible environment for stock trading, which is able to take as input stock price data, technical indicators widely used in finance and finally the two new created features. This environment is responsible for implementing all the trades and keeping all required memories requires for the RL training algorithms.

4) train.py


Placeholder. Probably need different .py per RL algorithm. Implements the training loop per algorithm.
Note, the model to be optimized is a Feedforward Neural Network, which outputs the weights per stock, symbolizing the percentage of our capital to be allocated in each stock.
