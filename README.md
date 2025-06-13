# Portfolio-Optimization-using-LLMs-and-RL

This is the repository for the final project of the course Deep Learning from the master's degree in Computer Science at the Athens University of Economics and Business, a course taken as part of my duties as a first year PhD student.

This project implements the following:

1) FNSPID_Preprocessing_Deep_Learning.ipynb


This notebook loads the FNSPID dataset which contains news headlines and news content summaries for stocks in the New York stock exchange from 2009 to the end of 2023. 
It filters and keeps only a predefined list of tickers. Some data cleaning is done and then, using FinBERT, the sentiment is extracted for the news content summaries. 
This sentiment can be Positive, Negative and Neutral. These datasets are uploaded to my private repo on HuggingFace.

2) data_prep.ipynb.


This notebook is responsible for loading the final version of the FNSPID data alongside stock price data from YahooFinance. 
Using the YahooFinance data, popular technical indicators are calculated for each ticker and each date. 
The FNSPID dataset is used to construct a Company_Sentiment for the companies in the portfolio but also a Sector_Sentiment, as the average sentiment of the companies in the same sector. 
Please note that the Company_Sentiment values are -1 for negative, 0 for neutral and 1 for positive. The Sector_Sentiment will also be a value from -1 to 1.


3) env_stocktrading_portfolio_allocation.py


This script contains the OpenAI gym environment for the portfolio allocation task. Handles all the observation, state, reward logic. This environment is ensured to be compatible with FinRL and Stable-Baselines3 for model training.

4) utils.py


Simple helper script containing a standardization by ticker function, helpful for scaling the data before training to ensure that price differences in stock closing prices do not affect the model training. 

5) evaluate_strategy.py


Simplified code to evaluate trading strategies. Used in the training loops to compare different hyperparameter configurations based on their Cumulative Return and Sharpe Ratio. Please note that the calculations here are slightly different than in backtest_metrics.py so there might be some small discrepancy in the values of CR or SR calculated by evaluate_strategy.py versus backtest_metrics.py. However, since the transformations applied to calculate these metrics are increasing, the results and choices made remain correct and properly justified. The only slight difference might be some small percentage point differences.  

6) config_pnl.py

   
Hyperparameter configuration settings for the hyperparameter tuning training run train_pnl.py. 

7) train_pnl.py

    
Smaller training run compared to the final one. Uses different hyperparameter configurations for all four training algorithms. After a model is trained, it is saved and evaluated on the validation data. This enabled a comparison between different hyperparameter settings. We choose the best configuration for each model based on the metrics calculated in evaluate_strategy.py.

8) config_final.py

 
Contains a dictionary for each algorithm with the final hyperparameter configuration.

9) train_final.py

 
Final training script. Uses the configuration from config_final.py to train and save four different models (one for each deep RL algorithm). These models are trained for more timesteps compared to the hyperparameter tuning run and are also trained on a dataset containing both the train and validation data from train_pnl.py. 

10) backtest_metrics.py

 
Utility script containing metric calculation functions for the final backtest evaluation. These metrics are calculated exactly as they are presented in the paper.

11) backtest.ipynb

 
Final script which uses the backtest_metrics.py to compute metrics for the test data and also create the plot presented in the paper.
