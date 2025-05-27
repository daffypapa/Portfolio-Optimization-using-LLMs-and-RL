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


3) 
