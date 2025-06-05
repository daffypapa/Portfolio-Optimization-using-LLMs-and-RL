import os
import json
from finrl.agents.stablebaselines3.models import DRLAgent
from env_stocktrading_portfolio_allocation import StockPortfolioEnv
from utils import standardize_by_ticker
import pandas as pd
import numpy as np

from config_final import algorithms

train_path = "/content/full_data_train.pkl"
train_data = pd.read_pickle(train_path)

feature_cols = [
    "macd", "rsi_30", "cci_30", "dx_30", "close_30_sma",
    "close_60_sma", "close_logdiff",
    "close_30_sma_logdiff", "close_60_sma_logdiff",
    "company_sentiment", "sector_sentiment"
]

# Add covariance columns to the features to standardize them as well
cov_cols = [col for col in train_data.columns if col.startswith("cov_")]
feature_cols += cov_cols

train_data = standardize_by_ticker(train_data, feature_cols)

# Define common kwargs 
stock_dimension = len(train_data.tic.unique())
state_space = stock_dimension
tech_indicator_list = [
    "macd", "rsi_30", "cci_30", "dx_30", "close_30_sma",
    "close_60_sma", "close", "close_logdiff", 
    "close_30_sma_logdiff", "close_60_sma_logdiff",
    "company_sentiment", "sector_sentiment"
]

common_env_kwargs = {
    "stock_dim": stock_dimension,
    "initial_amount": 1_000_000,
    "state_space": state_space,
    "action_space": stock_dimension,
    "tech_indicator_list": tech_indicator_list,
    "reward_mode": "sharpe",
    "reward_scaling": 100.0,
    "risk_free_rate": 0.0
}

# Training Loop
run_counts = {}
summary_records = []

for algo_name, hyperparam_list in algorithms.items():
    assert len(hyperparam_list) == 1, f"{algo_name} must have one hyperparam config"
    params = hyperparam_list[0]

    model_dir = os.path.join("trained_models", algo_name)
    results_dir = os.path.join(model_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    env_kwargs = common_env_kwargs.copy()
    env_kwargs["df"] = train_data
    env_kwargs["results_dir"] = results_dir

    env = StockPortfolioEnv(**env_kwargs)
    sb_env, _ = env.get_sb_env()

    model = DRLAgent(env=sb_env).get_model(algo_name, model_kwargs=params, tensorboard_log="./tensorboard_logs")
    model.learn(total_timesteps=409600, tb_log_name=algo_name, log_interval=1)

    model.save(os.path.join(model_dir, "final_model.zip"))

    with open(os.path.join(model_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=4)
