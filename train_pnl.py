import os
import json
from finrl.agents.stablebaselines3.models import DRLAgent
from env_stocktrading_portfolio_allocation import StockPortfolioEnv
from evaluate_strategy import evaluate_trading_model
from utils import standardize_by_ticker
import pandas as pd
import numpy as np

from config_pnl import algorithms


train_path = "/content/train_data.pkl"
validation_path = "/content/validation_data.pkl"

train_data = pd.read_pickle(train_path)
validation_data = pd.read_pickle(validation_path)

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
validation_data = standardize_by_ticker(validation_data, feature_cols)

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
    "reward_scaling": 1.0,
    "risk_free_rate": 0.0
}

# Training Loop
run_counts = {}
summary_records = []

for algo_name, hyperparam_list in algorithms.items():
    run_counts[algo_name] = 0
    
    for params in hyperparam_list:
        run_counts[algo_name] += 1
        run_id = f"{algo_name}_run_{run_counts[algo_name]}"
        run_dir = os.path.join("trained_models", run_id)
        results_dir = os.path.join(run_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Update environment kwargs
        train_env_kwargs = common_env_kwargs.copy()
        train_env_kwargs["df"] = train_data
        train_env_kwargs["results_dir"] = results_dir

        # Create environment
        train_env_gym = StockPortfolioEnv(**train_env_kwargs)
        env_train, _ = train_env_gym.get_sb_env()

        # Get model
        model = DRLAgent(env=env_train).get_model(algo_name, model_kwargs=params)

        # Train model
        #model.learn(total_timesteps=512) # Small for debug now
        #model.learn(total_timesteps=102400)
        model.learn(total_timesteps=204800)

        # Save model
        model_path = os.path.join(run_dir, "final_model.zip")
        model.save(model_path)

        # Save hyperparameters
        with open(os.path.join(run_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=4)

        # Evaluate on validation
        val_env_kwargs = common_env_kwargs.copy()
        val_env_kwargs["df"] = validation_data
        val_env_kwargs["results_dir"] = results_dir
        val_env_gym = StockPortfolioEnv(**val_env_kwargs)

        results = evaluate_trading_model(val_env_gym, model)

        # Save evaluation results
        with open(os.path.join(run_dir, "validation_results.json"), "w") as f:
            json.dump(
                    {
                        k: float(v) if np.isscalar(v)
                        else v.tolist() if isinstance(v, np.ndarray)
                        else v
                        for k, v in results.items()
                        if not isinstance(v, (pd.DataFrame, pd.Series))
                    },
                    f,
                    indent=4
                )
        
        summary_record = {
            "run_id": run_id,
            "algorithm": algo_name,
            **params,
            **{
                k: float(v) if np.isscalar(v)
                else v.tolist() if isinstance(v, np.ndarray)
                else v
                for k, v in results.items()
                if not isinstance(v, (pd.DataFrame, pd.Series))
            }
        }
        summary_records.append(summary_record)


summary_df = pd.DataFrame(summary_records)
summary_csv_path = os.path.join("trained_models", "summary_results.csv")
summary_df.to_csv(summary_csv_path, index=False)
