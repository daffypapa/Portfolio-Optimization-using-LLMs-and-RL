import numpy as np
import pandas as pd

def evaluate_trading_model(env, model):
    """
    Uses the trained model to perform trades in the evaluation environment. 
    This can be train, validation or backtest.
    """
    obs = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)

    results = compute_metrics(env)

    #results["actions_memory"] = env.actions_memory
    results["actions_memory"] = [list(a) for a in env.actions_memory]
    
    return results

def compute_metrics(env):
    """
    Computes metrics for a given environment.
    Presupposes a gym environment with attributes portfolio_return_memory and asset_memory.
    """
    
    # Calculate portfolio cumulative return
    returns = np.array(env.portfolio_return_memory)
    asset_values = np.array(env.asset_memory)
    cumulative_returns = np.cumprod(1 + returns)
    cumulative_return = cumulative_returns[-1] - 1 

    # Calculate (annualized) Sharpe Ratio
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()

    # Calculate max drawdown metric
    cumulative_max = np.maximum.accumulate(asset_values)
    drawdowns = (asset_values - cumulative_max) / cumulative_max
    max_drawdown = drawdowns.min()

    # Calculate annualized volatility metric
    annualized_volatility = returns.std() * np.sqrt(252)

    # Save metrics to dictionary
    results = {
        "cumulative_return": cumulative_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "annualized_volatility": annualized_volatility,
        "raw_returns": returns,
        "cumulative_returns": cumulative_returns
    }

    return results  
