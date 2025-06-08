import numpy as np
import pandas as pd

def backtest_trading_model(env, model):
    """
    Run the trading model on the given environment and return computed performance metrics.
    """
    obs = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)

    return compute_metrics(env)

def backtest_equal_weight(env):
    """
    Backtest an equal-weighted portfolio in the given environment.
    """
    obs = env.reset()
    done = False
    n_stocks = env.action_space.shape[0]
    equal_weights = np.ones(n_stocks) / n_stocks

    while not done:
        obs, reward, done, _ = env.step(equal_weights)

    return compute_metrics(env)

def backtest_random_weight(env, seed=42):
    """
    Backtest a randomly-weighted portfolio in the given environment.
    A new random portfolio allocation is generated at each step.
    """
    if seed is not None:
        np.random.seed(seed)

    obs = env.reset()
    done = False
    n_stocks = env.action_space.shape[0]

    while not done:
        random_weights = np.random.rand(n_stocks)
        obs, reward, done, _ = env.step(random_weights)

    return compute_metrics(env)


def compute_metrics(env):
    portfolio_values = np.array(env.asset_memory)

    # CR
    log_returns = np.diff(np.log(portfolio_values))  
    cumulative_return = np.exp(log_returns.sum()) - 1

    # SR
    pnl = np.diff(portfolio_values)
    mean_pnl = pnl.mean()
    std_pnl = pnl.std()
    sharpe_ratio = (np.sqrt(252) * mean_pnl / std_pnl) if std_pnl != 0 else 0.0

    # AV
    daily_vol = log_returns.std()
    annualized_volatility = daily_vol * np.sqrt(252)

    # MDD
    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - cumulative_max) / cumulative_max
    max_drawdown = drawdowns.min()

    cumulative_log_returns = np.cumsum(log_returns)
    cumulative_daily_returns = np.exp(cumulative_log_returns) - 1  

    return {
        "cumulative_return": cumulative_return,
        "sharpe_ratio": sharpe_ratio,
        "annualized_volatility": annualized_volatility,
        "max_drawdown": max_drawdown,
        "asset_values": portfolio_values,
        "cumulative_daily_returns": cumulative_daily_returns
    }

def compute_metrics_from_prices(prices, initial_investment=1_000_000):
    prices = np.asarray(prices).flatten()
    
    scaled_prices = prices / prices[0] * initial_investment
    portfolio_values = scaled_prices

    log_returns = np.diff(np.log(portfolio_values))
    cumulative_return = np.exp(log_returns.sum()) - 1

    pnl = np.diff(portfolio_values)
    mean_pnl = pnl.mean()
    std_pnl = pnl.std()
    sharpe_ratio = (np.sqrt(252) * mean_pnl / std_pnl) if std_pnl != 0 else np.nan

    daily_vol = log_returns.std()
    annualized_volatility = daily_vol * np.sqrt(252)

    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - cumulative_max) / cumulative_max
    max_drawdown = drawdowns.min()

    cumulative_log_returns = np.cumsum(log_returns)
    cumulative_daily_returns = np.exp(cumulative_log_returns) - 1

    return {
        "cumulative_return": cumulative_return,
        "sharpe_ratio": sharpe_ratio,
        "annualized_volatility": annualized_volatility,
        "max_drawdown": max_drawdown,
        "asset_values": portfolio_values,
        "cumulative_daily_returns": cumulative_daily_returns
    }
