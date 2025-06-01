from __future__ import annotations

from typing import List

# import gymnasium as gym
# from gymnasium import spaces
# from gymnasium.utils import seeding
import gym
from gym import spaces
from gym.utils import seeding
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
import os

matplotlib.use("Agg")

class StockPortfolioEnv(gym.Env):
    """A single stock trading environment for OpenAI gym

    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        initial_amount : int
            start money
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        day: int
            an increment number to control date
        reward_scaling : float
            a scaling factor to multiply with the reward in order to normalize it for better training during RL
        risk_free_rate : float
            risk free rate used to calculate the sharpe ratio. This can be set to 0.0 for simplicity
        reward_mode: str
            the reward for the model. ['pnl', 'sharpe']. pnl stands for the daily profits and losses and sharpe stands for the difference in 
            sharpe ratio between two timesteps. Default is 'pnl' if input is invalid

    Methods
    -------
    step()
        at each step the agent will return actions, then 
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                df : pd.DataFrame,
                stock_dim : int,
                initial_amount : int,
                state_space : int,
                action_space : int,
                tech_indicator_list: List[str],
                reward_mode : str,
                day : int = 0,
                reward_scaling : float = 1.0,
                risk_free_rate : float = 0.0,
                results_dir: str = "results"
                ):
        #super(StockEnv, self).__init__()
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.initial_amount = initial_amount
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.reward_scaling = reward_scaling
        self.risk_free_rate = risk_free_rate
        self.reward_mode = reward_mode
        self.results_dir = results_dir

        # Create directory to save results
        os.makedirs(self.results_dir, exist_ok = True)

        # action_space normalization and shape is self.stock_dim so essentially a list with 8 elements
        self.action_space = spaces.Box(low = 0, 
                                       high = 1,
                                       shape = (self.action_space,)
                                       ) 
        
        # (covariance matrix + technical indicators, stock_dim)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
        #                                     shape = (self.state_space+len(self.tech_indicator_list),
        #                                              self.state_space)
        #                                     )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape = (self.state_space,
                                                     self.state_space+len(self.tech_indicator_list))
                                            )

        # load data from a pandas dataframe
        self.covs_list = [col for col in self.df.columns.tolist() if "cov_" in col]
        self.features = self.tech_indicator_list + self.covs_list
        self.data = self.df.loc[self.day,:]
        
        self.state = self.data[self.features].to_numpy() # numpy array of 8x20 each day. 8 stocks, 20 features (indicators + 8 covariances)

        self.terminal = False     
        # initalize portfolio value
        self.portfolio_value = self.initial_amount

        # initialize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # initialize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory=[[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]]
        self.date_memory=[self.data.date.unique()[0]]
        self.rewards_memory = []

        # Initialize prev sharpe in case of reward_mode = "sharpe"
        self.previous_sharpe = 0

    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        excess_returns = np.array(returns) - risk_free_rate
        if excess_returns.std() == 0:
            sharpe_ratio = 0
        else:
            sharpe_ratio = (np.sqrt(252) * excess_returns.mean()) / excess_returns.std() # annualized hence we multiply it with sqrt(252)
        return sharpe_ratio

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique())-1

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ['daily_return']
            plt.plot(df.daily_return.cumsum(),'r')
            #plt.savefig('results/cumulative_reward.png')
            plt.savefig(os.path.join(self.results_dir, "cumulative_reward.png"))
            plt.close()
            
            plt.plot(self.portfolio_return_memory,'r')
            #plt.savefig('results/rewards.png')
            plt.savefig(os.path.join(self.results_dir, "rewards.png"))
            plt.close()

            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))           
            print("end_total_asset:{}".format(self.portfolio_value))

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ['daily_return']
            if df_daily_return['daily_return'].std() !=0:
              sharpe = (252**0.5)*df_daily_return['daily_return'].mean()/ \
                       df_daily_return['daily_return'].std()
              print("Sharpe: ",sharpe)
            print("=================================")
            
            return self.state, self.reward, self.terminal,{}

        else:
            # actions are the portfolio weight
            weights = self.softmax_normalization(actions)
            self.actions_memory.append(weights)
            last_day_memory = self.data

            #load next state
            self.day += 1
            self.data = self.df.loc[self.day,:]
            self.state = self.data[self.features].to_numpy() # numpy array of 8x20 each day. 8 stocks, 20 features (indicators + 8 covariances)
            # calcualte portfolio return
            # individual stocks' return * weight
            portfolio_return = sum(((self.data.close.values / last_day_memory.close.values)-1)*weights)
            # update portfolio value
            new_portfolio_value = self.portfolio_value*(1+portfolio_return)
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])            
            self.asset_memory.append(new_portfolio_value)

            # Reward Calculation
            if self.reward_mode == "sharpe":
                current_sharpe = self._calculate_sharpe_ratio(self.portfolio_return_memory, self.risk_free_rate)
                self.reward = (current_sharpe - self.previous_sharpe) # Reward is the difference between annualized Sharpe ratios per timestep
                self.previous_sharpe = current_sharpe

            else: # this defaults to "pnl" as most RL implementations do
                self.reward = portfolio_return

            # Save the unscaled rewards for easier debugging and checking    
            self.rewards_memory.append(self.reward)

            # Scale the reward
            self.reward = self.reward_scaling * self.reward

            # Clip reward to -10 and 10 range
            #self.reward = np.clip(self.reward, -10, 10)

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        # load states
        self.covs_list = [col for col in self.data.columns.tolist() if "cov_" in col]
        self.features = self.tech_indicator_list + self.covs_list
        self.state = self.data[self.features].to_numpy() # numpy array of 8x20 each day. 8 stocks, 20 features (indicators + 8 covariances)
        self.portfolio_value = self.initial_amount
        self.terminal = False 
        self.portfolio_return_memory = [0]
        self.actions_memory=[[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]]
        self.date_memory=[self.data.date.unique()[0]] 
        self.rewards_memory = []
        self.previous_sharpe = 0
        return self.state
    
    def render(self, mode='human'):
        return self.state
        
    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator/denominator
        return softmax_output

    def save_asset_memory(self):
        # return an updated list of portfolio daily returns
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        df_account_value = pd.DataFrame({'date':date_list,'daily_return':portfolio_return})
        return df_account_value

    def save_action_memory(self):
        # return an updated list of portfolio daily weights
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']
        
        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        return df_actions
    
    def save_reward_memory(self):
        # Save the rewards to a dataframe
        df_rewards = pd.DataFrame({
            'date': self.date_memory[1:],
            'reward': self.rewards_memory
            })
        return df_rewards

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
