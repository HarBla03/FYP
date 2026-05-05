import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class DPEnvironment(gym.Env):
    #Environment creation for Dynamic Pricing Model 
    
    metadata = {"render_modes": []}
    
    def __init__(self, data_path, price_level = [None], seed = None):
        super().__init__()

        #Load Dataset
        self.data = pd.read_csv(data_path, parse_dates=['timestamp'])

        #Ensure data is sorted by time
        self.data = self.data.sort_values('timestamp').reset_index(drop=True)

        #Group data by day
        self.data["date"] = self.data["timestamp"].dt.date
        self.days = self.data["date"].unique()
        self.current_day = None
        self.day_data = None
        self.hourly_data = 0

        #Define price levels
        if price_level == [None]:
            self.price_levels = np.array([0.6, 0.8, 1, 1.2, 1.4], dtype=np.float32)  # Default price levels (Low to Baseline to High)
        else:
            self.price_levels = np.array(price_level, dtype=np.float32)

        #Set action space the model can choose from
        self.action_space = spaces.Discrete(len(self.price_levels))

        #Define observation space for model
        low_space = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # demand_normalized [0, 1], hour [0, 23], day_of_week [0, 6], is_weekend [0, 1], prev_price_level [0, n_levels-1]
        high_space = np.array([1.0, 23.0, 6.0, 1.0, float(len(self.price_levels) - 1)], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_space, high=high_space, dtype=np.float32)

        #Internal State
        self.prev_price_idx = 2  # Start with baseline price
        self._rng = np.random.default_rng(seed)

    def get_observation(self) -> np.ndarray:
        #Get current hour's data and construct the observation vector            
        row = self.day_data.iloc[self.hourly_data]
        obs = np.array([
            row["demand_normalized"],
            float(row["hour"]),
            float(row["day_of_week"]),
            float(row["is_weekend"]),
            float(self.prev_price_idx)
        ], dtype=np.float32)
        return obs
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        #Sample a random day for the episode then reset internal state
        self.current_day = self._rng.choice(self.days)
        self.day_data = self.data[self.data["date"] == self.current_day].reset_index(drop=True)
        self.hourly_data = 0
        self.prev_price_idx = 2  #Reset to baseline price
        self.target_demand = self.day_data["demand_normalized"].mean()  #Set target demand to daily average
            
        obs = self.get_observation()
        info = {}
            
        return obs, info
        
    def step(self, action: int):
            
        #Clip action to valid range
        action = int(np.clip(action, 0, len(self.price_levels) - 1)) 

        #Get current hour's data
        row = self.day_data.iloc[self.hourly_data]
        baseline_demand = row["demand_normalized"]
        price_multiplier = self.price_levels[action]

        #Simple Elasticity Model: demand decreases as price increases (Modify later)
        alpha = -0.3 #Elasticity coefficient (negative for inverse relationship)
        demand = 1.0 + alpha * (price_multiplier - 1.0)  #Baseline demand is 1.0 at baseline price
        adj_demand = max(demand, 0.0) * baseline_demand #Ensure demand is non-negative

        #Target Demand and Revenue to calculate reward
        revenue = price_multiplier * adj_demand  #Revenue is price times adjusted demand

        #Calculate reward based on how close we are to target demand, penalize high prices and reward revenue
        reward = (-1.5 * (adj_demand - self.target_demand) ** 2 
                 -0.05 * (price_multiplier - self.price_levels[self.prev_price_idx]) ** 2
                 -0.05 * (price_multiplier - 1.0) ** 2
                  + 0.1 * revenue)   
                    
        self.prev_price_idx = action
        self.hourly_data += 1
            
        terminated = self.hourly_data >= len(self.day_data) #EoD
        truncated = False

        if not terminated:
            obs = self.get_observation()
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        info = { "baseline_demand": baseline_demand, "price_multiplier": float(price_multiplier), "adjusted_demand": adj_demand }

        return obs, reward, terminated, truncated, info
