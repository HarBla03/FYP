from stable_baselines3.common.env_checker import check_env
from environment import DPEnvironment

env = DPEnvironment(data_path="Data/Processed/processed_data_train.csv")

check_env(env, warn=True)

print("Environment check passed successfully!")