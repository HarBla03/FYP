import os
import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from environment import DPEnvironment
from evaluate import evaluate_model

#Data path to access test data and check for tuning directory, making it if needed
data_path = "Data/Processed/processed_data_test.csv"
os.makedirs("results/tuning", exist_ok=True)

#Paths to trained DQNS comined into a dictionary
models = {
    "Baseline Model": "models/base_dqn_model.zip",
    "Low LR Model": "models/low_lr_dqn_model.zip",
    "High Gamma Model": "models/high_gamma_dqn_model.zip",
    "Long Exploration Model": "models/long_exploration_dqn_model.zip"}

def policy_from_model(model_path):
    model = DQN.load(model_path)
    def policy(obs, info, env):
        #Use deterministic predictions so comparisons are reproducible
        action = model.predict(obs, deterministic=True)[0]
        return int(action)
    return policy


#Evaluate each model in the same test environment and save the results
for name, path in models.items():
    print(f"Evaluating {name}...")
    env = DPEnvironment(data_path=data_path, seed=100)
    policy = policy_from_model(path)
    output_path = f"results/tuning/{name}_test_results.csv"
    evaluate_model(env, policy, name=name, output_path=output_path)


#Load each models results and combine into one dataframe for comparison
model_results = []
for name in models.keys():
    df = pd.read_csv(f"results/tuning/{name}_test_results.csv")
    df["Model"] = name
    model_results.append(df)

#Group results by model and compare meand and variability
results_df = pd.concat(model_results, ignore_index=True)
results_df.to_csv("results/tuning/combined_model_results.csv", index=False)
summary = results_df.groupby("Model")[["ep_reward", "peak_demand", "avg_demand", "peak_to_avg",
"load_variability", "price_volatility","revenue"]].agg(["mean", "std"])
summary.to_csv("results/tuning/model_summary.csv")

print("Model comparison complete.")