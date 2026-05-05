import numpy as np
from environment import DPEnvironment
from evaluate import evaluate_model

#Flat Tariff Policy always returns baseline price
def flat_policy(obs, info, env: DPEnvironment):
    baseline_idx = int(np.where(env.price_levels == 1.0)[0][0])  
    return baseline_idx 

if __name__ == "__main__":
    env = DPEnvironment(data_path="Data/Processed/processed_data_test.csv", seed = 100)
    evaluate_model(env, flat_policy, name="Flat Tariff Policy", output_path="results/flat_tariff_test_results.csv")    