import numpy as np
import pandas as pd
from environment import DPEnvironment   

def evaluate_model(env: DPEnvironment, test_model, name: str, output_path: str):
    results = []
    num_eps = len(env.days)

    for ep in range(num_eps):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        hourly_base_demand = []
        hourly_adj_demand = []
        hourly_price = []

        current_day = env.current_day

        while not done:
            action = test_model(obs, info, env)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward

            hourly_base_demand.append(info["baseline_demand"])
            hourly_adj_demand.append(info["adjusted_demand"])
            hourly_price.append(info["price_multiplier"])

            done = terminated or truncated

        base_array = np.array(hourly_base_demand)
        adj_array = np.array(hourly_adj_demand)
        price_array = np.array(hourly_price)

        peak_demand = np.max(adj_array)
        avg_demand = np.mean(adj_array)
        peak_to_avg = peak_demand / avg_demand if avg_demand != 0 else np.nan
        load_variability = np.std(adj_array) / avg_demand if avg_demand != 0 else np.nan
        price_volatility = np.std(price_array)
        revenue = np.sum(price_array * adj_array)

        results.append({"episode": ep + 1,
                        "date": str(current_day),
                        "ep_reward": ep_reward,
                        "peak_demand": peak_demand,
                        "avg_demand": avg_demand,
                        "peak_to_avg": peak_to_avg,
                        "load_variability": load_variability,
                        "price_volatility": price_volatility,
                        "revenue": revenue})
        
    results_df = pd.DataFrame(results)

    results_df.to_csv(output_path, index=False)

    print(f"{name}: Evaluation complete")
    print(results_df.head())
    print("\nAverage metrics across all episodes:")
    print(results_df.mean(numeric_only=True))