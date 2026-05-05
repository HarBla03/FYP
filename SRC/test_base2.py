import numpy as np
from environment import DPEnvironment
from evaluate import evaluate_model

def TOU_policy(obs, info, env: DPEnvironment):
    hour = int(obs[1])
    is_weekend = bool(obs[3])

    #Peak and offpeak hours calculated previously
    weekday_peak_hours = [14, 15, 16, 17, 18, 19]
    weekend_peak_hours = [15, 16, 17, 18, 19, 20]

    weekday_offpeak_hours = [0, 1, 2, 3, 4, 5]
    weekend_offpeak_hours = [0, 1, 2, 3, 4, 5]

    if is_weekend: 
        if hour in weekend_peak_hours:
            target_price = 1.4
        elif hour in weekend_offpeak_hours:
            target_price = 0.6
        else:
            target_price = 1.0
    else:
        if hour in weekday_peak_hours:
            target_price = 1.4
        elif hour in weekday_offpeak_hours:
            target_price = 0.6
        else:
            target_price = 1.0

    action = int(np.where(env.price_levels == target_price)[0][0])
    return action

if __name__ == "__main__":
    env = DPEnvironment(data_path="Data/Processed/processed_data_test.csv", seed = 100)
    evaluate_model(env, TOU_policy, name="TOU Policy", output_path="results/tou_test_results.csv")    