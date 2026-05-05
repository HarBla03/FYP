import numpy as np
from environment import DPEnvironment
from evaluate import evaluate_model

def rule_based_policy(obs, info, env: DPEnvironment):
    demand = float(obs[0])

    #Quantiles calculated from training data (2.d.p)
    q1 = 0.47 
    q2 = 0.57
    q3 = 0.65
    q4 = 0.73

    if demand <= q1:
        target_price = 0.6  #Low demand -> low price
    elif demand <= q2:
        target_price = 0.8  #Mid low demand -> mid low price
    elif demand <= q3:
        target_price = 1.0 #Mid demand -> baseline price
    elif demand <= q4:
        target_price = 1.2  #Mid high demand -> mid high price
    else:
        target_price = 1.4  #High demand -> high price

    action = int(np.where(env.price_levels == target_price)[0][0])
    return action

if __name__ == "__main__":
    env = DPEnvironment(data_path = "Data/Processed/processed_data_test.csv", seed = 100)
    evaluate_model(env, rule_based_policy, name = "Rule Based Policy", output_path = "results/RB_test_results.csv")    