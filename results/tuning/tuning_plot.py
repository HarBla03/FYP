import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

base = pd.read_csv("results/tuning/Baseline Model_test_results.csv")
lr = pd.read_csv("results/tuning/Low LR Model_test_results.csv")
hg = pd.read_csv("results/tuning/High Gamma Model_test_results.csv")
explore = pd.read_csv("results/tuning/Long Exploration Model_test_results.csv")

base["Model"] = "Baseline DQN"
lr["Model"] = "Low Learning Rate"    
hg["Model"] = "High Gamma"
explore["Model"] = "Long Exploration"

models = [base, lr, hg, explore]

df = pd.concat([base, lr, hg, explore], ignore_index=True)

#Box Plots
box_plots = ["ep_reward", "revenue"]
for metric in box_plots:
    plt.figure(figsize=(8, 5))
    df.boxplot(column=metric, by="Model", grid=False)
    plt.title(f"{metric.replace('_', ' ').title()} Distribution by Model")
    plt.xlabel("Model")
    plt.ylabel(metric.replace("_", " ").title())
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.show()