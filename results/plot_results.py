import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

flat = pd.read_csv("results/flat_tariff_test_results.csv")
tou = pd.read_csv("results/tou_test_results.csv")
rb = pd.read_csv("results/RB_test_results.csv")
dqn = pd.read_csv("results/Tuning/High Gamma Model_test_results.csv")

flat["Model"] = "Flat Tariff"
tou["Model"] = "Time of Use"    
rb["Model"] = "Rule-Based"
dqn["Model"] = "DQN"

models = ["Flat Tariff", "Time of Use", "Rule-Based", "DQN"]

df = pd.concat([flat, tou, rb, dqn], ignore_index=True)

metrics = ["peak_demand", "peak_to_avg","load_variability", "price_volatility", "revenue", "ep_reward"]

#Summary Table
summary = df.groupby("Model")[metrics].mean().reset_index()
print(summary)

#Bar Charts
bar_metrics = ["peak_demand", "peak_to_avg", "load_variability", "price_volatility", "revenue"]
for metric in bar_metrics:
    means = [df[df["Model"] == model][metric].mean() for model in models]
    stds = [df[df["Model"] == model][metric].std() for model in models]

    plt.figure(figsize=(8, 5))
    x = np.arange(len(models))

    plt.bar(x, means, yerr=stds, capsize=5, color=['blue', 'orange', 'green', 'red'])
    plt.xticks(x, models, rotation=10)
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"Mean {metric.replace('_', ' ').title()} by Model")
    plt.tight_layout()
    plt.show()

#Box Plots
box_plots = ["peak_demand", "peak_to_avg", "load_variability"]
for metric in box_plots:
    plt.figure(figsize=(8, 5))
    df.boxplot(column=metric, by="Model", grid=False)
    plt.title(f"{metric.replace('_', ' ').title()} Distribution by Model")
    plt.xlabel("Model")
    plt.ylabel(metric.replace("_", " ").title())
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.show()

box_plots_2 = ["price_volatility", "revenue"]
for metric in box_plots_2:
    plt.figure(figsize=(8, 5))
    df.boxplot(column=metric, by="Model", grid=False)
    plt.title(f"{metric.replace('_', ' ').title()} Distribution by Model")
    plt.xlabel("Model")
    plt.ylabel(metric.replace("_", " ").title())
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.show()

#Per Episode Line Graphs
line_metrics = ["peak_demand", "peak_to_avg", "load_variability"]
for metric in line_metrics:
    plt.figure(figsize=(10, 5))
    for model in models:
        sub = df[df["Model"] == model].sort_values("episode")
        plt.plot(sub["episode"], sub[metric], marker = "o", label=model)

    plt.title(f"{metric.replace('_', ' ').title()} Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel(metric.replace("_", " ").title())
    plt.legend()
    plt.tight_layout()
    plt.show()

