import pandas as pd
import matplotlib.pyplot as plt

#Load Dataset
df = pd.read_csv("Data/Processed/processed_data_full.csv", parse_dates=["timestamp"])

col = "demand_normalized"

#Calc quantiles to be used in rule-based policy
quantiles = df[col].quantile([0.2, 0.4, 0.6, 0.8])
print (f"Quantiles for {col}:")
print(quantiles)