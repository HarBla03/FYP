import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load processed dataset and parse timestamps as date values
df = pd.read_csv("Data/Processed/processed_data_full.csv", parse_dates=["timestamp"])

#Extract time based features for analysis
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

#Colimn to identify demand levels
demand_col = "demand_normalized"

def best_block(values, block_size, find="max"):
    best_start = 0
    best_score = None

    #Create blocks based on the block size and calculate demand mean
    for start in range(0, len(values) - block_size + 1):
        block = values[start:start + block_size]
        score = block.mean()

        #Update best block depending on the max or min 'find'
        if best_score is None or (find == "max" and score > best_score) or (find == "min" and score < best_score):
            best_score = score
            best_start = start

    return list(range(best_start, best_start + block_size)), best_score

def classify_blocks(time_df, peak=6, offpeak=6):
    #Calculate mean demand for each hour
    hourly_mean = time_df.groupby("hour")[demand_col].mean().sort_index()

    values = hourly_mean.values
    hours = hourly_mean.index.tolist()

    #Find the highest demand and lowest demand blocks
    peak_hours = best_block(values, peak, find="max")[0]
    offpeak_hours = best_block(values, offpeak, find="min")[0]

    #If peak and off peak overlap conduct another search to find non overlapping blocks
    if set(peak_hours) & set(offpeak_hours):
        valid_hours = []
        best_score = None

        for start in range(0, len(values) - offpeak + 1):
            block = list(range(start, start + offpeak))
            block_hours = [hours[i] for i in block]

            #Skip blocks that overlap with peak gours
            if set(block_hours) & set(peak_hours):
                continue

            score = np.mean([hourly_mean[h] for h in block_hours])

            #Keep the lowest demand block that is valid
            if best_score is None or score < best_score:
                best_score = score
                valid_hours = block_hours

        offpeak_hours = valid_hours

    return peak_hours, offpeak_hours

#Classify peak and off peak periods for both weekends and weekdays
for day_type in ["weekday", "weekend"]:
    subset = df[df["is_weekend"] == (1 if day_type == "weekend" else 0)]
    peak_hours, offpeak_hours = classify_blocks(subset)

    print(f"{day_type.capitalize()} Peak Hours: {peak_hours}")
    print(f"{day_type.capitalize()} Off-Peak Hours: {offpeak_hours}")