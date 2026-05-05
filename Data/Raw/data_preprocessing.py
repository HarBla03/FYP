import os
import pandas as pd

# File path and output directory
file_path = "Data/Raw/time_series_60min_singleindex.csv"
output_dir = "Data/Processed"

# Define the columns for time and load
timestamp_col = "utc_timestamp"
load_col = "GB_GBN_load_actual_entsoe_transparency"

# Load the dataset with only the required columns
df = pd.read_csv(file_path, usecols=[timestamp_col, load_col])

# Convert the timestamp column to datetime format
df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)

# Rename columns for consistency
df = df.rename(columns={timestamp_col: "timestamp", load_col: "demand_mw"})

# Sort the dataset by timestamp
df = df.sort_values("timestamp").reset_index(drop=True)
df = df.dropna(subset=["timestamp"])

# Fill missing values in the demand column using linear interpolation
df["demand_mw"] = df["demand_mw"].interpolate(method="linear")

#Create time features the environment can observe
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# Normalize the demand values using Min-Max scaling
demand_min = df["demand_mw"].min()
demand_max = df["demand_mw"].max()
df["demand_normalized"] = (df["demand_mw"] - demand_min) / (demand_max - demand_min)

# Training Testing split - Keeping last 30 days for testing
split_date = df["timestamp"].max() - pd.Timedelta(days=30)
train_df = df[df["timestamp"] < split_date].copy()
test_df = df[df["timestamp"] >= split_date].copy()

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Print dataset information after cleaning and splitting
print("\nAfter cleaning shape:", df.shape)
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

print("\nTrain date range:")
print(train_df["timestamp"].min(), "to", train_df["timestamp"].max())

print("\nTest date range:")
print(test_df["timestamp"].min(), "to", test_df["timestamp"].max())

# Save the processed dataset
df.to_csv(os.path.join(output_dir, "processed_data_full.csv"), index=False)
train_df.to_csv(os.path.join(output_dir, "processed_data_train.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "processed_data_test.csv"), index=False)