import pandas as pd

# Load the dataset
df = pd.read_csv("Data/Raw/time_series_60min_singleindex.csv")

# Define the columns for time and load
time_col = "utc_timestamp"
load_col = "GB_GBN_load_actual_entsoe_transparency"

print("Dataset Shape: ", df.shape)

# Check for missing columns
required_columns = [time_col, load_col]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"Error: Missing columns in the dataset: {missing_columns}") 


uk_data = df[required_columns].copy()
uk_data[time_col] = pd.to_datetime(uk_data[time_col], utc=True)

uk_data = uk_data.rename(columns={time_col: "timestamp", load_col: "demand_mw"})


# Displaying basic information about the dataset
print("\nSelected Columns: ")
print(uk_data.columns.tolist())

print("|nFirst 5 rows of the processed dataset:")
print(uk_data.head())

print("\nMissing values in the dataset: ")
print(uk_data.isna().sum())

print("\nBasic Stats:")
print(uk_data["demand_mw"].describe())

print("\nDataset range:")
print(uk_data["timestamp"].min(), "to", uk_data["timestamp"].max())
