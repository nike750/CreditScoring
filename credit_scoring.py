import pandas as pd

# 1. Load the dataset from your folder
print("Loading dataset...")
df = pd.read_csv('cs-training.csv', index_col=0)

# 2. Show basic information about the data
print("\nDataset loaded successfully!")
print(f"Total rows and columns: {df.shape}")
print("\nFirst 5 rows of data:")
print(df.head())