import pandas as pd

# Load CSV data
data = pd.read_csv("healthcare_data.csv")

# Print column names
print(data.columns)
