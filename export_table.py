import pickle
import pandas as pd

# Load dataset
with open("data.pickle", "rb") as f:
    dataset = pickle.load(f)

data = dataset["data"]
labels = dataset["labels"]

# Create column names (63 features)
columns = []
for i in range(21):
    columns.append(f"landmark_{i}_x")
    columns.append(f"landmark_{i}_y")
    columns.append(f"landmark_{i}_z")

# Convert to DataFrame
df = pd.DataFrame(data, columns=columns)

# Add labels
df["label"] = labels

# Save only first 10 rows (for paper table)
df.head(10).to_csv("table_sample.csv", index=False)

print("✅ Table created successfully → table_sample.csv")