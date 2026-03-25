import pandas as pd

df = pd.read_csv("data/ai4i2020.csv")

print(df.shape)
print(df.head())
print(df.columns.tolist())
print(df["Machine failure"].value_counts())