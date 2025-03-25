import pandas as pd
import ast
from sklearn.model_selection import train_test_split


df = pd.read_parquet("de-en.parquet")
print(df.head())
df.to_csv("de-en.csv", index=False)

df = pd.read_csv("de-en.csv")

df["translation_dict"] = df["translation"].apply(ast.literal_eval)

df["source"] = df["translation_dict"].apply(lambda x: x.get("de", ""))
df["target"] = df["translation_dict"].apply(lambda x: x.get("en", ""))

df = df.drop(columns=["translation", "translation_dict"])

df.to_csv("de_en_parallel.csv", index=False)

print(df.head())

df = pd.read_csv("de_en_parallel.csv")

train_val_df, test_df = train_test_split(df, test_size=0.10, random_state=42)


train_df, dev_df = train_test_split(train_val_df, test_size=0.1111, random_state=42) 

# check the sizes of each split
print("Training samples:", len(train_df))
print("Development samples:", len(dev_df))
print("Test samples:", len(test_df))

train_df.to_csv("de_en_train.csv", index=False)
dev_df.to_csv("de_en_dev.csv", index=False)
test_df.to_csv("de_en_test.csv", index=False)
