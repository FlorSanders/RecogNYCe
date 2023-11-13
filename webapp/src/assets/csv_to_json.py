import pandas as pd
import os

csv_path = "./data/all/data_webapp.csv"
file_dir = os.path.dirname(os.path.realpath(__file__))
csv_path = os.path.join(file_dir, "..", "..", "..", csv_path)
assert os.path.exists(csv_path), "CSV file does not exist"

df = pd.read_csv(csv_path)
print(df.head(10))
df.to_json(os.path.join(file_dir, "./data.json"), orient="table", indent=4, index=False)
print("Done")