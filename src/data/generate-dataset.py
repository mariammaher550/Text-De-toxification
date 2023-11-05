import pandas as pd
from datasets import load_dataset

df = pd.read_csv("subset.tsv", sep="\t")

json_dict = [
    {"instruction": "Rephrase with lower toxicity level.",
    "input": input,
    "output": output,
    }
    for input,output in zip(df["reference"].values, df["translation"].values)
]

with open("alpaca-detoxify-dataset.json", "w") as f:
   json.dump(json_dict, f)

# you need to login
data = load_dataset("json", data_files="/content/alpaca-detoxify-dataset.json")
data.push_to_hub("mariammaher550/detoxify-dataset")
