# Import necessary libraries
import pandas as pd

# Read the unzipped data directly into a DataFrame
df = pd.read_csv('/filtered.tsv', sep='\t')

# Filter the data
sub = df[(df["ref_tox"] >= 0.99) & (df["trn_tox"] <= 0.01)]

# Save the filtered data to a new CSV file
sub.to_csv("subset.tsv", sep='\t')

# Read the filtered data back into a new DataFrame
sub = pd.read_csv("subset.tsv", sep="\t")

# Calculate the lengths of text in reference and translation columns
sub["text_ref_length"] = [len(row.split()) for row in sub["reference"].values]
sub["text_trn_length"] = [len(row.split()) for row in sub["translation"].values]

# Print summary statistics
print(sub.describe())
