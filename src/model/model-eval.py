import transformers
import textwrap
from transformers import LlamaTokenizer, LlamaForCausalLM
import os
import sys
from typing import List

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)

import fire
import torch
from datasets import load_dataset
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pylab import rcParams

from sklearn.metrics.pairwise import cosine_similarity
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline
import torch
import numpy as np
import evaluate
import re

%matplotlib inline
sns.set(rc={'figure.figsize':(10, 7)})
sns.set(rc={'figure.dpi':100})
sns.set(style='white', palette='muted', font_scale=1.2)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL = "baffo32/decapoda-research-llama-7B-hf"
FINETUNED_MODEL = "mariammaher550/detoxify-alpaca"
model = LlamaForCausalLM.from_pretrained(
    FINETUNED_MODEL,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload"
)

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"

data = load_dataset("mariammaher550/detoxify-dataset")
data["train"]


def generate_prompt(data_point):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""


def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=256,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < 256
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt

train_val = data["train"].train_test_split(
    test_size=0.2, shuffle=True, seed=42
)
train_data = (
    train_val["train"].map(generate_and_tokenize_prompt)
)
val_data = (
    train_val["test"].map(generate_and_tokenize_prompt)
)

# Load the toxicity evaluation module
toxicity = evaluate.load("toxicity", module_type="measurement")

def evaluate_cosine_similarity_toxicity_fluency(data, model, tokenizer, toxicity):
    cosine_similarity_scores = []
    toxicity_scores = []
  

    punctuation = r"[^\w\s]"

    for data_point in data:
        input_text = data_point["input"]

        # Tokenize the input text
        tokenized_input = tokenizer(input_text, return_tensors="pt")

        # Generate a response using the model
        with torch.no_grad():
            output = model.generate(**tokenized_input)

        # Decode the generated output
        predicted_output = tokenizer.decode(output[0], skip_special_tokens=True)
        target_output = data_point["output"]

        # Calculate cosine similarity
        predicted_embeddings = model.get_input_embeddings()(tokenized_input["input_ids"])
        target_embeddings = model.get_input_embeddings()(tokenizer(target_output, return_tensors="pt")["input_ids"])

        predicted_embeddings = predicted_embeddings.detach().cpu().numpy()
        target_embeddings = target_embeddings.detach().cpu().numpy()

        cs = cosine_similarity(predicted_embeddings.squeeze(0), target_embeddings.squeeze(0))[0][0]
        print(cs)

        # Calculate toxicity
        toxicity_input_texts = re.split(punctuation,predicted_output)
        toxicity_score = toxicity.compute(predictions=toxicity_input_texts)

        cosine_similarity_scores.append(cs)
        toxicity_scores.append(max(toxicity_score["toxicity"]))
        print(max(toxicity_score["toxicity"]))
        print(f"predicted {predicted_output}")
        print(f"ground truth : {target_output}")
    mean_cosine_similarity = np.mean(cosine_similarity_scores)
    mean_toxicity_score = np.mean(toxicity_scores)

    return mean_toxicity_score, mean_cosine_similarity



test_dataset = data["test"]
toxicity_score, cosine_similarity_score = evaluate_cosine_similarity_toxicity(
    test_dataset, model, tokenizer, toxicity
)
print("Cosine Similarity Score:", cosine_similarity_score)
print("Toxicity Score:", toxicity_score)

