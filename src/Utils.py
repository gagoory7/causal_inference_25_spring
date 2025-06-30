import os
import re
import random

from scipy import stats

from sklearn.preprocessing import LabelEncoder

import numpy as np

import pandas as pd

import torch

# seed setting

def set_seed(seed=42) :
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

# data setting function

def clean_str(text):
    text = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', ' ', text)

    text = re.sub(r'\s*[“”]\s*', '', text)

    text = re.sub(r'(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', ' ', text)

    text = re.sub(r'[ㄱ-ㅎㅏ-ㅣ]+', ' ', text)

    text = re.sub(r'<[^>]*>', ' ', text)

    text = re.sub(r'(?<!\d)[^\w\s\n.~/-]+(?!\d)', ' ', text)  

    text = re.sub(r'(?<!\d)[.~/-](?!\d)', ' ', text)

    text = text.replace('\n', ' ')

    text = re.sub(r'\s+', ' ', text).strip()

    return text

def fix_reporter_name(name):
    if pd.isna(name):
        return name 
    if not str(name).endswith("기자"):
        return str(name) + " 기자"
    return name

def encode_columns_with_offset(df, columns):
    le = LabelEncoder()
    offset = 0
    confounder_col_for_train = []

    for col in columns:
        new_col = f"{col}_label"
        df[new_col] = le.fit_transform(df[col]) + offset
        offset += df[new_col].max() + 1  # offset to avoid overlap
        confounder_col_for_train.append(new_col)
    return df, offset, confounder_col_for_train

def make_dataframe(confounder_col,thresholding) :
    folder = "data/crawl"
    csv_files = [f for f in os.listdir(folder) if f.endswith(".csv") and not f.endswith("_error.csv")]

    if not csv_files:
        raise FileNotFoundError("No data found.")
    
    dfs = []
    for fname in csv_files:
        path = os.path.join(folder, fname)
        df = pd.read_csv(path, index_col=0)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)

    df['reporter'] = df['reporter'].apply(fix_reporter_name)
    df = df.merge(gender, on=['media','reporter','reporter_link'], how='left')
    df = df[df['gender'].notna()]

    try :
        gender = pd.read_csv('data/prepare/gender.csv', index_col=0)
    except :
       raise FileNotFoundError("No gender data found.")

    df = df[df['content'].notna()]
    df['text'] = df['content'].apply(clean_str)

    df['reactions'] = pd.to_numeric(df['reactions'], errors='coerce').fillna(0).astype(int)

    df= df[df['reactions'] > thresholding]

    df['outcome'] = np.log(df['reactions'])

    df.to_csv('data/df.csv')

    return encode_columns_with_offset(df, confounder_col)

# model function

def make_bow_vector(ids, vocab_size, use_counts=False):
    if ids.dim() == 3:
        ids = ids.squeeze(1)

    device = ids.device
    batch_size = ids.size(0)

    vec = torch.zeros((batch_size, vocab_size), device=device)

    ones = torch.ones_like(ids, dtype=torch.float)

    vec.scatter_add_(dim=1, index=ids, src=ones)

    if not use_counts:
        vec = (vec > 0).float()

    return vec

# hypothsis function

def ATT_hypothsis(DR, Ts, alpha=0.05):
    treated_DR = DR[Ts == 1]

    n = len(treated_DR)
    mean_d = np.mean(treated_DR)
    std_d = np.std(treated_DR, ddof=1)
    
    print("ATT_DR", mean_d)
    print("std", std_d)

    t = mean_d / (std_d / np.sqrt(n) + 1e-10)
    critical_value = stats.t.ppf(alpha/2, df=n-1)

    if t < critical_value:
        print("Reject the null hypothesis")
    else:
        print("Fail to reject the null hypothesis")

# resume Currently not in use

def extract_i(filename, tag):
    match = re.match(rf"(\d+)_({tag})_LOG\.pth", filename)
    if match:
        return int(match.group(1))
    return None

def get_newest_model_path(tag):
    model_dir = 'result/model'
    files = os.listdir(model_dir)
    valid_files = [(extract_i(f, tag), f) for f in files if f.endswith(f"{tag}_LOG.pth")]
    valid_files = [(i, f) for i, f in valid_files if i is not None]
    if not valid_files:
        raise FileNotFoundError(f"No model files found with tag {tag}")
    max_i, filename = max(valid_files, key=lambda x: x[0])
    return max_i, os.path.join(model_dir, filename)

