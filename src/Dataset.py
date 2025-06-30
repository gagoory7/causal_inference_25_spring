from collections import defaultdict

import numpy as np

import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

class Dataset:
    def __init__(self, df, text, confounder, treatment, outcome, tokenizer, batch_size=32):
        self.data = df
        self.text = text
        self.confounder = confounder
        self.treatment = treatment
        self.outcome = outcome
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def get_dataloaders(self):
        texts = self.data[self.text].values.tolist()
        confounds = self.data[self.confounder].values.tolist()
        treatments = self.data.get(self.treatment, [-1] * len(confounds))

        if isinstance(treatments, pd.Series):
            treatments = treatments.tolist()

        outcomes = self.data.get(self.outcome, [-1] * len(confounds))

        if isinstance(outcomes, pd.Series):
            outcomes = outcomes.tolist()

        train_loader = self._build_dataloader(
            texts, confounds, treatments, outcomes
        )

        return train_loader

    def _build_dataloader(self, texts, confounds, treatments, outcomes):
        out = defaultdict(list)
        # text, Confounders, Treatments, Outcomes
        for W, C, T, Y in zip(texts, confounds, treatments, outcomes):
            encoded = self.tokenizer.encode_plus(
                W, add_special_tokens=True, max_length=512, truncation=True, padding='max_length'
            )
            out['W_ids'].append(encoded['input_ids'])
            out['W_mask'].append(encoded['attention_mask'])
            out['W_len'].append(sum(encoded['attention_mask']))
            out['C'].append(C)
            out['T'].append(T)
            out['Y'].append(Y)

        data = tuple(torch.tensor(out[key]) for key in ['W_ids', 'W_len', 'W_mask', 'C', 'T', 'Y'])
        dataset = TensorDataset(*data)

        return DataLoader(dataset, shuffle=True, batch_size=self.batch_size)

