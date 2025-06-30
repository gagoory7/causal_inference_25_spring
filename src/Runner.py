from tqdm import tqdm

import pandas as pd

import numpy as np

import torch
from torch.optim import AdamW

class Causal_KoBert_Runner:
    def __init__(self, model, propensity_weight=0.1, outcome_weight=0.1, mlm_weight=1):
        self.model = model

        self.device = model.device
        
        self.loss_weights = {
            'p': propensity_weight,
            'T': outcome_weight,
            'mlm': mlm_weight
        }

    def train(self, train_dataloader, learning_rate=2e-5, epochs=3):

        self.model.train()

        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)

        for epoch in range(epochs):

            losses = []

            for _, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                batch = [x.to(self.device) for x in batch]
                W_ids, W_len, W_mask, C, T, Y = batch

                self.model.zero_grad()
                _, _, _, propensity_loss, outcome_loss, mlm_loss = self.model(W_ids, W_len, W_mask, C, T, Y)

                loss = (
                    self.loss_weights['p'] * propensity_loss +
                    self.loss_weights['T'] * outcome_loss +
                    self.loss_weights['mlm'] * mlm_loss
                )

                loss.backward()
                optimizer.step()
                losses.append(loss.detach().cpu().item())

            print(f"[Train] Epoch {epoch + 1} - Loss: {np.mean(losses):.4f}")

    def inference(self, test_dataloader):

        self.model.eval()

        propensity, T0s, T1s, Ys, Ts = [], [], [], [], []

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluating"):

                batch = [x.to(self.device) for x in batch]
                W_ids, W_len, W_mask, C, T, Y = batch

                p, T0, T1, _, _, _ = self.model(W_ids, W_len, W_mask, C, T, use_mlm=False)

                propensity += p.cpu().numpy().tolist()
                T0s += T0.cpu().numpy().tolist()
                T1s += T1.cpu().numpy().tolist()
                Ys += Y.cpu().numpy().tolist()
                Ts += T.cpu().numpy().tolist()

        propensity = np.array(propensity)
        eps = 1e-3
        propensity = np.clip(np.array(propensity), eps, 1 - eps ) 

        T0s = np.exp(np.array(T0s))
        T1s = np.exp(np.array(T1s))
        Ys = np.exp(np.array(Ys))
        Ts = np.array(Ts)

        PTs = T0s * (1 - Ts) + T1s * Ts

        # DR score
        DR = ((Ts - propensity) / (propensity * (1 - propensity))) * (Ys - PTs) + (T1s - T0s)

        results = pd.DataFrame({
        'prob': propensity,          # gender prob
        'gender' : Ts,               # gender
        'score' : Ys, # real score
        'T0': T0s,  # T=0 score
        'T1': T1s,  # T=1 score
        'DR' : DR, # DR score
        })
        
        return results, DR
        
    