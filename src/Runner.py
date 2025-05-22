from tqdm import tqdm
import numpy as np
import torch
from torch.optim import AdamW


class Causal_KoBert_Runner:
    def __init__(self, model, loss_weights=None):
        self.model = model

        self.device = model.device

        default_weights = {'g': 0.1, 'Q': 0.1, 'mlm': 1.0}
        if loss_weights is None:
            self.loss_weights = default_weights
        else:
            for k, v in default_weights.items():
                loss_weights.setdefault(k, v)
            
            self.loss_weights = loss_weights 

    def train(self, train_dataloader, learning_rate=5e-5):
        self.model.train()

        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)

        losses = []
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            batch = [x.to(self.device) for x in batch]
            W_ids, W_len, W_mask, C, T, Y = batch

            self.model.zero_grad()
            g, _, _, g_loss, Q_loss, mlm_loss = self.model(W_ids, W_len, W_mask, C, T, Y)

            loss = (
                self.loss_weights['g'] * g_loss +
                self.loss_weights['Q'] * Q_loss +
                self.loss_weights['mlm'] * mlm_loss
            )

            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().item())

        print(f"[Train] Loss: {np.mean(losses):.4f}")

    def inference(self, test_dataloader):
        self.model.eval()
        gs, Q0s, Q1s, Ys, Ts = [], [], [], [], []

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluating"):

                batch = [x.to(self.device) for x in batch]
                W_ids, W_len, W_mask, C, T, Y = batch

                g, Q0, Q1, _, _, _ = self.model(W_ids, W_len, W_mask, C, T, use_mlm=False)
                gs += g.cpu().numpy().tolist()
                Q0s += Q0.cpu().numpy().tolist()
                Q1s += Q1.cpu().numpy().tolist()
                Ys += Y.cpu().numpy().tolist()
                Ts += T.cpu().numpy().tolist()

        Q_probs = np.array([Q0s, Q1s]).T
        return gs,Q_probs, Ys, Ts


class KoBert_Runner:
    def __init__(self, model,loss_weights=None):
        self.model = model

        self.device = model.device

        default_weights = {'Q': 0.1, 'mlm': 1.0}

        if loss_weights is None:
            self.loss_weights = default_weights
        else:
            for k, v in default_weights.items():
                loss_weights.setdefault(k, v)
            self.loss_weights = loss_weights 

    def train(self, train_dataloader, learning_rate=5e-5):

        self.model.train()

        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)

        losses = []

        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            
            batch = [x.to(self.device) for x in batch]
            W_ids, W_len, W_mask, C, T, Y = batch

            self.model.zero_grad()

            _, Q_loss, mlm_loss = self.model(W_ids, W_len, W_mask, C, T, Y)

            loss = (
                self.loss_weights['Q'] * Q_loss +
                self.loss_weights['mlm'] * mlm_loss
            )

            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().item())

            print(f"[Train] Loss: {np.mean(losses):.4f}")

    def inference(self, test_dataloader):
        self.model.eval()
        Qs,Ys, Ts = [], [], []

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluating"):
                batch = [x.to(self.device) for x in batch]
                W_ids, W_len, W_mask, C, T, Y = batch

                Q, _, _ = self.model(W_ids, W_len, W_mask, C, T, use_mlm=False)
                Qs += Q.cpu().numpy().tolist()
                Ys += Y.cpu().numpy().tolist()
                Ts += T.cpu().numpy().tolist()

        return Qs, Ys, Ts