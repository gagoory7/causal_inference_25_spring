import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel

from src.Utils import make_bow_vector


class Causal_Kobert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.MASK_IDX = config.MASK_IDX

        self.bert = BertModel(config)
        self.vocab_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.vocab_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.vocab_projector = nn.Linear(config.hidden_size, config.vocab_size)

        # Q functions for T=0 and T=1 (output a scalar)
        self.Q_cls = nn.ModuleDict({
            '0': nn.Sequential(
                nn.Linear(self.hidden_size + self.num_labels, 200),
                nn.ReLU(),
                nn.Linear(200, 1)),
            '1': nn.Sequential(
                nn.Linear(self.hidden_size + self.num_labels, 200),
                nn.ReLU(),
                nn.Linear(200, 1))
        })

        # g function (also outputs scalar)
        self.g_cls = nn.Linear(self.hidden_size + self.num_labels, 2)

        self.init_weights()


    def forward(self, W_ids, W_len, W_mask, C, T, Y=None, use_mlm=True):
        if use_mlm:
            W_len = W_len.unsqueeze(1) - 2
            mask_class = torch.cuda.FloatTensor if W_ids.is_cuda else torch.FloatTensor
            mask = (mask_class(W_len.shape).uniform_() * W_len.float()).long() + 1
            target_words = torch.gather(W_ids, 1, mask)
            mlm_labels = torch.ones_like(W_ids).long() * -100
            mlm_labels = mlm_labels.cuda() if W_ids.is_cuda else mlm_labels
            mlm_labels.scatter_(1, mask, target_words)
            W_ids.scatter_(1, mask, self.MASK_IDX)

        outputs = self.bert(W_ids, attention_mask=W_mask)
        seq_output = outputs.last_hidden_state

        pooled_output = seq_output[:, 0]

        if use_mlm:
            prediction_logits = self.vocab_transform(seq_output)
            prediction_logits = F.gelu(prediction_logits)
            prediction_logits = self.vocab_layer_norm(prediction_logits)
            prediction_logits = self.vocab_projector(prediction_logits)
            mlm_loss = nn.CrossEntropyLoss()(
                prediction_logits.view(-1, self.vocab_size), mlm_labels.view(-1))
        else:
            mlm_loss = 0.0

        C_bow = make_bow_vector(C.unsqueeze(1), self.num_labels)

        inputs = torch.cat((pooled_output, C_bow), 1)

        g = self.g_cls(inputs)

        if Y is not None:
            g_loss = nn.CrossEntropyLoss()(g.view(-1, 2), T.view(-1).long())
        else:
            g_loss = 0.0

        Q0 = self.Q_cls['0'](inputs).clamp(min=0).squeeze(-1)
        Q1 = self.Q_cls['1'](inputs).clamp(min=0).squeeze(-1)

        if Y is not None:
            mask_T0 = (T == 0)
            mask_T1 = (T == 1)

            Q_loss_T0 = nn.MSELoss()(Q0[mask_T0], Y[mask_T0].float()) if mask_T0.any() else 0.0
            Q_loss_T1 = nn.MSELoss()(Q1[mask_T1], Y[mask_T1].float()) if mask_T1.any() else 0.0
            Q_loss = Q_loss_T0 + Q_loss_T1
        else:
            Q_loss = 0.0

        g = F.softmax(g, dim=1)[:, 1]

        return g, Q0, Q1, g_loss, Q_loss, mlm_loss



class Kobert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.MASK_IDX = config.MASK_IDX
        
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.bert = BertModel(config)
        self.vocab_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.vocab_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.vocab_projector = nn.Linear(config.hidden_size, config.vocab_size)

        # Q functions for T=0 and T=1 (output a scalar)
        self.Q = nn.Sequential(
                nn.Linear(self.hidden_size + self.num_labels, 200),
                nn.ReLU(),
                nn.Linear(200, 1)
        )

        self.init_weights()

    def forward(self, W_ids, W_len, W_mask, C, T, Y=None, use_mlm=True):

        if use_mlm:
            W_len = W_len.unsqueeze(1) - 2
            mask_class = torch.cuda.FloatTensor if W_ids.is_cuda else torch.FloatTensor
            mask = (mask_class(W_len.shape).uniform_() * W_len.float()).long() + 1
            target_words = torch.gather(W_ids, 1, mask)
            mlm_labels = torch.ones_like(W_ids).long() * -100
            mlm_labels = mlm_labels.cuda() if W_ids.is_cuda else mlm_labels
            mlm_labels.scatter_(1, mask, target_words)
            W_ids.scatter_(1, mask, self.MASK_IDX)

        outputs = self.bert(W_ids, attention_mask=W_mask)
        seq_output = outputs.last_hidden_state

        ### CLS token

        pooled_output = seq_output[:, 0]

        if use_mlm:
            prediction_logits = self.vocab_transform(seq_output)
            prediction_logits = F.gelu(prediction_logits)
            prediction_logits = self.vocab_layer_norm(prediction_logits)
            prediction_logits = self.vocab_projector(prediction_logits)
            mlm_loss = nn.CrossEntropyLoss()(
                prediction_logits.view(-1, self.vocab_size), mlm_labels.view(-1))
        else:
            mlm_loss = 0.0

        # confounder(C)를 BoW 형태로 변환
        C_bow = make_bow_vector(C.unsqueeze(1), self.num_labels)

        inputs = torch.cat((pooled_output, C_bow), 1)  # BERT 결과 + C 결합

        Q = self.Q(inputs).clamp(min=0).squeeze(-1)

        if Y is not None:
            Q_loss = nn.MSELoss()(Q,Y.float())
        else :
            Q_loss = 0.0

        return Q, Q_loss, mlm_loss