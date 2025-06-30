import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel

from src.Utils import make_bow_vector

MASK_IDX = 4  # KoBert MASK_IDX

class Causal_Kobert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        # kobert 
        self.kobert = BertModel(config)

        # MLM
        self.vocab_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.vocab_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.vocab_projector = nn.Linear(config.hidden_size, config.vocab_size)

        # outcome 
        self.outcome_cls = nn.ModuleDict({
            '0': nn.Sequential(
                nn.Linear(self.hidden_size + self.num_labels, 200),
                nn.ReLU(),
                nn.Linear(200, 1)),
            '1': nn.Sequential(
                nn.Linear(self.hidden_size + self.num_labels, 200),
                nn.ReLU(),
                nn.Linear(200, 1))
        })

        # propensity 
        self.propensity_cls = nn.Linear(self.hidden_size + self.num_labels, 2)
        
        self._init_custom_weights()

        for param in self.bert.parameters():
            param.requires_grad = False
        
    def _init_custom_weights(self):
      
      for name, module in self.named_modules():
          if any(x in name for x in ['outcome_cls', 'propensity_cls', 'vocab_transform', 'vocab_layer_norm', 'vocab_projector']):
              self._init_weights(module) 


    def forward(self, W_ids, W_len, W_mask, C, T, Y=None, use_mlm=True):

        if use_mlm:
            W_len = W_len.unsqueeze(1) - 2
            mask_class = torch.cuda.FloatTensor if W_ids.is_cuda else torch.FloatTensor
            mask = (mask_class(W_len.shape).uniform_() * W_len.float()).long() + 1
            target_words = torch.gather(W_ids, 1, mask)
            mlm_labels = torch.ones_like(W_ids).long() * -100
            mlm_labels = mlm_labels.cuda() if W_ids.is_cuda else mlm_labels
            mlm_labels.scatter_(1, mask, target_words)
            W_ids.scatter_(1, mask, MASK_IDX)

        outputs = self.kobert(W_ids, attention_mask=W_mask)

        seq_output = outputs.last_hidden_state

        # cls token
        pooled_output = seq_output[:, 0]

        # mlm loss
        if use_mlm:
            prediction_logits = self.vocab_transform(seq_output)
            prediction_logits = F.gelu(prediction_logits)
            prediction_logits = self.vocab_layer_norm(prediction_logits)
            prediction_logits = self.vocab_projector(prediction_logits)
            mlm_loss = nn.CrossEntropyLoss()(
                prediction_logits.view(-1, self.vocab_size), mlm_labels.view(-1))
        else:
            mlm_loss = 0.0

        # confounder one-hot encoding
        C_bow = make_bow_vector(C.unsqueeze(1), self.num_labels)

        # Concatenate CLS and Confounder
        inputs = torch.cat((pooled_output, C_bow), 1)

        # Propensity score
        p = self.propensity_cls(inputs)

        if Y is not None:
            p_loss = nn.CrossEntropyLoss()(p.view(-1, 2), T.view(-1).long())
        else:
            p_loss = 0.0

        # Outcome        
        T0 = self.outcome_cls['0'](inputs).clamp(min=0).squeeze(-1)
        T1 = self.outcome_cls['1'](inputs).clamp(min=0).squeeze(-1)

        if Y is not None:
            mask_T0 = (T == 0)
            mask_T1 = (T == 1)
            outcome_loss_T0 = nn.MSELoss()(T0[mask_T0], Y[mask_T0].float()) if mask_T0.any() else 0.0
            outcome_loss_T1 = nn.MSELoss()(T1[mask_T1], Y[mask_T1].float()) if mask_T1.any() else 0.0
            outcome_loss = outcome_loss_T0 + outcome_loss_T1
        else:
            outcome_loss = 0.0

        p = F.softmax(p, dim=1)[:, 1]

        return p, T0, T1, p_loss, outcome_loss, mlm_loss