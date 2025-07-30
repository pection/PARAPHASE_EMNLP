from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch
import torch.nn as nn
import torch.nn.functional as F



class SPLINT_T5(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.label_smoothing = 0.1
        self.entropy_reg_weight = 0.005     # Encourage confident predictions
        self.sparsity_weight = 0.002        # Penalize dense distributions
        self.kl_div_weight = 0.05           # Align with uniform prior
        self.similarity_weight = 0.05       # Optional: similarity alignment (future)

    def compute_entropy(self, probs):
        return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        kwargs.pop("labels", None)
        kwargs.pop("output_hidden_states", None)

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )

        logits = outputs.logits  # [B, T, V]
        loss = None

        if labels is not None:
            labels = labels.to(logits.device)
            vocab_size = logits.size(-1)
            probs = F.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-9)

            # --- 1. CrossEntropy with Label Smoothing ---
            one_hot = torch.zeros_like(logits).scatter(2, labels.unsqueeze(2), 1.0)
            one_hot = one_hot * (1 - self.label_smoothing) + self.label_smoothing / vocab_size
            ce_loss = -(one_hot * log_probs).sum(dim=-1)
            ce_loss = ce_loss.masked_fill(labels == -100, 0.0)
            ce_loss = ce_loss.sum() / (labels != -100).sum()

            # --- 2. Entropy Regularization ---
            entropy = self.compute_entropy(probs)
            entropy = entropy.masked_fill(labels == -100, 0.0)
            entropy_loss = entropy.sum() / (labels != -100).sum()

            # --- 3. Sparsity Regularization (L1 norm) ---
            sparsity_loss = probs.abs().sum() / probs.numel()

            # --- 4. KL Divergence to Uniform Prior ---
            uniform = torch.full_like(probs, 1.0 / vocab_size)
            kl_div = F.kl_div(log_probs, uniform, reduction='none').sum(-1)
            kl_div = kl_div.masked_fill(labels == -100, 0.0)
            kl_div_loss = kl_div.sum() / (labels != -100).sum()

            # --- Total Combined Loss ---
            loss = (
                ce_loss +
                self.entropy_reg_weight * entropy_loss +
                self.sparsity_weight * sparsity_loss +
                self.kl_div_weight * kl_div_loss
            )

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions
        )