import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from pytorch_toolbelt.losses import CrossEntropyFocalLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch.nn import CrossEntropyLoss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='binary', num_classes=None):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Focal Loss based on the specified task type.
        :param inputs: Predictions (logits) from the model.
                       Shape:
                         - binary/multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                         - binary: (batch_size,)
                         - multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size,)
        """
        if self.task_type == 'binary':
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == 'multi-class':
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == 'multi-label':
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'.")

    def binary_focal_loss(self, inputs, targets):
        """ Focal loss for binary classification. """
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_class_focal_loss(self, inputs, targets):
        """ Focal loss for multi-class classification. """
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        probs = F.softmax(inputs, dim=1)

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        """ Focal loss for multi-label classification. """
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, ignore_index=-100):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)

        pred = pred.view(-1, self.cls)
        target = target.view(-1)

        mask = target != self.ignore_index
        target = target[mask]
        pred = pred[mask]

        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.cls - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=-1))
class ZLossCrossEntropy(nn.Module):


    def __init__(self, ignore_index=-100, z_loss=1e-4):
        super().__init__()
        self.ignore_index = ignore_index
        self.z_loss = z_loss

    def forward(self, logits, targets):
        vocab_size = logits.size(-1)

        logits = logits.view(-1, vocab_size)
        targets = targets.view(-1)

        # Mask padding tokens
        mask = targets != self.ignore_index
        logits = logits[mask]
        targets = targets[mask]

        # CrossEntropyLoss
        ce_loss = F.cross_entropy(logits, targets, reduction='mean')

        # Z-Loss: log(Z)^2 where Z = sum(exp(logits))
        if self.z_loss > 0:
            log_z = torch.logsumexp(logits, dim=-1)
            z_term = self.z_loss * (log_z ** 2).mean()
            ce_loss += z_term

        return ce_loss

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = F.relu(pos_dist.pow(2) - neg_dist.pow(2) + self.margin)
        return loss.mean()

class T5WithTripletLoss(T5ForConditionalGeneration):
    def __init__(self, config, margin=1.0):
        super().__init__(config)
        self.triplet_loss_fn = TripletLoss(margin=margin)

    def encode_mean(self, input_ids, attention_mask):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = encoder_outputs.last_hidden_state  # [B, T, D]
        return last_hidden.mean(dim=1)  # [B, D] average pooling

    def forward(
        self,
        anchor_input_ids=None,
        anchor_attention_mask=None,
        positive_decoder_input_ids=None,
        negative_decoder_input_ids=None,
        labels=None,
        **kwargs
    ):
        # Encode Anchor
        anchor_embed = self.encode_mean(anchor_input_ids, anchor_attention_mask)

        # Encode Positive (with teacher forcing or decoder inputs)
        pos_outputs = self(
            input_ids=anchor_input_ids,
            attention_mask=anchor_attention_mask,
            decoder_input_ids=positive_decoder_input_ids,
            use_cache=False,
            output_hidden_states=True,
        )
        positive_embed = pos_outputs.decoder_hidden_states[-1].mean(dim=1)

        # Encode Negative
        neg_outputs = self(
            input_ids=anchor_input_ids,
            attention_mask=anchor_attention_mask,
            decoder_input_ids=negative_decoder_input_ids,
            use_cache=False,
            output_hidden_states=True,
        )
        negative_embed = neg_outputs.decoder_hidden_states[-1].mean(dim=1)

        # Compute Triplet Loss
        loss = self.triplet_loss_fn(anchor_embed, positive_embed, negative_embed)

        # Still return a Seq2SeqLMOutput for compatibility
        return Seq2SeqLMOutput(
            loss=loss,
            logits=pos_outputs.logits,  # just return the logits from the positive path
            past_key_values=pos_outputs.past_key_values,
            decoder_hidden_states=pos_outputs.decoder_hidden_states,
            decoder_attentions=pos_outputs.decoder_attentions,
            cross_attentions=pos_outputs.cross_attentions,
            encoder_last_hidden_state=pos_outputs.encoder_last_hidden_state,
            encoder_hidden_states=pos_outputs.encoder_hidden_states,
            encoder_attentions=pos_outputs.encoder_attentions
        )

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
original_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
inputs = tokenizer("summarize: studies have shown that owning a dog is good for you", return_tensors="pt")
labels = tokenizer("owning a dog is good for yoy.", return_tensors="pt").input_ids
# original_model = T5ForConditionalGeneration.from_pretrained("t5-base")
original_model.eval()


class T5WithCrossEntropyLoss(T5ForConditionalGeneration):
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        lm_logits = outputs.logits  # [B, T, V]

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)

            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
        return {"loss": loss, "logits": lm_logits}



class T5WithCrossEntropyLoss(T5ForConditionalGeneration):
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        lm_logits = outputs.logits  # [B, T, V]

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions
        )

class T5WithCustomLoss(T5ForConditionalGeneration):
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        lm_logits = outputs.logits  # [B, T, V]

        loss = None
        if labels is not None:
            # loss_fct = CrossEntropyFocalLoss(ignore_index=-100)
            loss_fct = ZLossCrossEntropy(ignore_index=-100, z_loss=1e-4)

            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions
        )

ori_cross = T5WithCrossEntropyLoss.from_pretrained("google/flan-t5-large")
ori_cross.eval()

custom_model = T5WithTripletLoss.from_pretrained("google/flan-t5-large")
custom_model.eval()

with torch.no_grad():
    orig_out = original_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels)
    ori_cross_out = ori_cross(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels)
    custom_out = custom_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels)

print(f"{'Original Loss:':<25}{orig_out.loss.item()}")
print(f"{'Original Entropy Loss:':<25}{ori_cross_out.loss.item()}")

# print(f"{'Original Entropy Loss:':<25}{ori_cross_out['loss'].item()}")
print(f"{'Custom Loss:':<25}{custom_out.loss.item()}")

# print(f"{'Custom Loss:':<25}{custom_out['loss'].item()}")



print("Loss equal?", torch.allclose(orig_out.loss, custom_out["loss"], atol=1e-6))
with torch.no_grad():
    generated_ids = original_model.generate(inputs.input_ids, attention_mask=inputs.attention_mask)
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print("Original translate:", output_text)