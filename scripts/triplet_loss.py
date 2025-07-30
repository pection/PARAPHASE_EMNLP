import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration

class T5WithTripletLoss(T5ForConditionalGeneration):
    def __init__(self, margin=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def get_embedding(self, input_ids, attention_mask):
        """
        Returns pooled encoder outputs as fixed-size embeddings.
        """
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = encoder_outputs.last_hidden_state.mean(dim=1)  # average pooling across tokens
        return pooled

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                pos_ids=None, pos_mask=None, neg_ids=None, neg_mask=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        lm_logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]

        # CrossEntropy loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        # Triplet loss on encoder embeddings
        if pos_ids is not None and neg_ids is not None:
            anchor = self.get_embedding(input_ids, attention_mask)
            positive = self.get_embedding(pos_ids, pos_mask)
            negative = self.get_embedding(neg_ids, neg_mask)
            triplet = self.triplet_loss(anchor, positive, negative)
            loss = loss + triplet if loss is not None else triplet

        return {"loss": loss, "logits": lm_logits}

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5WithTripletLoss.from_pretrained("t5-base")

# Input: anchor sentence
anchor_text = "summarize: studies have shown that owning a dog is good for your health"
anchor = tokenizer(anchor_text, return_tensors="pt", padding=True)

# Positive: good summary
positive_text = "owning a dog improves your health"
positive = tokenizer(positive_text, return_tensors="pt", padding=True)

# Negative: irrelevant or wrong summary
negative_text = "dogs can be trained to fetch the ball"
negative = tokenizer(negative_text, return_tensors="pt", padding=True)

# Labels for generation loss
label_text = "owning a dog is good for your health"
labels = tokenizer(label_text, return_tensors="pt", padding=True).input_ids

# Padding mask
anchor_mask = anchor["attention_mask"]
positive_mask = positive["attention_mask"]
negative_mask = negative["attention_mask"]

# Forward pass
model.eval()
with torch.no_grad():
    outputs = model(
        input_ids=anchor["input_ids"],
        attention_mask=anchor_mask,
        labels=labels,
        pos_ids=positive["input_ids"],
        pos_mask=positive_mask,
        neg_ids=negative["input_ids"],
        neg_mask=negative_mask
    )

print("Total Loss (CE + Triplet):", outputs["loss"].item())

# Optional: generate summary
generated_ids = model.generate(anchor["input_ids"], attention_mask=anchor_mask)
summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("Generated Summary:", summary)
