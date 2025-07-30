from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model and tokenizer
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# JSON-style input
data = {
    "question": "Having lunch seemed like the logically thing to do, this was because he was feeling what?",
    "choices": ["good mood", "hunger", "food", "eating", "anger"]
}

# Build T5-friendly input
input_text = f"et: {data['question']} Choices: " + \
             " ".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(data['choices'])])

# Tokenize input
input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

# Generate output
output_ids = model.generate(
    input_ids,
    max_length=10,
    num_beams=5,
    early_stopping=True
)

# Decode and print
output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Predicted answer:", output)
