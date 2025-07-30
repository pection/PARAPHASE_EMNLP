import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model,IA3Config,AdaLoraConfig

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
                # "q", "k", "v", "o", "wi_0", "wi_1", "wo"  # T5 decoder layers
for name, param in model.named_parameters():
    print(name)
    if 'SelfAttention.q' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

t_init = 200
t_final = 800
total_steps = t_init + 100 + t_final
# peft_config = AdaLoraConfig(peft_type="ADALORA", task_type="CAUSAL_LM", init_r=8, target_r=4, lora_alpha=32, 
#                             target_modules='.*(self_attn|mlp).*(q_proj|v_proj|k_proj|o_proj|up_proj|gate_proj|down_proj)$',
#                             lora_dropout=0.1, tinit=t_init, tfinal=t_final, deltaT=10, orth_reg_weight=0.1, total_step=total_steps)
peft_config = LoraConfig(
    peft_type="LORA",
    task_type="SEQ_2_SEQ_LM",
    r=4,  # LoRA rank
    lora_alpha=32,
    target_modules=[
        "q", "k", "v", "o", "wi_0", "wi_1", "wo"  # T5 decoder layers
    ],
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
print(f"--------------------")

# trainable_params = []
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)
#         trainable_params.append(name)

# Print trainable parameters
print("\n========== Trainable Parameters ==========\n")


trainable_params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
        trainable_params.append(name)

# Calculate and print trainable parameter ratio
total_params = sum(p.numel() for p in model.parameters())
trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
trainable_percentage = 100 * trainable_param_count / total_params

print("\n========== Parameter Summary ==========")
print(f"Total Parameters: {total_params}")
print(f"Trainable Parameters: {trainable_param_count}")
print(f"Trainable Percentage: {trainable_percentage:.4f}%")
print("========================================\n")

# Optional: Print PEFT summary
model.print_trainable_parameters()
