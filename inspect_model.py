import torch
from transformers import AutoModelForCausalLM, AutoConfig

model_id = "ibm-granite/granite-4.0-h-tiny-moe"
print(f"Inspecting {model_id}...")

try:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    print(f"Config: {config}")
    
    # Let's see the layer structure without loading the whole 6GB model
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    print("\n--- Model Architecture ---")
    for name, _ in model.named_modules():
        if "router" in name.lower() or "moe" in name.lower():
            print(f"Found MoE/Router component: {name}")
            
except Exception as e:
    print(f"Error: {e}")
