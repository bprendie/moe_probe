import torch
from transformers import AutoModelForCausalLM, AutoConfig

# Potential Granite 4.0 MoE IDs
model_ids = ["ibm-granite/granite-4.0-tiny-preview", "ibm-granite/granite-4.0-h-tiny-preview"]

for model_id in model_ids:
    print(f"\n--- Checking {model_id} ---")
    try:
        # Load config only to see if it exists
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        print(f"Success! Found config for {model_id}")
        
        # Load the model structure (meta device) to find the router hook point
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            
        print("\nSearching for MoE Router components:")
        for name, module in model.named_modules():
            if "router" in name.lower() or "gate" in name.lower():
                print(f"Found Hook Candidate: {name} ({type(module)})")
        
        # Expert count check
        num_experts = getattr(config, "num_local_experts", "Unknown")
        top_k = getattr(config, "num_experts_per_tok", "Unknown")
        print(f"Architecture: {num_experts} experts, Top-{top_k} active.")
        break
        
    except Exception as e:
        print(f"Could not load {model_id}: {e}")
