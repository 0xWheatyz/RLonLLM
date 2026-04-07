"""
Model registry and shared GRPO configuration.
Defines HuggingFace model IDs, LoRA targets, and training hyperparameters.
"""

MODELS = {
    "qwen-1.5b": {
        "hf_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "lora_target_modules": ["q_proj", "v_proj"],
        "chat_template": "qwen",
    },
    "deepseek-1.3b": {
        "hf_id": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "lora_target_modules": ["q_proj", "v_proj"],
        "chat_template": "deepseek",
    },
}

# Stretch goal — 3.8B params, ~15GB fp32, 2-3x slower training.
# Only attempt if Qwen + DeepSeek complete with time to spare.
STRETCH_MODELS = {
    "phi-3.5": {
        "hf_id": "microsoft/Phi-3.5-mini-instruct",
        "lora_target_modules": ["qkv_proj"],  # Phi uses fused QKV
        "chat_template": "phi",
    },
}

# Shared GRPO hyperparameters
GRPO_CONFIG = {
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "group_size": 4,           # GRPO: generate N completions per prompt
    "max_new_tokens": 512,
    "learning_rate": 1e-5,
    "kl_coeff": 0.05,          # KL penalty against frozen reference policy
    "num_iterations": 30,
    "temperature": 0.7,
    "dtype": "bfloat16",        # GPU — efficient on A100/H100, supported on A10
    "device": "cuda",
    "gradient_checkpointing": True,  # save VRAM for smaller GPUs (e.g. A10 24GB)
}

ALL_MODELS = {**MODELS, **STRETCH_MODELS}
