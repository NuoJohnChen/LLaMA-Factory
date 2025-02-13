import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class AdapterFusion(nn.Module):
    """
    Adapter Fusion layer that dynamically combines multiple adapter outputs using attention.
    """
    def __init__(self, hidden_size, num_adapters):
        super().__init__()
        self.attention = nn.Linear(hidden_size, num_adapters)  # Attention weights for adapters
    
    def forward(self, adapter_outputs):
        """
        Args:
            adapter_outputs (torch.Tensor): Tensor of shape [batch_size, num_adapters, hidden_size].
        Returns:
            torch.Tensor: Fused output of shape [batch_size, hidden_size].
        """
        # Compute attention weights
        attention_weights = torch.softmax(self.attention(adapter_outputs.mean(dim=1)), dim=-1)  # [batch_size, num_adapters]
        
        # Fuse adapter outputs using attention weights
        fused_output = torch.einsum("bn,bnh->bh", attention_weights, adapter_outputs)  # [batch_size, hidden_size]
        return fused_output

def merge_lora_adapters(base_model_path, adapter_paths, output_dir, trained_tokenizer_path, method="simple_merge", weights=None):
    """
    Merge LoRA adapters and reconstruct the full vocabulary mapping.

    Args:
        base_model_path (str): Path to the base model.
        adapter_paths (list): List of paths to multiple LoRA adapters.
        output_dir (str): Path to save the merged model.
        trained_tokenizer_path (str): Path to the tokenizer used during training.
        method (str): Method to merge adapters. Options: "simple_merge", "weight_interpolation", "task_weighted", "layer_wise", "adapter_fusion", "adapter_distillation".
        weights (list): List of weights for weighted merging methods. Required for "weight_interpolation" and "task_weighted".
    """
    # 1. Load the trained tokenizer
    print("Loading trained tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, ignore_mismatched_sizes=True)
    print(f"Trained tokenizer vocab size: {len(tokenizer)}")

    # 2. Load the base model and resize embeddings
    print("Loading base model and resizing embeddings...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        vocab_size=len(tokenizer),  # Force the vocabulary size to match the trained tokenizer
        ignore_mismatched_sizes=True
    )
    model.resize_token_embeddings(len(tokenizer))
    print(f"Base model vocab size after resizing: {model.get_input_embeddings().weight.shape[0]}")

    # 3. Merge LoRA adapters based on the specified method
    if method == "simple_merge":
        # Simple merging using merge_and_unload()
        for i, adapter_path in enumerate(adapter_paths):
            print(f"\nMerging adapter {i+1}/{len(adapter_paths)}: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path, ignore_mismatched_sizes=True)
            model = model.merge_and_unload()  # Merge and unload the adapter
            print(f"Adapter {i+1} merged successfully.")

    elif method == "weight_interpolation":
        # Weight interpolation: Linearly interpolate weights of multiple adapters
        if weights is None or len(weights) != len(adapter_paths):
            raise ValueError("Weights must be provided and match the number of adapters for weight interpolation.")
        
        adapter_models = []
        for adapter_path in adapter_paths:
            adapter_model = PeftModel.from_pretrained(model, adapter_path, ignore_mismatched_sizes=True)
            adapter_models.append(adapter_model)
        
        # Interpolate weights
        for name, param in model.named_parameters():
            if "lora" in name:  # Only interpolate LoRA weights
                param.data = torch.zeros_like(param.data)
                for i, adapter_model in enumerate(adapter_models):
                    param.data += weights[i] * adapter_model.state_dict()[name].data

    elif method == "task_weighted":
        # Task-weighted merging: Weighted sum of adapter weights based on task importance
        if weights is None or len(weights) != len(adapter_paths):
            raise ValueError("Weights must be provided and match the number of adapters for task-weighted merging.")
        
        for i, adapter_path in enumerate(adapter_paths):
            print(f"\nMerging adapter {i+1}/{len(adapter_paths)}: {adapter_path}")
            adapter_model = PeftModel.from_pretrained(model, adapter_path, ignore_mismatched_sizes=True)
            for name, param in model.named_parameters():
                if "lora" in name:  # Only merge LoRA weights
                    param.data += weights[i] * adapter_model.state_dict()[name].data
            print(f"Adapter {i+1} merged with weight {weights[i]}.")

    elif method == "layer_wise":
        # Layer-wise merging: Merge adapters layer by layer
        for i, adapter_path in enumerate(adapter_paths):
            print(f"\nMerging adapter {i+1}/{len(adapter_paths)}: {adapter_path}")
            adapter_model = PeftModel.from_pretrained(model, adapter_path, ignore_mismatched_sizes=True)
            for name, param in model.named_parameters():
                if "lora" in name and f"layer_{i}" in name:  # Merge only specific layers
                    param.data = adapter_model.state_dict()[name].data
            print(f"Adapter {i+1} merged for specific layers.")

    elif method == "adapter_fusion":
        # Adapter Fusion: Dynamically combine adapter outputs using attention
        adapter_models = []
        for adapter_path in adapter_paths:
            adapter_model = PeftModel.from_pretrained(model, adapter_path, ignore_mismatched_sizes=True)
            adapter_models.append(adapter_model)
        
        # Initialize Adapter Fusion layer
        hidden_size = model.config.hidden_size
        fusion_layer = AdapterFusion(hidden_size=hidden_size, num_adapters=len(adapter_paths))
        
        # Example forward pass with fusion
        def forward_with_fusion(input_ids):
            adapter_outputs = []
            for adapter_model in adapter_models:
                output = adapter_model(input_ids=input_ids).last_hidden_state
                adapter_outputs.append(output)
            adapter_outputs = torch.stack(adapter_outputs, dim=1)  # [batch_size, num_adapters, hidden_size]
            fused_output = fusion_layer(adapter_outputs)
            return fused_output
        
        # Save the fusion layer along with the model
        model.fusion_layer = fusion_layer
        print("Adapter Fusion layer added to the model.")

    elif method == "adapter_distillation":
        # Adapter Distillation: Compress multiple adapters into one using knowledge distillation
        teacher_models = []
        for adapter_path in adapter_paths:
            teacher_model = PeftModel.from_pretrained(model, adapter_path, ignore_mismatched_sizes=True)
            teacher_models.append(teacher_model)
        
        # Define distillation loss
        def distillation_loss(student_output, teacher_outputs, temperature=2.0):
            soft_targets = torch.stack([torch.softmax(t / temperature, dim=-1) for t in teacher_outputs], dim=0).mean(dim=0)
            log_probs = torch.log_softmax(student_output / temperature, dim=-1)
            return -(soft_targets * log_probs).sum(dim=-1).mean()
        
        # Distillation training loop (simplified)
        optimizer = torch.optim.Adam(model.parameters())
        for batch in dataloader:  # Replace with your dataloader
            input_ids = batch["input_ids"]
            student_output = model(input_ids=input_ids).logits
            teacher_outputs = [teacher_model(input_ids=input_ids).logits for teacher_model in teacher_models]
            loss = distillation_loss(student_output, teacher_outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print("Adapter Distillation completed.")

    else:
        raise ValueError(f"Unknown merging method: {method}")

    # 4. Save the merged model and tokenizer
    print(f"\nSaving merged model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Merge completed!")

# Example usage
if __name__ == "__main__":
    # Example for simple merge
    merge_lora_adapters(
        base_model_path="/ssd1/models/Qwen2.5-1.5B-Instruct",
        adapter_paths=[
            "/shared/ssd/models/fl-MED-SYN0-CLEVELAND/",
            "/shared/ssd/models/fl-MED-SYN0-HUNGARIAN/",
            "/shared/ssd/models/fl-MED-SYN0-SWITZERLAND/",
            "/shared/ssd/models/fl-MED-SYN0-VA/"
        ],
        output_dir="/shared/ssd/models/Qwen2.5-1.5B-fl-MED-SYN0-25",
        trained_tokenizer_path="/shared/ssd/models/fl-MED-SYN1-CLEVELAND-train/",
        method="simple_merge"
    )
    merge_lora_adapters(
        base_model_path="/ssd1/models/Qwen2.5-1.5B-Instruct",
        adapter_paths=[
            "/shared/ssd/models/fl-MED-SYN1-CLEVELAND-train/",
            "/shared/ssd/models/fl-MED-SYN1-HUNGARIAN/",
            "/shared/ssd/models/fl-MED-SYN1-SWITZERLAND/",
            "/shared/ssd/models/fl-MED-SYN1-VA/"
        ],
        output_dir="/shared/ssd/models/Qwen2.5-1.5B-fl-MED-SYN1-25",
        trained_tokenizer_path="/shared/ssd/models/fl-MED-SYN1-CLEVELAND-train/",
        method="simple_merge"
    )
    # # Example for Adapter Distillation
    # merge_lora_adapters(
    #     base_model_path="/ssd1/models/Qwen2.5-1.5B-Instruct",
    #     adapter_paths=[
    #         "/shared/ssd/models/fl-MED-SYN1-CLEVELAND-train/",
    #         "/shared/ssd/models/fl-MED-SYN1-HUNGARIAN/",
    #         "/shared/ssd/models/fl-MED-SYN1-SWITZERLAND/",
    #         "/shared/ssd/models/fl-MED-SYN1-VA/"
    #     ],
    #     output_dir="/shared/ssd/models/Qwen2.5-1.5B-fl-MED-SYN1-distill",
    #     trained_tokenizer_path="/shared/ssd/models/fl-MED-SYN1-CLEVELAND-train/",
    #     method="adapter_distillation"
    # )