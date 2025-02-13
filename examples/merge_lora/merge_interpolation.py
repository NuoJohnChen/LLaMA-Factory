from transformers import AutoModelForCausalLM, LoraConfig, get_peft_model

# 假设有两个LoRA适配器
model = AutoModelForCausalLM.from_pretrained("base_model")
lora_config_1 = LoraConfig(...)
lora_config_2 = LoraConfig(...)

# 加载两个LoRA适配器
model_1 = get_peft_model(model, lora_config_1)
model_2 = get_peft_model(model, lora_config_2)

# 插值权重
alpha = 0.5  # 插值系数
for (name_1, param_1), (name_2, param_2) in zip(model_1.named_parameters(), model_2.named_parameters()):
    if "lora" in name_1:  # 只对LoRA权重进行插值
        param_1.data = alpha * param_1.data + (1 - alpha) * param_2.data

# 保存插值后的模型
model_1.save_pretrained("interpolated_lora_model")