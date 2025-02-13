import torch
from transformers import AutoTokenizer, AutoModel

# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载基础模型和 tokenizer
model_name = "/shared/ssd/models/fl-MED-SYN1-CLEVELAND-train/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 将模型移动到 GPU
model.to(device)

# 获取 tokenizer 和模型的词汇表大小
tokenizer_vocab_size = len(tokenizer)
model_vocab_size = model.get_input_embeddings().weight.size(0)

print(f"Tokenizer vocab size: {tokenizer_vocab_size}")
print(f"Model vocab size: {model_vocab_size}")

# 检查是否一致
if tokenizer_vocab_size != model_vocab_size:
    print("Warning: Tokenizer and model vocab sizes do not match!")
else:
    print("Tokenizer and model vocab sizes match.")