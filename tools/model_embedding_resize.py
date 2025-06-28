import sys
from pathlib import Path
from transformers import T5ForConditionalGeneration, T5TokenizerFast

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# 加载模型
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

print(f"原模型嵌入层大小: {model.get_input_embeddings().weight.shape[0]}")

tokenizer = T5TokenizerFast.from_pretrained("custom_temperature_tokenizer")

# 调整嵌入层以适应新token
model.resize_token_embeddings(len(tokenizer))

# 保存调整后的模型
model.save_pretrained("custom_temperature_model")

print(f"新模型嵌入层大小: {model.get_input_embeddings().weight.shape[0]}")
