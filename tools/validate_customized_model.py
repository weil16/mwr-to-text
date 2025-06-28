import sys
from pathlib import Path
from transformers import T5ForConditionalGeneration

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# 加载模型
model = T5ForConditionalGeneration.from_pretrained("custom_temperature_model")

print(f"自定义模型嵌入层大小: {model.get_input_embeddings().weight.shape[0]}")
