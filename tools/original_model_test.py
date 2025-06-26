# Standard library imports
from pathlib import Path

# Third-party imports
import sys
from transformers import T5ForConditionalGeneration, T5TokenizerFast

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# 加载预训练的 T5-small 模型
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# 加载对应的分词器
tokenizer = T5TokenizerFast.from_pretrained("t5-small")

# 验证模型嵌入层大小
print(f"模型嵌入层大小: {model.get_input_embeddings().weight.shape[0]}")

# 验证 tokenizer 词汇表大小
print(f"tokenizer 词汇表大小: {len(tokenizer)}")

# 准备英文输入
text = "MWR data: Left breast temps: internal temperature 0: 0.3°C; skin temperature 0: -0.9°C; internal temperature 1: 0.7°C; skin temperature 1: -0.6°C; internal temperature 2: 0.7°C; skin temperature 2: -0.2°C; internal temperature 3: 0.5°C; skin temperature 3: -0.5°C; internal temperature 4: 0.4°C; skin temperature 4: -0.6°C; internal temperature 5: 1.0°C; skin temperature 5: 0.3°C; internal temperature 6: 1.3°C; skin temperature 6: 0.3°C; internal temperature 7: 1.2°C; skin temperature 7: -0.7°C; internal temperature 8: 0.5°C; skin temperature 8: -1.0°C; internal temperature 9: 0.1°C; skin temperature 9: -0.1°C; Right breast temps: internal temperature 0: 1.4°C; skin temperature 0: 0.7°C; internal temperature 1: 1.4°C; skin temperature 1: 0.2°C; internal temperature 2: 0.9°C; skin temperature 2: 0.6°C; internal temperature 3: 1.2°C; skin temperature 3: 0.3°C; internal temperature 4: 1.0°C; skin temperature 4: 0.3°C; internal temperature 5: 1.2°C; skin temperature 5: 0.9°C; internal temperature 6: 1.7°C; skin temperature 6: 1.2°C; internal temperature 7: 1.5°C; skin temperature 7: 1.3°C; internal temperature 8: 1.4°C; skin temperature 8: 0.3°C; internal temperature 9: 0.6°C; skin temperature 9: 1.0°C; Cancer history: True; Breast ops: False; Age: 43; Cycle: The menstrual cycle is 28 days; Day: Day 16 of the menstrual cycle; Pregnancies: 1; Hormonal meds: False"
inputs = tokenizer(text, return_tensors="pt")

# 检查分词结果
tokens = tokenizer.tokenize(text)

# print(f"分词结果: {tokens}")
print(f"Token数量: {len(tokens)}")

# 生成回答
outputs = model.generate(**inputs, max_length=50, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

# 解码回答
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(f"输入: {text}")
print(f"模型回答: {response}")
