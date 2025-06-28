# Standard library imports
from pathlib import Path

# Third-party imports
import sys
import yaml
from transformers import T5ForConditionalGeneration, T5TokenizerFast

# Add project root to path
""" project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root) """
config_path = Path(__file__).parent.parent / "config" / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
# 从本地目录加载模型
# model = T5ForConditionalGeneration.from_pretrained("custom_temperature_model")
model = T5ForConditionalGeneration.from_pretrained(config["model"]["model_path"])
# 验证模型嵌入层大小
print(f"模型嵌入层大小: {model.get_input_embeddings().weight.shape[0]}")
# 应输出与 tokenizer 词汇表大小一致的值（如 32201）

# 加载自定义 tokenizer
tokenizer = T5TokenizerFast.from_pretrained(config["model"]["tokenizer_path"], legacy=True)

# 验证 tokenizer 词汇表大小
print(f"tokenizer 词汇表大小: {len(tokenizer)}")
# 应与模型嵌入层大小一致

# 准备英文输入
text = "MWR data: Left breast temps: internal temperature 0: [TEMP]0.3°C[/TEMP]; skin temperature 0: [TEMP]-0.9°C[/TEMP]; internal temperature 1: [TEMP]0.7°C[/TEMP]; skin temperature 1: [TEMP]-0.6°C[/TEMP]; internal temperature 2: [TEMP]0.7°C[/TEMP]; skin temperature 2: [TEMP]-0.2°C[/TEMP]; internal temperature 3: [TEMP]0.5°C[/TEMP]; skin temperature 3: [TEMP]-0.5°C[/TEMP]; internal temperature 4: [TEMP]0.4°C[/TEMP]; skin temperature 4: [TEMP]-0.6°C[/TEMP]; internal temperature 5: [TEMP]1.0°C[/TEMP]; skin temperature 5: [TEMP]0.3°C[/TEMP]; internal temperature 6: [TEMP]1.3°C[/TEMP]; skin temperature 6: [TEMP]0.3°C[/TEMP]; internal temperature 7: [TEMP]1.2°C[/TEMP]; skin temperature 7: [TEMP]-0.7°C[/TEMP]; internal temperature 8: [TEMP]0.5°C[/TEMP]; skin temperature 8: [TEMP]-1.0°C[/TEMP]; internal temperature 9: [TEMP]0.1°C[/TEMP]; skin temperature 9: [TEMP]-0.1°C[/TEMP]; Right breast temps: internal temperature 0: [TEMP]1.4°C[/TEMP]; skin temperature 0: [TEMP]0.7°C[/TEMP]; internal temperature 1: [TEMP]1.4°C[/TEMP]; skin temperature 1: [TEMP]0.2°C[/TEMP]; internal temperature 2: [TEMP]0.9°C[/TEMP]; skin temperature 2: [TEMP]0.6°C[/TEMP]; internal temperature 3: [TEMP]1.2°C[/TEMP]; skin temperature 3: [TEMP]0.3°C[/TEMP]; internal temperature 4: [TEMP]1.0°C[/TEMP]; skin temperature 4: [TEMP]0.3°C[/TEMP]; internal temperature 5: [TEMP]1.2°C[/TEMP]; skin temperature 5: [TEMP]0.9°C[/TEMP]; internal temperature 6: [TEMP]1.7°C[/TEMP]; skin temperature 6: [TEMP]1.2°C[/TEMP]; internal temperature 7: [TEMP]1.5°C[/TEMP]; skin temperature 7: [TEMP]1.3°C[/TEMP]; internal temperature 8: [TEMP]1.4°C[/TEMP]; skin temperature 8: [TEMP]0.3°C[/TEMP]; internal temperature 9: [TEMP]0.6°C[/TEMP]; skin temperature 9: [TEMP]1.0°C[/TEMP]; Cancer history: True; Breast ops: False; Age: [AGE]43[/AGE]; Cycle: The menstrual cycle is [DAY]28[/DAY] days; Day: Day [DAY]16[/DAY] of the menstrual cycle; Pregnancies: [COUNT]1[/COUNT]; Hormonal meds: False"
text_wrong = "Generate medical assessment from following data: left int 0: [TEMP]0.3°C[/TEMP], right int 0: [TEMP]1.4°C[/TEMP], left sk 0: [TEMP]-0.9°C[/TEMP], right sk 0: [TEMP]0.7°C[/TEMP], left int 1: [TEMP]0.7°C[/TEMP], right int 1: [TEMP]1.4°C[/TEMP], left sk 1: [TEMP]-0.6°C[/TEMP], right sk 1: [TEMP]0.2°C[/TEMP], left int 2: [TEMP]0.7°C[/TEMP], right int 2: [TEMP]0.9°C[/TEMP], left sk 2: [TEMP]-0.2°C[/TEMP], right sk 2: [TEMP]0.6°C[/TEMP], left int 3: [TEMP]0.5°C[/TEMP], right int 3: [TEMP]1.2°C[/TEMP], left sk 3: [TEMP]-0.5°C[/TEMP], right sk 3: [TEMP]0.3°C[/TEMP], left int 4: [TEMP]0.4°C[/TEMP], right int 4: [TEMP]1.0°C[/TEMP], left sk 4: [TEMP]-0.6°C[/TEMP], right sk 4: [TEMP]0.3°C[/TEMP], left int 5: [TEMP]1.0°C[/TEMP], right int 5: [TEMP]1.2°C[/TEMP], left sk 5: [TEMP]0.3°C[/TEMP], right sk 5: [TEMP]0.9°C[/TEMP], left int 6: [TEMP]1.3°C[/TEMP], right int 6: [TEMP]1.7°C[/TEMP], left sk 6: [TEMP]0.3°C[/TEMP], right sk 6: [TEMP]1.2°C[/TEMP], left int 7: [TEMP]1.2°C[/TEMP], right int 7: [TEMP]1.5°C[/TEMP], left sk 7: [TEMP]-0.7°C[/TEMP], right sk 7: [TEMP]1.3°C[/TEMP], left int 8: [TEMP]0.5°C[/TEMP], right int 8: [TEMP]1.4°C[/TEMP], left sk 8: [TEMP]-1.0°C[/TEMP], right sk 8: [TEMP]0.3°C[/TEMP], left int 9: [TEMP]0.1°C[/TEMP], right int 9: [TEMP]0.6°C[/TEMP], left sk 9: [TEMP]-0.1°C[/TEMP], right sk 9: [TEMP]1.0°C[/TEMP], Cancer history: True, Breast ops: False, Age: [AGE]43[/AGE], Cycle: [DAY]28[/DAY], Day: [DAY]16[/DAY], Pregnancies: [COUNT]1[/COUNT], Hormonal meds: False"
inputs = tokenizer(text, return_tensors="pt")
inputs_wrong = tokenizer(text_wrong, return_tensors="pt")
# 处理第一条文本
print("\n处理第一条文本:")
tokens = tokenizer.tokenize(text)
# print(f"分词结果: {tokens}")
print(f"Token数量: {len(tokens)}")
outputs = model.generate(**inputs, max_length=50, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(f"输入: {text}")
print(f"模型回答: {response}")

# 处理第二条文本
print("\n处理第二条文本:")
tokens_wrong = tokenizer.tokenize(text_wrong)
print(f"分词结果: {tokens_wrong}")
print(f"Token数量: {len(tokens_wrong)}")
outputs123 = model.generate(**inputs_wrong, max_length=50, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
response123 = tokenizer.decode(outputs123[0], skip_special_tokens=True)
# print(f"输入: {text_wrong}")
print(f"模型回答: {response123}")
