from transformers import T5TokenizerFast

# 生成从-5.0到5.0的所有0.1间隔的温度值
temperatures = [f"{temp/10}°C" for temp in range(-103, 86)]  # -5.0°C 到 5.0°C

# 打印前10个和后10个示例
print("生成的温度示例:", temperatures[:10], "...", temperatures[-10:])

# 加载预训练tokenizer
tokenizer = T5TokenizerFast.from_pretrained("google/flan-t5-small")

print(f"原词汇表大小: {len(tokenizer)}")

# 添加温度token
tokenizer.add_tokens(temperatures)

# 添加特殊标记
tokenizer.add_special_tokens(
    {
        "additional_special_tokens": [
            "[TEMP]",
            "[/TEMP]",
            "[AGE]",
            "[/AGE]",
            "[DAY]",
            "[/DAY]",
            "[COUNT]",
            "[/COUNT]",
        ]
    }
)

# 保存自定义tokenizer
tokenizer.save_pretrained("custom_temperature_tokenizer")

# 打印添加后的词汇表大小
print(f"新词汇表大小: {len(tokenizer)}")

# 测试不同温度值的分词效果
test_temps = ["-3.5°C", "0.0°C", "2.7°C", "5.0°C"]

for temp in test_temps:
    text = f"当前温度 [TEMP]{temp}[/TEMP]"
    tokens = tokenizer.tokenize(text)
    print(f"输入: {text}")
    print(f"分词结果: {tokens}")
    print("-" * 50)
