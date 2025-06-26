import os
import sys
from pathlib import Path
import sentencepiece as spm
from tokenizers import Tokenizer, pre_tokenizers
from transformers import T5TokenizerFast, PreTrainedTokenizerFast, AddedToken

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# 加载 t5-small 的 tokenizer
base_tokenizer = T5TokenizerFast.from_pretrained("t5-small")

# 定义特殊标记
special_tokens = ["[TEMP]", "[/TEMP]"]

# 确保保存目录存在
os.makedirs("custom_tokenizer", exist_ok=True)

# 准备训练参数（简化版，使用默认ID分配）
train_args = [
    f"--input=tokenizer_retrain_data.txt",
    f"--model_prefix=custom_tokenizer/sp_model",
    f"--vocab_size=57",  # 系统限制的最大值
    f"--user_defined_symbols={','.join(special_tokens)}",
    "--model_type=unigram",
    "--pad_piece=[PAD]",
    "--unk_piece=[UNK]",
    "--bos_piece=[BOS]",
    "--eos_piece=[EOS]",
    "--character_coverage=0.9995",
]

# 训练 SentencePiece 模型
spm.SentencePieceTrainer.train(" ".join(train_args))

# 创建 Hugging Face 兼容的 tokenizer
try:
    custom_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="custom_tokenizer/sp_model.model",
        pad_token="[PAD]",
        eos_token="[EOS]",
        bos_token="[BOS]",
        unk_token="[UNK]",
        additional_special_tokens=special_tokens,
    )
except Exception as e:
    print(f"加载tokenizer失败: {e}")
    print("尝试使用T5TokenizerFast直接加载模型...")
    custom_tokenizer = T5TokenizerFast(
        vocab_file="custom_tokenizer/sp_model.model",
        extra_ids=0,
        additional_special_tokens=special_tokens,
        legacy=False  # 使用新的tokenizer行为
    )

# 保存自定义 tokenizer
custom_tokenizer.save_pretrained("custom_tokenizer")

# 加载自定义tokenizer
custom_tokenizer = T5TokenizerFast.from_pretrained("custom_tokenizer")

# 验证tokenizer大小
print(f"自定义tokenizer大小: {len(custom_tokenizer)}")

# 测试分词效果（更严格的验证）
test_texts = [
    "The temperature is [TEMP]37.5°C[/TEMP]",
    "Normal text without tags",
    "[TEMP]25.3°F[/TEMP] in winter"
]

for text in test_texts:
    print(f"原文: {text}")
    tokens = custom_tokenizer.tokenize(text)
    print(f"分词结果: {tokens}")
    ids = custom_tokenizer.encode(text)
    print(f"ID序列: {ids}")
# 期望输出: ['▁The', '▁temperature', '▁is', '[TEMP]', '37.5', '°C', '[/TEMP]']
