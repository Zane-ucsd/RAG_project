from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# ✅ 选择 DeepSeek-V2.5 作为 LLM
model_name = "deepseek-ai/Janus-Pro-1B"

# ✅ 4-bit 量化配置（减少内存占用）
quant_config = BitsAndBytesConfig(
    
    load_in_4bit=True,  # 启用 4-bit 量化
    bnb_4bit_compute_dtype=torch.float16,  # 计算时仍使用 FP16
    bnb_4bit_use_double_quant=True,  # 进一步压缩
)

# ✅ 加载 DeepSeek-V2.5 Tokenizer（添加 trust_remote_code=True）
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# ✅ 加载 DeepSeek-V2.5 模型（4-bit 量化 & trust_remote_code=True）
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map={"": "cpu"},  # 强制在 CPU 运行
    trust_remote_code=True
)

def generate_answer(query, context):
    """使用 DeepSeek-V2.5 生成答案"""
    prompt = f"""
    你是一位专业的AI助手，请基于以下背景信息回答问题：
    {context}

    问题：{query}
    答案：
    """

    # ✅ Tokenization
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

    # ✅ 模型推理
    output = model.generate(**inputs, max_new_tokens=200)

    # ✅ 解码输出
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return answer



if __name__ == "__main__":
    test_query = "小青马是人还是东西？"
    test_context = "杨政熹，周子涵，小青马都是ucsd 计算机专业的学生。"
    print(f"🤖 生成的答案: {generate_answer(test_query, test_context)}")
