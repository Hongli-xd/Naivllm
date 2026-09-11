# /home/selom/cuda/MinivLLM/spec_main.py
import sys
from pathlib import Path
from transformers import AutoTokenizer
from myvllm.spec_engine.spec_llm_engine import SpecLLMEngine
from myvllm.sampling_parameters import SamplingParams

def main():

    """
    主函数，初始化配置、模型和执行投机解码
    """
    # 配置字典，包含模型和推理的各种参数
    config = {
        'max_num_sequences': 16,          # 最大序列数量
        'max_num_batched_tokens': 1024,   # 最大批处理token数
        'max_cached_blocks': 1024,        # 最大缓存块数
        'block_size': 256,                # 块大小
        'world_size': 1,                  # 世界大小（分布式训练用）
        'draft_model': 'Qwen/Qwen3-0.6B',  # 草稿模型，用于投机解码
        'target_model': 'Qwen/Qwen3-0.6B',  # 目标模型，最终生成结果
        'num_speculative_tokens': 5,  # 投机令牌数量
        'enforce_eager': True,           # 强制急切执行
        'vocab_size': 151936,
        'hidden_size': 1024,
        'num_heads': 16,
        'head_dim': 128,
        'num_kv_heads': 8,
        'intermediate_size': 3072,
        'num_layers': 28,
        'tie_word_embeddings': True,
        'base': 1000000,
        'rms_norm_epsilon': 1e-6,
        'qkv_bias': False,
        'scale': 1,
        'max_position': 32768,
        'ffn_bias': False,
        'max_num_batch_tokens': 4096,
        'max_model_length': 128,
        'gpu_memory_utilization': 0.9,
        'eos': 151645,
    }
    
    # 初始化投机解码引擎
    engine = SpecLLMEngine(config)
    tokenizer = AutoTokenizer.from_pretrained(config['target_model'])
    
    # 测试提示
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
        "give me your opinion on AI impact",
    ]
    
    # 应用聊天模板
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    
    # 采样参数
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=256,
        max_model_length=128
    )
    
    # 执行投机解码
    print("Running speculative decoding...")
    outputs = engine.generate(formatted_prompts, sampling_params)
    
    # 打印结果
    for prompt, output in zip(formatted_prompts, outputs['text']):
        print(f"\nPrompt: {prompt}")
        print(f"Completion: {output}")

if __name__ == "__main__":
    main()
