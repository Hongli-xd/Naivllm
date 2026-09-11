# /home/selom/cuda/MinivLLM/src/myvllm/engine/rejection_sampler.py
import torch
import numpy as np
from typing import List, Tuple
from myvllm.engine.sequence import Sequence

class RejectionSampler:
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        
    def sample(self, draft_logits: torch.Tensor, target_logits: torch.Tensor, 
               draft_tokens: List[int]) -> Tuple[List[int], int]:
        """
        执行拒绝采样算法
        返回: (接受的令牌列表, 第一个被拒绝的索引)
        """
        accepted_tokens = []
        rejected_idx = len(draft_tokens)  # 默认全部接受
        
        for i, (draft_token, draft_logit, target_logit) in enumerate(
            zip(draft_tokens, draft_logits, target_logits)):
            
            # 计算草稿模型和目标模型的概率
            draft_prob = torch.softmax(draft_logit / self.temperature, dim=-1)[draft_token]
            target_prob = torch.softmax(target_logit / self.temperature, dim=-1)[draft_token]
            
            # 计算接受概率
            acceptance_prob = min(1.0, (target_prob / draft_prob).item())
            
            if np.random.random() < acceptance_prob:
                accepted_tokens.append(draft_token)
            else:
                rejected_idx = i
                # 从残差分布中重新采样
                residual_probs = torch.relu(target_prob - draft_prob)
                if residual_probs.sum() > 0:
                    residual_probs = residual_probs / residual_probs.sum()
                    new_token = torch.multinomial(residual_probs, 1).item()
                    accepted_tokens.append(new_token)
                break
                
        return accepted_tokens, rejected_idx


# 在现有的 block_manager.py 中添加回滚方法
def rollback(self, seq: Sequence, num_tokens: int) -> None:
    """回滚序列的最后num_tokens个令牌"""
    if num_tokens <= 0:
        return
        
    # 移除令牌
    seq.token_ids = seq.token_ids[:-num_tokens]
    seq.num_tokens -= num_tokens
    
    # 计算需要释放的块
    old_num_blocks = len(seq.block_table)
    new_num_blocks = seq.num_blocks
    
    # 释放多余的块
    if new_num_blocks < old_num_blocks:
        for i in range(new_num_blocks, old_num_blocks):
            block_id = seq.block_table.pop()
            self._deallocate_block(block_id)