# /home/selom/cuda/MinivLLM/src/myvllm/engine/spec_scheduler.py
import torch
import numpy as np
from collections import deque
from typing import List, Tuple, Optional
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.engine.block_manager import BlockManager
from myvllm.spec_engine.rejection_sampler import RejectionSampler


class SpecScheduler:
    def __init__(self, max_num_sequences: int, max_num_batched_tokens: int, 
                 max_cached_blocks: int, block_size: int, eos: int, 
                 num_speculative_tokens: int = 5):
        # 目标模型块管理器
        self.target_block_manager = BlockManager(max_cached_blocks, block_size)
        # 草稿模型块管理器  
        self.draft_block_manager = BlockManager(max_cached_blocks, block_size)
        
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_sequences = max_num_sequences
        self.block_size = block_size
        self.eos = eos
        self.num_speculative_tokens = num_speculative_tokens
        
        # 序列队列
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        
        # 拒绝采样器
        self.rejection_sampler = RejectionSampler()
        
    def is_finished(self):
        return len(self.waiting) == 0 and len(self.running) == 0
    
    def add_sequence(self, sequence: Sequence):
        self.waiting.append(sequence)
        
    def schedule_draft(self, sequences: List[Sequence]) -> List[Sequence]:
        """调度草稿模型生成投机令牌"""
        # 为每个序列分配草稿KV缓存
        for seq in sequences:
            if self.draft_block_manager.can_allocate(seq):
                self.draft_block_manager.allocate(seq)
                seq.status = SequenceStatus.RUNNING
            else:
                # 如果无法分配，需要抢占
                self.preempt(seq)
                return []
        return sequences
    
    def schedule_verification(self, sequences: List[Sequence], 
                            draft_tokens: List[List[int]], 
                            draft_logits: List[torch.Tensor]) -> Tuple[List[Sequence], List[List[int]], List[torch.Tensor]]:
        """调度目标模型验证投机令牌"""
        # 准备验证输入：将草稿令牌添加到序列中
        verification_sequences = []
        for seq, tokens in zip(sequences, draft_tokens):
            # 创建验证序列，包含原始序列 + 草稿令牌
            verify_seq = self._create_verification_sequence(seq, tokens)
            verification_sequences.append(verify_seq)
            
        return verification_sequences, draft_tokens, draft_logits
    
    def _create_verification_sequence(self, original_seq: Sequence, 
                                    draft_tokens: List[int]) -> Sequence:
        """创建用于验证的序列"""
        # 复制原始序列
        verify_seq = Sequence(
            token_ids=original_seq.token_ids + draft_tokens,
            sampling_params=original_seq.sampling_params
        )
        verify_seq.seq_id = original_seq.seq_id
        verify_seq.num_prompt_tokens = original_seq.num_prompt_tokens
        verify_seq.num_cached_tokens = original_seq.num_cached_tokens
        verify_seq.block_table = original_seq.block_table.copy()
        return verify_seq
    
    def postprocess_speculative(self, sequences: List[Sequence], 
                              draft_tokens: List[List[int]], 
                              draft_logits: List[torch.Tensor],
                              target_logits: List[torch.Tensor]) -> List[int]:
        """处理后处理：拒绝采样和KV缓存回滚"""
        accepted_tokens = []
        
        for i, (seq, draft_toks, draft_logs, target_logs) in enumerate(
            zip(sequences, draft_tokens, draft_logits, target_logits)):
            
            # 执行拒绝采样
            accepted, rejected_idx = self.rejection_sampler.sample(
                draft_logs, target_logs, draft_toks
            )
            
            # 更新序列：只接受被采样的令牌
            for token in accepted:
                seq.append_token(token)
                
            # 回滚草稿模型KV缓存到接受的位置
            if rejected_idx < len(draft_toks):
                rollback_tokens = len(draft_toks) - rejected_idx
                self.draft_block_manager.rollback(seq, rollback_tokens)
                
            accepted_tokens.append(len(accepted))
            
        return accepted_tokens
    
    def preempt(self, seq: Sequence) -> None:
        """抢占序列"""
        self.draft_block_manager.deallocate(seq)
        self.target_block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)

# /home/selom/cuda/MinivLLM/src/myvllm/engine/rejection_sampler.py
import torch
import numpy as np
from typing import List, Tuple


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