from config import *
import torch
import numpy as np
import random
class MemoryEnv:
    def __init__(self, request_sequence=None):
        self.memory_size = MEMORY_SIZE
        # 如果提供了请求序列，则用于测试；否则，随机生成用于训练
        self.request_sequence = request_sequence
        self.request_idx = 0
        self.reset()

    def _merge_free_blocks(self):
        self.free_blocks.sort()
        if len(self.free_blocks) < 2:
            return
        merged = []
        current_start, current_size = self.free_blocks[0]
        for i in range(1, len(self.free_blocks)):
            next_start, next_size = self.free_blocks[i]
            if current_start + current_size == next_start:
                current_size += next_size
            else:
                merged.append((current_start, current_size))
                current_start, current_size = next_start, next_size
        merged.append((current_start, current_size))
        self.free_blocks = merged

    def _get_state(self):
        state = np.zeros(STATE_SIZE)
        '''以worstfit为基础做改良
        sorted_free_blocks = sorted(self.free_blocks, key=lambda x: x[1], reverse=True)
        for i in range(min(len(sorted_free_blocks), N_CANDIDATE_BLOCKS)):
            start, size = sorted_free_blocks[i]
            state[i*2] = size / self.memory_size
            state[i*2 + 1] = start / self.memory_size
        '''
        #以firstfit为基础做改良
        sorted_free_blocks = sorted(self.free_blocks)
        i=0
        for j in range(len(sorted_free_blocks)):
            start, size = sorted_free_blocks[j]
            if i*2 >= STATE_SIZE-1:
                break
            if size >= self.current_request_size:
                state[i*2] = size / self.memory_size
                state[i*2 + 1] = start / self.memory_size
                i += 1
        state[-1] = self.current_request_size / self.memory_size
        return torch.FloatTensor(state).unsqueeze(0)

    def reset(self):
        self.allocated_blocks = {}
        self.free_blocks = [(0, self.memory_size)]
        self.next_alloc_id = 0
        self.request_idx = 0
        self.current_request_size = self._generate_request()
        return self._get_state()

    def _generate_request(self):
        # 如果是测试模式，从序列中读取请求
        if self.request_sequence and self.request_idx < len(self.request_sequence):
            req_type, value = self.request_sequence[self.request_idx]
            self.request_idx += 1
            if req_type == 'alloc':
                return value
            elif req_type == 'free':
                self._release_block(value)
                return self._generate_request() # 释放后立即处理下一个请求
        # 否则随机生成
        # normal(80, 20)
        #value = np.random.normal(64, 10)
        #return max(1, int(round(value)))
        #return random.randint(1, 128)
        samples = np.random.lognormal(mean=np.log(32), sigma=0.9)
        samples = np.clip(samples, 1, 512)  
        return samples.astype(int)  

    def _release_block(self, block_id):
        if block_id in self.allocated_blocks:
            r_start, r_size = self.allocated_blocks.pop(block_id)
            self.free_blocks.append((r_start, r_size))
            self._merge_free_blocks()
            
    def is_done(self):
        if self.request_sequence and self.request_idx >= len(self.request_sequence):
            return True # 测试序列结束
        possible_to_allocate = any(size >= self.current_request_size for _, size in self.free_blocks)
        return not possible_to_allocate
        
    def allocate_block(self, chosen_block_start, chosen_block_size):
        self.free_blocks.remove((chosen_block_start, chosen_block_size))
        
        alloc_id = self.next_alloc_id
        self.allocated_blocks[alloc_id] = (chosen_block_start, self.current_request_size)
        self.next_alloc_id += 1
        
        if chosen_block_size > self.current_request_size:
            new_free_start = chosen_block_start + self.current_request_size
            new_free_size = chosen_block_size - self.current_request_size
            self.free_blocks.append((new_free_start, new_free_size))
            self.free_blocks.sort()
        
        # 模拟随机释放（仅在训练时）
        if not self.request_sequence and self.allocated_blocks and random.random() < RELEASE_RATE:
            release_id = random.choice(list(self.allocated_blocks.keys()))
            self._release_block(release_id)
            
        total_free_size = sum(s for _, s in self.free_blocks)
        if total_free_size == 0:
            reward = 0.0
        else :
            reward = sum(
                (size / total_free_size) ** 2
                for _, size in self.free_blocks
            )
        
        self.current_request_size = self._generate_request()
        done = self.is_done()
        next_state = self._get_state()
        
        return next_state, torch.tensor([reward]), torch.tensor([done])

    def step(self, action):
        """执行一个DQN动作"""
        sorted_free_blocks = sorted(self.free_blocks, key=lambda x: x[1], reverse=True)
        
        if action >= len(sorted_free_blocks) or sorted_free_blocks[action][1] < self.current_request_size:
            reward = -1.0 # 惩罚无效选择
            done = self.is_done()
            return self._get_state(), torch.tensor([reward]), torch.tensor([done])

        start, size = sorted_free_blocks[action]
        return self.allocate_block(start, size)