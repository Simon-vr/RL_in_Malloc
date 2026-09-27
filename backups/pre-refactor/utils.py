from env import MemoryEnv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from config import *

def first_fit_policy(free_blocks, request_size):
    for start, size in sorted(free_blocks): # 按地址排序
        if size >= request_size:
            return start, size
    return None

def best_fit_policy(free_blocks, request_size):
    best_block = None
    min_waste = float('inf')
    for start, size in free_blocks:
        if size >= request_size:
            waste = size - request_size
            if waste < min_waste:
                min_waste = waste
                best_block = (start, size)
    return best_block

def worst_fit_policy(free_blocks, request_size):
    worst_block = None
    max_size = -1
    for start, size in free_blocks:
        if size >= request_size and size > max_size:
            max_size = size
            worst_block = (start, size)
    return worst_block

def dqn_policy(agent, state, free_blocks):
    sorted_free_blocks = sorted(free_blocks, key=lambda x: x[1], reverse=True)
    action = agent.select_action(state, is_test=True).item()
    if action < len(sorted_free_blocks):
        return sorted_free_blocks[action]
    return None

def run_simulation(policy_func, request_sequence, agent=None):
    """通用模拟器"""
    env = MemoryEnv(request_sequence=request_sequence.copy())
    state = env.reset()
    done = False
    duration = 0
    while not done:
        duration += 1
        request_size = env.current_request_size
        
        if policy_func.__name__ == 'dqn_policy':
            chosen_block = policy_func(agent, state, env.free_blocks)
        else:
            chosen_block = policy_func(env.free_blocks, request_size)
        
        if chosen_block and chosen_block[1] >= request_size:
            state, _, done_tensor = env.allocate_block(chosen_block[0], chosen_block[1])
            done = done_tensor.item()
        else:
            # 无法分配，终止
            break
            
    # 计算最终的分配率
    total_free_size = sum(s for _, s in env.free_blocks)
    occupy  = (MEMORY_SIZE - total_free_size)/MEMORY_SIZE
    if total_free_size == 0:
        max_free_size = 0
        fragmentation = 0
        hhi = 0
    else:
        max_free_size = max(s for _, s in env.free_blocks) 
        fragmentation = 1 - (max_free_size / total_free_size)
        hhi = 1 - sum(s**2 for _, s in env.free_blocks)/total_free_size**2

    #occupy：越趋近于1利用率越高
    #duration：越大满足的服务次数越高
    #fragmentation：越趋近于1碎片化越高
    #HHI:越趋近于1集中程度越高
    return occupy, duration, fragmentation, hhi