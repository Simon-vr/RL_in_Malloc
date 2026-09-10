import numpy as np
import random
from tqdm import tqdm
from config import *
from env import MemoryEnv
from agent import DQNAgent
from utils import *


def main():
    env = MemoryEnv()
    agent = DQNAgent()
    '''训练
    for i_episode in tqdm(range(NUM_EPISODES)):
        state = env.reset()
        for t in range(2000):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action.item())
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            agent.learn()
            if done: 
                break
        agent.update_epsilon()
        if i_episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
    agent.save("agent")
    '''
    agent.load('agent')
    # --- 测试阶段 ---
    print("---Test---")
    
    policies = {
        "DQN Agent": dqn_policy,
        "Best-Fit": best_fit_policy,
        "First-Fit": first_fit_policy,
        "Worst-Fit": worst_fit_policy
    }
    
    results = {name: [] for name in policies}

    for i in range(N_TEST_ROUNDS):
        # 为每一轮生成一个固定的请求序列，保证公平
        request_sequence = []
        alloc_ids = []
        for _ in range(N_REQUESTS_PER_ROUND):
            size = env._generate_request()
            request_sequence.append(('alloc', size))
            alloc_ids.append(len(alloc_ids))
            if random.random() < RELEASE_RATE:
                release_id = random.choice(alloc_ids)
                request_sequence.append(('free', release_id))
                alloc_ids.remove(release_id)

        print(f"\n--- Round {i+1}/{N_TEST_ROUNDS} ---")
        for name, policy in policies.items():
            occupy,duration,frag,hhi = run_simulation(policy, request_sequence, agent if name == "DQN Agent" else None)
            results[name].append((occupy,duration,frag,hhi))
            print(f"  {name}: occupy = {occupy:.4f}, duration = {duration}, fragmentation = {frag:.4f}, HHI = {hhi:.4f}")
    # 打印平均结果
    print("\n---Average Fragmentation Results ---")
    for name, values in results.items():
        avg = np.mean(np.array(values),axis=0)
        occupy = avg[0]
        duration = avg[1]
        frag = avg[2]
        hhi = avg[3]
        print(f"  {name}: occupy = {occupy:.4f}, duration = {duration:.4f}, fragmentation = {frag:.4f}, HHI = {hhi:.4f}")




if __name__ == '__main__':
    main()