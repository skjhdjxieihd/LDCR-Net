import numpy as np


def generate_markov_chain(P, start_state, n_steps):
    """
    生成马尔可夫链状态序列

    参数:
    - P: 状态转移概率矩阵 (shape: N x N)
    - start_state: 初始状态 (整数索引)
    - n_steps: 序列长度

    返回:
    - 状态序列 (列表)
    """
    states = [start_state]
    for _ in range(n_steps - 1):
        current_state = states[-1]
        next_state = np.random.choice(len(P), p=P[current_state])
        states.append(next_state)
    return states
