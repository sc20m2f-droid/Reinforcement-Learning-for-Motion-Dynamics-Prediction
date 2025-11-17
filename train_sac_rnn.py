import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader
from collections import deque
import random
import time
from tqdm import tqdm # 引入进度条

# =============================================================================
# --- 实验配置 (EXPERIMENT CONFIGURATION) ---
# =============================================================================
# 路径 4: 使用 RNN (LSTM)
CONFIG = {
    # --- 实验选择 ---
    'EXPERIMENT_NAME': '4_SAC_RNN', # 实验名称 (包含 'RNN' 会切换模型)

    # --- 模型架构 ---
    'MODEL_TYPE': 'RNN',      # <-- 关键: 自动设置为 RNN
    'RNN_SEQUENCE_LENGTH': 10, # RNN 回看的步数

    # --- 奖励工程 ---
    'USE_JERK_PENALTY': False,
    'JERK_WEIGHT': 0.1,      

    # --- 从演示中学习 ---
    'PREFILL_BUFFER': False, 
    
    # --- 研究设计 ---
    'ROBUSTNESS_NOISE_LEVEL': 0.0, # (标准测试)
    'GENERALIZATION_TRAIN_IDS': [0, 1, 2, 3, 4], # 使用所有数据训练
    'GENERALIZATION_TEST_IDS': [4],             # 在 ID 4 上测试
    
    # --- 训练超参数 ---
    'MAX_EPISODES': 800,
    'BUFFER_SIZE': 50000,
    'BATCH_SIZE': 256,
    'HIDDEN_DIM1': 256,
    'HIDDEN_DIM2': 128,
    'GAMMA': 0.99,
    'TAU': 0.005,
    'ACTOR_LR': 1e-4,
    'CRITIC_LR': 1e-4,
    'ALPHA_LR': 3e-4,
    'L2_LAMBDA': 1e-4,
    'MAX_ACTION': 1.0,
}
# =============================================================================

# 根据实验名称自动更新 CONFIG
if 'RNN' in CONFIG['EXPERIMENT_NAME']:
    CONFIG['MODEL_TYPE'] = 'RNN'
if 'SACfD' in CONFIG['EXPERIMENT_NAME']:
    CONFIG['PREFILL_BUFFER'] = True

# 设置 PyTorch 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Running Experiment: {CONFIG['EXPERIMENT_NAME']}")
print(f"Using Model Type: {CONFIG['MODEL_TYPE']}")
if CONFIG['PREFILL_BUFFER']:
    print("Learning from Demonstrations (Prefill) is ENABLED.")

# 状态维度
STATE_DIM_MLP = 3 * 3 * 4  # 3 关节 * 3 特征 * 4 帧
STATE_DIM_RNN = 9          # 3 关节 * 3 特征 (Jnt, Vel, Acc)
ACTION_DIM = 3

#################################################################
# 数据处理与辅助函数
#################################################################

def load_and_preprocess_data(csv_path, train_ids, test_ids):
    """加载、预处理、并根据 ID 拆分数据"""
    print(f"Loading data from {csv_path}...")
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found at '{csv_path}'.")
        return None, None, None
    
    # (预处理代码与之前相同)
    for joint in ['joint1', 'joint2', 'joint3']:
        mu = data[joint].mean()
        sigma = data[joint].std()
        outlier_idx = np.abs(data[joint] - mu) > 3 * sigma
        if outlier_idx.sum() > 0:
            for i in np.where(outlier_idx)[0]:
                if i == 0: data.loc[i, joint] = data.loc[i + 1, joint]
                elif i == len(data) - 1: data.loc[i, joint] = data.loc[i - 1, joint]
                else: data.loc[i, joint] = (data.loc[i - 1, joint] + data.loc[i + 1, joint]) / 2
    data.interpolate(method='linear', inplace=True)
    
    data['joint1'] = kalman_filter(data['joint1'].values, 0.01, 0.1)
    data['joint2'] = kalman_filter(data['joint2'].values, 0.01, 0.12)
    data['joint3'] = kalman_filter(data['joint3'].values, 0.01, 0.09)
    
    dt = 1.0
    data['velocity1'] = np.concatenate(([0], np.diff(data['joint1']) / dt))
    data['velocity2'] = np.concatenate(([0], np.diff(data['joint2']) / dt))
    data['velocity3'] = np.concatenate(([0], np.diff(data['joint3']) / dt))
    data['acceleration1'] = np.concatenate(([0], np.diff(data['velocity1']) / dt))
    data['acceleration2'] = np.concatenate(([0], np.diff(data['velocity2']) / dt))
    data['acceleration3'] = np.concatenate(([0], np.diff(data['velocity3']) / dt))

    feature_names = ['joint1', 'joint2', 'joint3', 'velocity1', 'velocity2', 'velocity3',
                     'acceleration1', 'acceleration2', 'acceleration3']
    data_info = {}
    
    # 创建一个全零的 'jerk' 列，稍后计算
    data['jerk1'] = 0.0
    data['jerk2'] = 0.0
    data['jerk3'] = 0.0

    # 归一化
    for name in feature_names:
        mu = data[name].mean()
        sigma = data[name].std()
        if sigma < 1e-6: sigma = 1.0
        data[name] = (data[name] - mu) / sigma
        data_info[name] = {'mu': mu, 'sigma': sigma}

    # 计算归一化后的 Jerk
    data['jerk1'] = np.concatenate(([0], np.diff(data['acceleration1']) / dt))
    data['jerk2'] = np.concatenate(([0], np.diff(data['acceleration2']) / dt))
    data['jerk3'] = np.concatenate(([0], np.diff(data['acceleration3']) / dt))

    # 根据 ID 拆分数据
    train_data = data[data['sample_id'].isin(train_ids)].reset_index(drop=True)
    test_data = data[data['sample_id'].isin(test_ids)].reset_index(drop=True)
    
    print(f"Data loaded: {len(train_data)} training samples, {len(test_data)} testing samples.")
    return train_data, test_data, data_info

def kalman_filter(data, Q, R):
    n = len(data)
    filtered = np.zeros(n)
    x_hat = data[0]; P = 1.0; filtered[0] = x_hat
    for k in range(1, n):
        x_hat_minus = x_hat; P_minus = P + Q
        K = P_minus / (P_minus + R)
        x_hat = x_hat_minus + K * (data[k] - x_hat_minus)
        P = (1 - K) * P_minus; filtered[k] = x_hat
    return filtered

def calculate_reward(predicted, actual, pred_jerk, actual_jerk, 
                     prev_predicted, prev_actual, is_terminal, 
                     use_jerk=False, jerk_weight=0.1):
    mse = np.mean((predicted - actual)**2)
    reward = -mse
    
    if prev_predicted is not None and prev_actual is not None:
        pred_velocity = predicted - prev_predicted
        actual_velocity = actual - prev_actual
        velocity_mse = np.mean((pred_velocity - actual_velocity)**2)
        reward -= 0.2 * velocity_mse
        
        actual_direction = np.sign(actual - prev_actual)
        pred_direction = np.sign(predicted - prev_predicted)
        direction_reward = np.sum(actual_direction == pred_direction) / len(actual_direction)
        reward += 0.1 * direction_reward
    
    # --- 新增: Jerk 惩罚 ---
    if use_jerk:
        jerk_mse = np.mean((pred_jerk - actual_jerk)**2)
        reward -= jerk_weight * jerk_mse
        
    if is_terminal:
        reward = reward * 2.0
    
    reward = reward / (1 + abs(reward))
    return reward

def polyak_update(target, source, tau):
    with torch.no_grad():
        for target_param, source_param in zip(target.state_dict().values(), source.state_dict().values()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

# --- 状态管理 (MLP vs RNN) ---

def get_state_MLP(data, t):
    """获取 4 帧堆叠的 MLP 状态"""
    state = np.zeros(STATE_DIM_MLP)
    for i in range(4):
        idx = max(0, t - i) # 从 t 往回取
        row = data.iloc[idx]
        offset = (3 - i) * 9 # 将 t 放在最后
        state[offset:offset+9] = row[['joint1', 'joint2', 'joint3',
                                     'velocity1', 'velocity2', 'velocity3',
                                     'acceleration1', 'acceleration2', 'acceleration3']].values
    return state

def get_state_RNN(data, t):
    """获取 RNN 的当前 9-dim 状态"""
    row = data.iloc[t]
    return row[['joint1', 'joint2', 'joint3',
                'velocity1', 'velocity2', 'velocity3',
                'acceleration1', 'acceleration2', 'acceleration3']].values

def get_jerk_RNN(data, t):
    """获取 RNN 的当前 Jerk (用于奖励)"""
    row = data.iloc[t]
    return row[['jerk1', 'jerk2', 'jerk3']].values

def get_action_from_row(row):
    """从数据行中获取真实动作 (目标)"""
    return row[['joint1', 'joint2', 'joint3']].values
    
def get_current_features(state_vec_9d):
    """从 9-dim 状态向量中提取特征"""
    pos = state_vec_9d[0:3]
    vel = state_vec_9d[3:6]
    acc = state_vec_9d[6:9]
    return pos, vel, acc

#################################################################
# 标准经验回放缓冲区 (MLP)
#################################################################

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        if len(self.buffer) < self.batch_size:
            return None
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(device).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def prefill(self, data):
        """使用专家数据预填充缓冲区 (SACfD)"""
        print(f"Prefilling buffer with {len(data)-4} transitions...")
        for t in tqdm(range(3, len(data) - 1)): # 至少需要 4 帧
            state = get_state_MLP(data, t)
            next_state = get_state_MLP(data, t + 1)
            
            row = data.iloc[t+1]
            action = get_action_from_row(row)
            
            # 這裡我們用一個簡化的獎勵 (1.0)
            reward = 1.0 
            done = False # 假设演示数据都是连续的
            self.add(state, action, reward, next_state, done)
        print("Buffer prefill complete.")

#################################################################
# 序列经验回放缓冲区 (RNN)
#################################################################

class ReplayBuffer_RNN:
    def __init__(self, buffer_size, batch_size, seq_len):
        self.buffer = deque(maxlen=buffer_size) # 缓冲区存储 (s, a, r, s', d)
        self.batch_size = batch_size
        self.seq_len = seq_len
        print(f"RNN Replay Buffer initialized with SeqLen: {seq_len}")

    def add(self, state, action, reward, next_state, done):
        # 存储 9-dim 的状态
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        if len(self.buffer) < self.batch_size * self.seq_len:
            return None

        states_seq, actions_seq, rewards_seq, next_states_seq, dones_seq = [], [], [], [], []
        
        for _ in range(self.batch_size):
            # 随机选择一个序列的 *结束* 点
            idx = random.randint(self.seq_len - 1, len(self.buffer) - 1)
            
            # 检查序列中是否有 episode 结束
            has_done = any(self.buffer[i][4] for i in range(idx - self.seq_len + 1, idx))
            
            # 如果序列无效 (跨越了 episode)，则重新采样
            # (一个简单的拒绝采样)
            while has_done:
                idx = random.randint(self.seq_len - 1, len(self.buffer) - 1)
                has_done = any(self.buffer[i][4] for i in range(idx - self.seq_len + 1, idx))

            # 提取有效序列
            sequence = [self.buffer[i] for i in range(idx - self.seq_len + 1, idx + 1)]
            s, a, r, s_next, d = zip(*sequence)
            
            states_seq.append(s)
            actions_seq.append(a)
            rewards_seq.append(r)
            next_states_seq.append(s_next)
            dones_seq.append(d)

        # 转换为 (batch_size, seq_len, dim)
        return (
            torch.tensor(np.array(states_seq), dtype=torch.float32).to(device),
            torch.tensor(np.array(actions_seq), dtype=torch.float32).to(device),
            torch.tensor(np.array(rewards_seq), dtype=torch.float32).to(device).unsqueeze(-1),
            torch.tensor(np.array(next_states_seq), dtype=torch.float32).to(device),
            torch.tensor(np.array(dones_seq), dtype=torch.float32).to(device).unsqueeze(-1)
        )

    def __len__(self):
        return len(self.buffer)

    def prefill(self, data):
        """使用专家数据预填充 (RNN)"""
        print(f"Prefilling RNN buffer with {len(data)-1} transitions...")
        for t in tqdm(range(len(data) - 1)):
            state = get_state_RNN(data, t)
            next_state = get_state_RNN(data, t + 1)
            row = data.iloc[t+1]
            action = get_action_from_row(row)
            reward = 1.0
            done = False # TODO: 需要检测 sample_id 或 timestep 的变化
            self.add(state, action, reward, next_state, done)
        print("RNN Buffer prefill complete.")

#################################################################
# 网络定义 (MLP)
#################################################################

class StochasticActor_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, max_action=1.0):
        super(StochasticActor_MLP, self).__init__()
        self.bn_eps = 1e-6
        self.bn_momentum = 0.1
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1, eps=self.bn_eps, momentum=self.bn_momentum)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2, eps=self.bn_eps, momentum=self.bn_momentum)
        self.mean_layer = nn.Linear(hidden_dim2, output_dim)
        self.log_std_layer = nn.Linear(hidden_dim2, output_dim)
        self.max_action = max_action
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.layer1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.mean_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.log_std_layer.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        h1 = torch.tanh(self.bn1(self.layer1(x)))
        h2 = torch.tanh(self.bn2(self.layer2(h1)))
        mean = self.mean_layer(h2)
        log_std = self.log_std_layer(h2)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        z = dist.rsample()
        action = self.max_action * torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(self.max_action * (1 - torch.tanh(z).pow(2)) + 1e-6)
        log_prob = log_prob.sum(axis=-1, keepdim=True)
        return action, log_prob

class Critic_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim=1):
        super(Critic_MLP, self).__init__()
        self.bn_eps = 1e-6
        self.bn_momentum = 0.1
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1, eps=self.bn_eps, momentum=self.bn_momentum)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2, eps=self.bn_eps, momentum=self.bn_momentum)
        self.layer3 = nn.Linear(hidden_dim2, output_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.layer1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer3.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        h1 = torch.tanh(self.bn1(self.layer1(x)))
        h2 = torch.tanh(self.bn2(self.layer2(h1)))
        q_value = self.layer3(h2)
        return q_value

#################################################################
# 网络定义 (RNN)
#################################################################

class StochasticActor_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, rnn_hidden_dim, output_dim, max_action=1.0):
        super(StochasticActor_RNN, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.max_action = max_action
        
        # 预处理层
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # LSTM 层
        self.lstm = nn.LSTM(hidden_dim, rnn_hidden_dim, batch_first=True)
        
        # 输出层
        self.layer2 = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.log_std_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h=None, c=None):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # BatchNorm 需要 (N, C) 或 (N, C, L)
        # 我们将 batch 和 seq 合并
        x = x.view(batch_size * seq_len, -1)
        h1 = F.relu(self.bn1(self.layer1(x)))
        
        # 恢复序列形状
        h1_rnn = h1.view(batch_size, seq_len, -1)
        
        # LSTM
        if h is None or c is None:
            lstm_out, (h_n, c_n) = self.lstm(h1_rnn)
        else:
            lstm_out, (h_n, c_n) = self.lstm(h1_rnn, (h, c))
            
        # 我们只关心最后一个时间步的输出 (用于决策) 或所有输出 (用于训练)
        # lstm_out shape: (batch_size, seq_len, rnn_hidden_dim)
        
        # 同样合并 batch 和 seq 以便 BatchNorm
        lstm_out_flat = lstm_out.contiguous().view(batch_size * seq_len, -1)
        h2 = F.relu(self.bn2(self.layer2(lstm_out_flat)))
        
        mean = self.mean_layer(h2).view(batch_size, seq_len, -1)
        log_std = self.log_std_layer(h2).view(batch_size, seq_len, -1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        
        return mean, std, (h_n, c_n) # 返回均值、标准差和隐藏状态

    def sample(self, state_seq, h=None, c=None):
        mean, std, (h_n, c_n) = self.forward(state_seq, h, c)
        
        # 仅在最后一步采样 (用于决策) 或在每一步采样 (用于训练)
        dist = Normal(mean, std)
        z = dist.rsample()
        action = self.max_action * torch.tanh(z)
        
        log_prob = dist.log_prob(z) - torch.log(self.max_action * (1 - torch.tanh(z).pow(2)) + 1e-6)
        log_prob = log_prob.sum(axis=-1, keepdim=True)
        
        # 为了决策, 我们只返回最后一步
        # 为了训练, 我们返回整个序列
        return action, log_prob, (h_n, c_n)

class Critic_RNN(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim, rnn_hidden_dim, output_dim=1):
        super(Critic_RNN, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim

        # 状态处理
        self.state_layer1 = nn.Linear(input_dim, hidden_dim)
        self.state_bn1 = nn.BatchNorm1d(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, rnn_hidden_dim, batch_first=True)
        
        # 动作处理
        self.action_layer1 = nn.Linear(action_dim, hidden_dim)
        
        # 组合
        self.combo_layer1 = nn.Linear(rnn_hidden_dim + hidden_dim, hidden_dim)
        self.combo_bn1 = nn.BatchNorm1d(hidden_dim)
        self.combo_layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state_seq, action_seq, h=None, c=None):
        # state_seq shape: (batch_size, seq_len, input_dim)
        # action_seq shape: (batch_size, seq_len, action_dim)
        batch_size, seq_len, _ = state_seq.shape
        
        # 1. 处理状态序列
        state_flat = state_seq.view(batch_size * seq_len, -1)
        s_h1 = F.relu(self.state_bn1(self.state_layer1(state_flat)))
        s_h1_rnn = s_h1.view(batch_size, seq_len, -1)
        
        if h is None or c is None:
            lstm_out, _ = self.lstm(s_h1_rnn)
        else:
            lstm_out, _ = self.lstm(s_h1_rnn, (h, c))
            
        # lstm_out shape: (batch_size, seq_len, rnn_hidden_dim)
        
        # 2. 处理动作序列
        action_flat = action_seq.view(batch_size * seq_len, -1)
        a_h1 = F.relu(self.action_layer1(action_flat))
        
        # 3. 组合
        # 我们需要将 (N, L, H_rnn) 和 (N*L, H_act) 组合
        # 将 lstm_out 也 flat
        lstm_out_flat = lstm_out.contiguous().view(batch_size * seq_len, -1)
        
        combined = torch.cat([lstm_out_flat, a_h1], dim=1)
        
        c_h1 = F.relu(self.combo_bn1(self.combo_layer1(combined)))
        q_value = self.combo_layer2(c_h1)
        
        # 恢复序列形状
        return q_value.view(batch_size, seq_len, 1)

#################################################################
#  SAC Agent (同时支持 MLP 和 RNN)
#################################################################

class SAC_Agent:
    def __init__(self, config):
        self.config = config
        self.model_type = config['MODEL_TYPE']
        self.gamma = config['GAMMA']
        self.tau = config['TAU']
        self.max_action = config['MAX_ACTION']
        
        state_dim = STATE_DIM_RNN if self.model_type == 'RNN' else STATE_DIM_MLP
        h1, h2 = config['HIDDEN_DIM1'], config['HIDDEN_DIM2']
        
        if self.model_type == 'RNN':
            rnn_h = h2 # 使用 h2 作为 rnn 隐藏维度
            self.actor = StochasticActor_RNN(state_dim, h1, rnn_h, ACTION_DIM, self.max_action).to(device)
            self.critic_1 = Critic_RNN(state_dim, ACTION_DIM, h1, rnn_h, 1).to(device)
            self.critic_2 = Critic_RNN(state_dim, ACTION_DIM, h1, rnn_h, 1).to(device)
            self.target_critic_1 = Critic_RNN(state_dim, ACTION_DIM, h1, rnn_h, 1).to(device)
            self.target_critic_2 = Critic_RNN(state_dim, ACTION_DIM, h1, rnn_h, 1).to(device)
        else:
            self.actor = StochasticActor_MLP(state_dim, h1, h2, ACTION_DIM, self.max_action).to(device)
            self.critic_1 = Critic_MLP(state_dim + ACTION_DIM, h1, h2, 1).to(device)
            self.critic_2 = Critic_MLP(state_dim + ACTION_DIM, h1, h2, 1).to(device)
            self.target_critic_1 = Critic_MLP(state_dim + ACTION_DIM, h1, h2, 1).to(device)
            self.target_critic_2 = Critic_MLP(state_dim + ACTION_DIM, h1, h2, 1).to(device)
            
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        
        self.actor_optimizer = optim.SGD(self.actor.parameters(), lr=config['ACTOR_LR'], momentum=0.9, weight_decay=config['L2_LAMBDA'])
        self.critic_optimizer = optim.SGD(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), 
            lr=config['CRITIC_LR'], momentum=0.9, weight_decay=config['L2_LAMBDA']
        )
        
        self.target_entropy = -torch.prod(torch.Tensor(ACTION_DIM).to(device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.SGD([self.log_alpha], lr=config['ALPHA_LR'], momentum=0.9)
        self.alpha = self.log_alpha.exp().item()
        
        # RNN 隐藏状态
        self.actor_h = None
        self.actor_c = None

    def select_action(self, state, deterministic=False):
        # state 应为 (1, state_dim) or (1, seq_len, state_dim)
        if self.model_type == 'RNN':
            if state.ndim == 2: # (seq_len, dim)
                state_tensor = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0) # (1, seq_len, dim)
            elif state.ndim == 1: # (dim)
                state_tensor = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0) # (1, 1, dim)
            else:
                state_tensor = state
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0) # (1, dim)
            
        self.actor.eval()
        with torch.no_grad():
            if self.model_type == 'RNN':
                # RNN 决策: state_tensor is (1, 1, dim)
                mean, std, (h_n, c_n) = self.actor(state_tensor, self.actor_h, self.actor_c)
                self.actor_h, self.actor_c = h_n, c_n # 保存隐藏状态
                
                # 仅使用最后一步的动作
                mean, std = mean[:, -1, :], std[:, -1, :] # (1, dim)
            else:
                # MLP 决策
                mean, std = self.actor(state_tensor)

            if deterministic:
                action = self.max_action * torch.tanh(mean)
            else:
                dist = Normal(mean, std)
                z = dist.sample() # SAC 评估时也采样
                action = self.max_action * torch.tanh(z)
                
        self.actor.train()
        return action.detach().cpu().numpy()[0]

    def reset_rnn_hidden_state(self):
        self.actor_h, self.actor_c = None, None

    def update(self, batch):
        if self.model_type == 'RNN':
            self.update_rnn(batch)
        else:
            self.update_mlp(batch)

    def update_mlp(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        # --- Critic Update ---
        self.critic_1.train(); self.critic_2.train()
        self.target_critic_1.eval(); self.target_critic_2.eval()
        self.actor.eval()
        
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1 = self.target_critic_1(next_states, next_actions)
            target_q2 = self.target_critic_2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            y = rewards + self.gamma * (1.0 - dones) * (target_q - self.alpha * next_log_probs)
        
        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)
        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # --- Actor & Alpha Update ---
        self.actor.train()
        self.critic_1.eval(); self.critic_2.eval()
        
        pi_actions, log_probs = self.actor.sample(states)
        q1_pi = self.critic_1(states, pi_actions)
        q2_pi = self.critic_2(states, pi_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        actor_loss = (self.alpha * log_probs - min_q_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()
        
        # --- Target Update ---
        polyak_update(self.target_critic_1, self.critic_1, self.tau)
        polyak_update(self.target_critic_2, self.critic_2, self.tau)

    def update_rnn(self, batch):
        # RNN 的批次是 (batch_size, seq_len, dim)
        states, actions, rewards, next_states, dones = batch
        
        # --- Critic Update ---
        self.critic_1.train(); self.critic_2.train()
        self.target_critic_1.eval(); self.target_critic_2.eval()
        self.actor.eval()

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            target_q1 = self.target_critic_1(next_states, next_actions)
            target_q2 = self.target_critic_2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            y = rewards + self.gamma * (1.0 - dones) * (target_q - self.alpha * next_log_probs)

        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)
        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor & Alpha Update ---
        self.actor.train()
        self.critic_1.eval(); self.critic_2.eval()

        pi_actions, log_probs, _ = self.actor.sample(states)
        q1_pi = self.critic_1(states, pi_actions)
        q2_pi = self.critic_2(states, pi_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_probs - min_q_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # --- Target Update ---
        polyak_update(self.target_critic_1, self.critic_1, self.tau)
        polyak_update(self.target_critic_2, self.critic_2, self.tau)

#################################################################
# 行为克隆 (Behavioral Cloning)
#################################################################

class BehavioralCloning:
    def __init__(self, config):
        self.config = config
        self.model_type = config['MODEL_TYPE']
        self.seq_len = config['RNN_SEQUENCE_LENGTH']
        self.batch_size = config['BATCH_SIZE']
        
        state_dim = STATE_DIM_RNN if self.model_type == 'RNN' else STATE_DIM_MLP
        h1, h2 = config['HIDDEN_DIM1'], config['HIDDEN_DIM2']

        if self.model_type == 'RNN':
            rnn_h = h2
            # BC 是一个确定性模型, 所以我们只输出均值
            self.model = StochasticActor_RNN(state_dim, h1, rnn_h, ACTION_DIM, 1.0).to(device)
        else:
            self.model = StochasticActor_MLP(state_dim, h1, h2, ACTION_DIM, 1.0).to(device)
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['ACTOR_LR'], weight_decay=config['L2_LAMBDA'])
        self.criterion = nn.MSELoss()

    def create_dataset(self, data):
        """将时间序列数据转换为监督学习数据集 (X, Y)"""
        print(f"Creating {self.model_type} dataset for BC...")
        X, Y = [], []
        
        if self.model_type == 'RNN':
            # X: (N, seq_len, 9), Y: (N, 3)
            # 注意: Y 是 (N, 3) 而不是 (N, seq_len, 3)
            for t in tqdm(range(self.seq_len, len(data) - 1)):
                # TODO: 检查 episode 边界
                x_seq = [get_state_RNN(data, i) for i in range(t - self.seq_len, t)]
                X.append(x_seq)
                Y.append(get_action_from_row(data.iloc[t])) # 预测当前时间 t 的动作
        else:
            # X: (N, 36), Y: (N, 3)
            for t in tqdm(range(3, len(data) - 1)):
                X.append(get_state_MLP(data, t))
                Y.append(get_action_from_row(data.iloc[t+1])) # 预测 t+1
        
        X = torch.tensor(np.array(X), dtype=torch.float32)
        Y = torch.tensor(np.array(Y), dtype=torch.float32)
        return TensorDataset(X, Y)

    def train(self, train_data, epochs=20):
        dataset = self.create_dataset(train_data)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        print("Starting Behavioral Cloning training...")
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                if self.model_type == 'RNN':
                    mean, _, _ = self.model(x_batch)
                    # 我们只关心序列的最后一步输出
                    y_pred = self.model.max_action * torch.tanh(mean[:, -1, :])
                else:
                    mean, _ = self.model(x_batch)
                    y_pred = self.model.max_action * torch.tanh(mean)
                
                loss = self.criterion(y_pred, y_batch)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, BC Loss: {epoch_loss/len(loader):.6f}")

    def evaluate(self, test_data, data_info):
        """评估 BC 模型的闭环性能"""
        print("Evaluating Behavioral Cloning model...")
        
        test_size = len(test_data)
        
        if self.model_type == 'RNN':
            state_idx = self.seq_len
            state = [get_state_RNN(test_data, i) for i in range(state_idx)] # 初始序列
        else:
            state_idx = 3
            state = get_state_MLP(test_data, state_idx) # 初始 4 帧
            
        predicted_trajectory = []
        actual_trajectory = []
        
        self.model.eval()
        
        for i in tqdm(range(state_idx, test_size)):
            with torch.no_grad():
                if self.model_type == 'RNN':
                    # 输入 (1, seq_len, 9)
                    state_tensor = torch.tensor(np.array(state), dtype=torch.float32).to(device).unsqueeze(0)
                    mean, _, _ = self.model(state_tensor)
                    action = self.model.max_action * torch.tanh(mean[:, -1, :]) # 取最后一步
                else:
                    # 输入 (1, 36)
                    state_tensor = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
                    mean, _ = self.model(state_tensor)
                    action = self.model.max_action * torch.tanh(mean)
                    
            action = action.detach().cpu().numpy()[0]
            
            # --- 获去真实数据 ---
            actual_row = test_data.iloc[i]
            actual_action = get_action_from_row(actual_row)
            
            predicted_trajectory.append(action)
            actual_trajectory.append(actual_action)
            
            # --- 滚动 state ---
            # 闭环评估: 使用 *自己* 的预测来构建下一个状态
            if self.model_type == 'RNN':
                current_state_9d = get_state_RNN(test_data, i-1) # 简化的 state
                pos, vel, acc = get_current_features(current_state_9d)
                new_vel = (action - pos) / 1.0
                new_acc = (new_vel - vel) / 1.0
                next_state_9d = np.concatenate([action, new_vel, new_acc])
                
                state.append(next_state_9d)
                state.pop(0) # 保持 seq_len
            else:
                pos, vel, acc = get_current_features(state[-9:]) # 从堆叠状态中获取
                new_vel = (action - pos) / 1.0
                new_acc = (new_vel - vel) / 1.0
                new_state_9d = np.concatenate([action, new_vel, new_acc])
                state = np.concatenate([state[9:], new_state_9d])
                
        # --- 计算指标 ---
        return calculate_metrics(predicted_trajectory, actual_trajectory, data_info)

#################################################################
# 强化学习 (RL) 训练与评估
#################################################################

def run_training_rl(config, train_data, data_info):
    """运行一次完整的 RL 训练"""
    
    agent = SAC_Agent(config)
    
    # 选择缓冲区
    if config['MODEL_TYPE'] == 'RNN':
        replay_buffer = ReplayBuffer_RNN(config['BUFFER_SIZE'], config['BATCH_SIZE'], config['RNN_SEQUENCE_LENGTH'])
        get_state_fn = get_state_RNN
        state_start_idx = config['RNN_SEQUENCE_LENGTH'] - 1
    else:
        replay_buffer = ReplayBuffer(config['BUFFER_SIZE'], config['BATCH_SIZE'])
        get_state_fn = get_state_MLP
        state_start_idx = 3 # MLP 需要 4 帧
        
    # SACfD: 预填充
    if config['PREFILL_BUFFER']:
        replay_buffer.prefill(train_data)
        
    max_episodes = config['MAX_EPISODES']
    data_length = len(train_data)
    max_steps = min(300, data_length - (state_start_idx + 2))
    
    episode_rewards = []
    episode_mse = []
    
    for episode in range(max_episodes):
        max_start_idx = data_length - max_steps - (state_start_idx + 2)
        if max_start_idx <= state_start_idx:
            print("Dataset too short for training.")
            break
            
        start_idx = np.random.randint(state_start_idx, max_start_idx)
        current_data = train_data.iloc[start_idx : start_idx + max_steps + 5]
        current_data = current_data.reset_index(drop=True) # 重置索引
        
        agent.reset_rnn_hidden_state() # 重置 RNN 隐藏状态
        
        episode_reward = 0
        episode_total_mse = 0
        step_count = 0
        prev_predicted = None
        
        for step in range(max_steps):
            step_count += 1
            t = state_start_idx + step # current_data 的本地索引
            
            if config['MODEL_TYPE'] == 'RNN':
                # RNN Agent 决策时只看当前帧 (1, 1, 9)
                action_state = get_state_RNN(current_data, t)
            else:
                # MLP Agent 决策时看堆叠帧 (1, 36)
                action_state = get_state_MLP(current_data, t)

            action = agent.select_action(action_state)
            
            # --- 环境交互 ---
            actual_row = current_data.iloc[t + 1]
            actual_action = get_action_from_row(actual_row)
            
            # 计算 Jerk (用于奖励)
            pos, vel, acc = get_current_features(get_state_RNN(current_data, t))
            pred_vel = (action - pos) / 1.0
            pred_acc = (pred_vel - vel) / 1.0
            pred_jerk = (pred_acc - acc) / 1.0
            actual_jerk = get_jerk_RNN(current_data, t + 1)
            
            reward = calculate_reward(
                action, actual_action, pred_jerk, actual_jerk,
                prev_predicted, get_action_from_row(current_data.iloc[t]),
                is_terminal=False, 
                use_jerk=config['USE_JERK_PENALTY'], 
                jerk_weight=config['JERK_WEIGHT']
            )
            
            done = False # 在这个设定中, 我们不处理 done
            
            # 存儲到緩衝區 (MLP 存 36-dim, RNN 存 9-dim)
            if config['MODEL_TYPE'] == 'MLP':
                state = get_state_MLP(current_data, t)
                next_state = get_state_MLP(current_data, t + 1)
                replay_buffer.add(state, action, reward, next_state, done)
            else:
                # RNN 存储 9-dim 的 state 和 next_state
                state_9d = get_state_RNN(current_data, t)
                next_state_9d = get_state_RNN(current_data, t + 1)
                replay_buffer.add(state_9d, action, reward, next_state_9d, done)
            
            episode_reward += reward
            episode_total_mse += np.mean((action - actual_action)**2)
            prev_predicted = action
        
        episode_rewards.append(episode_reward)
        episode_mse.append(episode_total_mse / step_count)
        
        # --- 網路更新 ---
        if len(replay_buffer) >= config['BATCH_SIZE'] * (config['RNN_SEQUENCE_LENGTH'] if config['MODEL_TYPE'] == 'RNN' else 1):
            for _ in range(5): # 每個 episode 更新 5 次
                batch = replay_buffer.sample()
                if batch is not None:
                    agent.update(batch)

        if (episode + 1) % 50 == 0:
            print(f'[RL] Episode: {episode+1}/{max_episodes}, Reward: {episode_reward:.2f}, MSE: {episode_mse[-1]:.4f}')
            
    return agent, {'rewards': episode_rewards}

def evaluate_rl_agent(agent, config, test_data, data_info):
    """评估 RL Agent 的闭环性能"""
    print("Evaluating RL Agent...")
    
    test_size = len(test_data)
    
    if config['MODEL_TYPE'] == 'RNN':
        state_idx = config['RNN_SEQUENCE_LENGTH']
        state = [get_state_RNN(test_data, i) for i in range(state_idx)]
        agent.reset_rnn_hidden_state()
    else:
        state_idx = 3
        state = get_state_MLP(test_data, state_idx)
            
    predicted_trajectory = []
    actual_trajectory = []
    
    for i in tqdm(range(state_idx, test_size)):
        
        # --- 鲁棒性测试: 注入噪声 ---
        noise_vec = np.random.randn(*state.shape) * config['ROBUSTNESS_NOISE_LEVEL']
        action_state = state + noise_vec

        if config['MODEL_TYPE'] == 'RNN':
            # 评估 RNN 时, 我们需要传入整个序列 (或最后一步)
            # 简化: 仅传入最后一步 (1, 1, 9)
            action_state_rnn = (get_state_RNN(test_data, i) + noise_vec[-1] if config['ROBUSTNESS_NOISE_LEVEL'] > 0 else get_state_RNN(test_data, i))
            action = agent.select_action(action_state_rnn, deterministic=True)
        else:
            action = agent.select_action(action_state, deterministic=True)
            
        actual_row = test_data.iloc[i]
        actual_action = get_action_from_row(actual_row)
        
        predicted_trajectory.append(action)
        actual_trajectory.append(actual_action)
        
        # --- 滚动 state (使用真实数据, 这叫开环测试, 更公平) ---
        # 闭环测试 (使用 action) 会导致误差累积, 更难
        if config['MODEL_TYPE'] == 'RNN':
            # --- 闭环测试 (更真实) ---
            pos, vel, acc = get_current_features(get_state_RNN(test_data, i-1)) # 简化
            new_vel = (action - pos) / 1.0
            new_acc = (new_vel - vel) / 1.0
            next_state_9d = np.concatenate([action, new_vel, new_acc])
            state.append(next_state_9d)
            state.pop(0)
        else:
            # --- 闭环测试 (更真实) ---
            pos, vel, acc = get_current_features(state[-9:])
            new_vel = (action - pos) / 1.0
            new_acc = (new_vel - vel) / 1.0
            new_state_9d = np.concatenate([action, new_vel, new_acc])
            state = np.concatenate([state[9:], new_state_9d])
    
    return calculate_metrics(predicted_trajectory, actual_trajectory, data_info)

def calculate_metrics(predicted, actual, data_info):
    """反归一化并计算最终指标"""
    predicted_traj = np.array(predicted)
    actual_traj = np.array(actual)
    
    predicted_actual = np.zeros_like(predicted_traj)
    actual_actual = np.zeros_like(actual_traj)

    for i, joint_name in enumerate(['joint1', 'joint2', 'joint3']):
        sigma = data_info[joint_name]['sigma']
        mu = data_info[joint_name]['mu']
        predicted_actual[:, i] = predicted_traj[:, i] * sigma + mu
        actual_actual[:, i] = actual_traj[:, i] * sigma + mu

    mse_actual = np.mean((predicted_actual - actual_actual)**2)
    rmse_actual = np.sqrt(mse_actual)
    mae_actual = np.mean(np.abs(predicted_actual - actual_actual))
    
    print(f'Test Results (Actual Angle Space): RMSE: {rmse_actual:.4f}, MAE: {mae_actual:.4f}')
    return {'rmse': rmse_actual, 'mae': mae_actual}, (predicted_actual, actual_actual)

#################################################################
# 主函数 (Main Function)
#################################################################

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'motion_trajectory_dataset.csv')
    
    train_data, test_data, data_info = load_and_preprocess_data(
        csv_path,
        CONFIG['GENERALIZATION_TRAIN_IDS'],
        CONFIG['GENERALIZATION_TEST_IDS']
    )
    if train_data is None:
        return

    test_results = {}
    training_history = {}
    plot_data = None

    if CONFIG['EXPERIMENT_NAME'] == 'BC':
        agent = BehavioralCloning(CONFIG)
        agent.train(train_data, epochs=20)
        test_results, plot_data = agent.evaluate(test_data, data_info)
        
    elif 'SAC' in CONFIG['EXPERIMENT_NAME']:
        agent, training_history = run_training_rl(CONFIG, train_data, data_info)
        test_results, plot_data = evaluate_rl_agent(agent, CONFIG, test_data, data_info)
        
    else:
        print(f"Unknown experiment name: {CONFIG['EXPERIMENT_NAME']}")
        return

    # --- 绘图 ---
    print("\n" + "="*50)
    print(f"Experiment: {CONFIG['EXPERIMENT_NAME']} Complete.")
    print(f"Final Test RMSE: {test_results['rmse']:.4f}")
    print("="*50)

    # 图 1: 训练奖励
    if 'rewards' in training_history:
        plt.figure(figsize=(12, 6))
        smooth_rewards = pd.Series(training_history['rewards']).rolling(20, min_periods=1).mean()
        plt.plot(smooth_rewards)
        plt.title(f"{CONFIG['EXPERIMENT_NAME']} Training Rewards (Smoothed)")
        plt.xlabel('Episode')
        plt.ylabel('Smoothed Reward')
        plt.grid(True)
        plt.savefig(f"{CONFIG['EXPERIMENT_NAME']}_training_rewards.png")

    # 图 2: 轨迹对比
    if plot_data:
        predicted_actual, actual_actual = plot_data
        plt.figure(figsize=(12, 10))
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(predicted_actual[:, i], 'r', linewidth=1.5, label='Predicted')
            plt.plot(actual_actual[:, i], 'b--', linewidth=1.5, label='Actual')
            plt.title(f'Joint {i+1} Angle Prediction vs Actual (Test Set)')
            plt.legend()
            plt.grid(True)
        plt.suptitle(f"{CONFIG['EXPERIMENT_NAME']} Motion Trajectory Prediction")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{CONFIG['EXPERIMENT_NAME']}_trajectory_comparison.png")

    print(f"Charts saved for {CONFIG['EXPERIMENT_NAME']}.")

if __name__ == "__main__":
    main()