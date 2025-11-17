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
from collections import deque
import random
import time
from tqdm import tqdm

# =============================================================================
# --- 实验配置区 (在此处切换路径 4 ） ---
# =============================================================================

# 
# 说明: 使用 Sample 0-2 训练，在 Sample 3-4 上测试，并加入噪声模拟传感器误差。
CONFIG = {
    'EXPERIMENT_NAME': 'Path5_Generalization_Robustness',
    'MODEL_TYPE': 'MLP',           # 使用标准 MLP
    'RNN_SEQUENCE_LENGTH': 10,     # (MLP 模式下此参数无效)
    'USE_JERK_PENALTY': True,      # 开启平滑度奖励
    'JERK_WEIGHT': 0.1,
    'PREFILL_BUFFER': False,
    'ROBUSTNESS_NOISE_LEVEL': 0.05, # 关键: 注入 5% 噪声测试鲁棒性
    'GENERALIZATION_TRAIN_IDS': [0, 1, 2], # 训练集 ID
    'GENERALIZATION_TEST_IDS': [3, 4],     # 测试集 ID (未见过的数据)
    
    # 训练参数
    'MAX_EPISODES': 800,
    'BUFFER_SIZE': 50000,
    'BATCH_SIZE': 256,
    'HIDDEN_DIM1': 256,
    'HIDDEN_DIM2': 128,
    'GAMMA': 0.99, 'TAU': 0.005,
    'ACTOR_LR': 3e-4, 'CRITIC_LR': 3e-4, 'ALPHA_LR': 3e-4,
    'L2_LAMBDA': 1e-4, 'MAX_ACTION': 1.0,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Experiment: {CONFIG['EXPERIMENT_NAME']}, Model: {CONFIG['MODEL_TYPE']}")

STATE_DIM_MLP = 3 * 3 * 4 
STATE_DIM_RNN = 9          
ACTION_DIM = 3

# --- 1. 数据处理 ---

def kalman_filter(data, Q, R):
    n = len(data); filtered = np.zeros(n); x_hat = data[0]; P = 1.0; filtered[0] = x_hat
    for k in range(1, n):
        x_hat_minus = x_hat; P_minus = P + Q; K = P_minus / (P_minus + R)
        x_hat = x_hat_minus + K * (data[k] - x_hat_minus); P = (1 - K) * P_minus; filtered[k] = x_hat
    return filtered

def load_and_preprocess_data(csv_path, train_ids, test_ids):
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found at '{csv_path}'")
        return None, None, None
    
    for joint in ['joint1', 'joint2', 'joint3']:
        mu = data[joint].mean(); sigma = data[joint].std()
        outlier = np.abs(data[joint] - mu) > 3 * sigma
        if outlier.sum() > 0:
            for i in np.where(outlier)[0]:
                if i==0: data.loc[i,joint] = data.loc[i+1,joint]
                elif i==len(data)-1: data.loc[i,joint] = data.loc[i-1,joint]
                else: data.loc[i,joint] = (data.loc[i-1,joint]+data.loc[i+1,joint])/2
    data.interpolate(method='linear', inplace=True)
    
    data['joint1'] = kalman_filter(data['joint1'].values, 0.01, 0.1)
    data['joint2'] = kalman_filter(data['joint2'].values, 0.01, 0.12)
    data['joint3'] = kalman_filter(data['joint3'].values, 0.01, 0.09)
    
    dt = 1.0
    data['velocity1'] = np.concatenate(([0], np.diff(data['joint1'])/dt))
    data['velocity2'] = np.concatenate(([0], np.diff(data['joint2'])/dt))
    data['velocity3'] = np.concatenate(([0], np.diff(data['joint3'])/dt))
    data['acceleration1'] = np.concatenate(([0], np.diff(data['velocity1'])/dt))
    data['acceleration2'] = np.concatenate(([0], np.diff(data['velocity2'])/dt))
    data['acceleration3'] = np.concatenate(([0], np.diff(data['velocity3'])/dt))
    
    data['jerk1'] = np.concatenate(([0], np.diff(data['acceleration1'])/dt))
    data['jerk2'] = np.concatenate(([0], np.diff(data['acceleration2'])/dt))
    data['jerk3'] = np.concatenate(([0], np.diff(data['acceleration3'])/dt))

    data_info = {}
    for name in ['joint1', 'joint2', 'joint3', 'velocity1', 'velocity2', 'velocity3', 
                 'acceleration1', 'acceleration2', 'acceleration3']:
        mu = data[name].mean(); sigma = data[name].std()
        if sigma < 1e-6: sigma = 1.0
        data[name] = (data[name] - mu) / sigma
        data_info[name] = {'mu': mu, 'sigma': sigma}
        
    # Jerk 也可以简单的归一化
    for name in ['jerk1', 'jerk2', 'jerk3']:
         data[name] = data[name] / (data[name].std() + 1e-6)

    train_data = data[data['sample_id'].isin(train_ids)].reset_index(drop=True)
    test_data = data[data['sample_id'].isin(test_ids)].reset_index(drop=True)
    print(f"Loaded: Train {len(train_data)}, Test {len(test_data)}")
    return train_data, test_data, data_info

def get_state_MLP(data, t):
    state = np.zeros(STATE_DIM_MLP)
    for i in range(4):
        idx = max(0, t - i)
        row = data.iloc[idx]
        offset = (3 - i) * 9
        state[offset:offset+9] = row[['joint1', 'joint2', 'joint3','velocity1', 'velocity2', 'velocity3','acceleration1', 'acceleration2', 'acceleration3']].values
    return state

def get_state_RNN(data, t):
    return data.iloc[t][['joint1', 'joint2', 'joint3','velocity1', 'velocity2', 'velocity3','acceleration1', 'acceleration2', 'acceleration3']].values

def get_action_target(data, t):
    return data.iloc[t][['joint1', 'joint2', 'joint3']].values

def get_jerk_target(data, t):
    return data.iloc[t][['jerk1', 'jerk2', 'jerk3']].values

def calculate_reward(pred, actual, pred_jerk, actual_jerk, prev_pred, prev_actual, use_jerk, jerk_weight):
    mse = np.mean((pred - actual)**2)
    reward = -mse
    if prev_pred is not None:
        v_mse = np.mean(((pred-prev_pred) - (actual-prev_actual))**2)
        reward -= 0.2 * v_mse
    if use_jerk:
        j_mse = np.mean((pred_jerk - actual_jerk)**2)
        reward -= jerk_weight * j_mse
    return reward / (1 + abs(reward))

# --- 2. 经验回放缓冲区 ---

class ReplayBuffer_MLP:
    def __init__(self, cap, batch):
        self.buffer = deque(maxlen=cap); self.batch = batch
    def add(self, s, a, r, sn, d): self.buffer.append((s,a,r,sn,d))
    def sample(self):
        if len(self.buffer) < self.batch: return None
        b = random.sample(self.buffer, self.batch)
        s, a, r, sn, d = zip(*b)
        return (torch.tensor(np.array(s), dtype=torch.float32).to(device),
                torch.tensor(np.array(a), dtype=torch.float32).to(device),
                torch.tensor(np.array(r), dtype=torch.float32).to(device).unsqueeze(1),
                torch.tensor(np.array(sn), dtype=torch.float32).to(device),
                torch.tensor(np.array(d), dtype=torch.float32).to(device).unsqueeze(1))
    def __len__(self): return len(self.buffer)

class ReplayBuffer_RNN:
    def __init__(self, cap, batch, seq_len):
        self.buffer = deque(maxlen=cap); self.batch = batch; self.seq_len = seq_len
    def add(self, s, a, r, sn, d): self.buffer.append((s,a,r,sn,d))
    def sample(self):
        if len(self.buffer) < self.batch * self.seq_len: return None
        s_seq, a_seq, r_seq, sn_seq, d_seq = [],[],[],[],[]
        for _ in range(self.batch):
            idx = random.randint(self.seq_len-1, len(self.buffer)-1)
            seq = [self.buffer[i] for i in range(idx-self.seq_len+1, idx+1)]
            s,a,r,sn,d = zip(*seq)
            s_seq.append(s); a_seq.append(a); r_seq.append(r); sn_seq.append(sn); d_seq.append(d)
        return (torch.tensor(np.array(s_seq), dtype=torch.float32).to(device),
                torch.tensor(np.array(a_seq), dtype=torch.float32).to(device),
                torch.tensor(np.array(r_seq), dtype=torch.float32).to(device).unsqueeze(-1),
                torch.tensor(np.array(sn_seq), dtype=torch.float32).to(device),
                torch.tensor(np.array(d_seq), dtype=torch.float32).to(device).unsqueeze(-1))
    def __len__(self): return len(self.buffer)

# --- 3. 网络定义 ---

# MLP Networks
class StochasticActor_MLP(nn.Module):
    def __init__(self, s_dim, h1, h2, a_dim, max_a=1.0):
        super().__init__()
        self.l1 = nn.Linear(s_dim, h1); self.bn1 = nn.BatchNorm1d(h1)
        self.l2 = nn.Linear(h1, h2); self.bn2 = nn.BatchNorm1d(h2)
        self.mu = nn.Linear(h2, a_dim); self.log_std = nn.Linear(h2, a_dim)
        self.max_a = max_a
    def forward(self, x):
        x = F.relu(self.bn1(self.l1(x)))
        x = F.relu(self.bn2(self.l2(x)))
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return self.mu(x), torch.exp(log_std)
    def sample(self, s):
        mu, std = self.forward(s)
        dist = Normal(mu, std); z = dist.rsample()
        action = self.max_a * torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(self.max_a*(1-torch.tanh(z).pow(2))+1e-6)
        return action, log_prob.sum(-1, keepdim=True)

class Critic_MLP(nn.Module):
    def __init__(self, dim, h1, h2):
        super().__init__()
        self.l1 = nn.Linear(dim, h1); self.bn1 = nn.BatchNorm1d(h1)
        self.l2 = nn.Linear(h1, h2); self.bn2 = nn.BatchNorm1d(h2)
        self.l3 = nn.Linear(h2, 1)
    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = F.relu(self.bn1(self.l1(x)))
        x = F.relu(self.bn2(self.l2(x)))
        return self.l3(x)

# RNN Networks
class StochasticActor_RNN(nn.Module):
    def __init__(self, s_dim, h, rnn_h, a_dim, max_a=1.0):
        super().__init__()
        self.l1 = nn.Linear(s_dim, h); self.bn1 = nn.BatchNorm1d(h)
        self.lstm = nn.LSTM(h, rnn_h, batch_first=True)
        self.l2 = nn.Linear(rnn_h, h); self.bn2 = nn.BatchNorm1d(h)
        self.mu = nn.Linear(h, a_dim); self.log_std = nn.Linear(h, a_dim)
        self.max_a = max_a
    def forward(self, x, h=None, c=None):
        # BN workaround for sequence
        B, L, D = x.shape
        x_flat = x.view(B*L, -1)
        x_emb = F.relu(self.bn1(self.l1(x_flat))).view(B, L, -1)
        if h is None: out, (hn, cn) = self.lstm(x_emb)
        else: out, (hn, cn) = self.lstm(x_emb, (h, c))
        out_flat = out.contiguous().view(B*L, -1)
        feat = F.relu(self.bn2(self.l2(out_flat))).view(B, L, -1)
        log_std = torch.clamp(self.log_std(feat), -20, 2)
        return self.mu(feat), torch.exp(log_std), (hn, cn)
    def sample(self, s, h=None, c=None):
        mu, std, hidden = self.forward(s, h, c)
        dist = Normal(mu, std); z = dist.rsample()
        action = self.max_a * torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(self.max_a*(1-torch.tanh(z).pow(2))+1e-6)
        return action, log_prob.sum(-1, keepdim=True), hidden

class Critic_RNN(nn.Module):
    def __init__(self, s_dim, a_dim, h, rnn_h):
        super().__init__()
        self.sl1 = nn.Linear(s_dim, h); self.sbn = nn.BatchNorm1d(h)
        self.lstm = nn.LSTM(h, rnn_h, batch_first=True)
        self.al1 = nn.Linear(a_dim, h)
        self.cl1 = nn.Linear(rnn_h+h, h); self.cbn = nn.BatchNorm1d(h)
        self.cl2 = nn.Linear(h, 1)
    def forward(self, s, a):
        B, L, _ = s.shape
        s_flat = s.view(B*L, -1)
        s_emb = F.relu(self.sbn(self.sl1(s_flat))).view(B, L, -1)
        rnn_out, _ = self.lstm(s_emb)
        a_flat = a.view(B*L, -1)
        a_emb = F.relu(self.al1(a_flat)).view(B, L, -1)
        comb = torch.cat([rnn_out, a_emb], 2).view(B*L, -1)
        x = F.relu(self.cbn(self.cl1(comb)))
        return self.cl2(x).view(B, L, 1)

# --- 4. SAC Agent ---

class SAC_Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.is_rnn = (cfg['MODEL_TYPE'] == 'RNN')
        s_dim = STATE_DIM_RNN if self.is_rnn else STATE_DIM_MLP
        h1, h2 = cfg['HIDDEN_DIM1'], cfg['HIDDEN_DIM2']
        
        if self.is_rnn:
            self.actor = StochasticActor_RNN(s_dim, h1, h2, ACTION_DIM).to(device)
            self.c1 = Critic_RNN(s_dim, ACTION_DIM, h1, h2).to(device)
            self.c2 = Critic_RNN(s_dim, ACTION_DIM, h1, h2).to(device)
            self.tc1 = Critic_RNN(s_dim, ACTION_DIM, h1, h2).to(device)
            self.tc2 = Critic_RNN(s_dim, ACTION_DIM, h1, h2).to(device)
        else:
            self.actor = StochasticActor_MLP(s_dim, h1, h2, ACTION_DIM).to(device)
            self.c1 = Critic_MLP(s_dim + ACTION_DIM, h1, h2).to(device)
            self.c2 = Critic_MLP(s_dim + ACTION_DIM, h1, h2).to(device)
            self.tc1 = Critic_MLP(s_dim + ACTION_DIM, h1, h2).to(device)
            self.tc2 = Critic_MLP(s_dim + ACTION_DIM, h1, h2).to(device)
            
        self.tc1.load_state_dict(self.c1.state_dict())
        self.tc2.load_state_dict(self.c2.state_dict())
        
        self.a_opt = optim.Adam(self.actor.parameters(), lr=cfg['ACTOR_LR'], weight_decay=cfg['L2_LAMBDA'])
        self.c_opt = optim.Adam(list(self.c1.parameters())+list(self.c2.parameters()), lr=cfg['CRITIC_LR'], weight_decay=cfg['L2_LAMBDA'])
        
        self.target_entropy = -ACTION_DIM
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg['ALPHA_LR'])
        self.alpha = 1.0
        self.h_state = None # RNN hidden state
        
    def select_action(self, state, deterministic=False):
        self.actor.eval()
        with torch.no_grad():
            if self.is_rnn: # state: (1, 1, 9)
                st = torch.FloatTensor(state).to(device).unsqueeze(0).unsqueeze(0)
                mu, std, self.h_state = self.actor(st, self.h_state[0] if self.h_state else None, self.h_state[1] if self.h_state else None)
                mu, std = mu[:,-1,:], std[:,-1,:]
            else: # state: (36,)
                st = torch.FloatTensor(state).to(device).unsqueeze(0)
                mu, std = self.actor(st)
                
            if deterministic: action = torch.tanh(mu)
            else: action = torch.tanh(Normal(mu, std).sample())
            
        self.actor.train()
        return action.cpu().numpy()[0]

    def reset_rnn(self): self.h_state = None

    def update(self, batch):
        s, a, r, sn, d = batch
        
        # Critic
        with torch.no_grad():
            if self.is_rnn: next_a, next_log_p, _ = self.actor.sample(sn)
            else: next_a, next_log_p = self.actor.sample(sn)
            tq = torch.min(self.tc1(sn, next_a), self.tc2(sn, next_a))
            y = r + self.cfg['GAMMA'] * (1-d) * (tq - self.alpha * next_log_p)
        
        loss_c = F.mse_loss(self.c1(s,a), y) + F.mse_loss(self.c2(s,a), y)
        self.c_opt.zero_grad(); loss_c.backward(); self.c_opt.step()
        
        # Actor
        if self.is_rnn: pi, log_p, _ = self.actor.sample(s)
        else: pi, log_p = self.actor.sample(s)
        q_pi = torch.min(self.c1(s, pi), self.c2(s, pi))
        loss_a = (self.alpha * log_p - q_pi).mean()
        self.a_opt.zero_grad(); loss_a.backward(); self.a_opt.step()
        
        # Alpha
        loss_alpha = -(self.log_alpha * (log_p.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(); loss_alpha.backward(); self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()
        
        # Target Polyak
        with torch.no_grad():
            for p, tp in zip(self.c1.parameters(), self.tc1.parameters()): tp.data.copy_(self.cfg['TAU']*p.data+(1-self.cfg['TAU'])*tp.data)
            for p, tp in zip(self.c2.parameters(), self.tc2.parameters()): tp.data.copy_(self.cfg['TAU']*p.data+(1-self.cfg['TAU'])*tp.data)

# --- 5. 训练与评估主逻辑 ---

def run_experiment():
    # 路径修复
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'motion_trajectory_dataset.csv')
    
    # 加载数据
    train_data, test_data, data_info = load_and_preprocess_data(
        csv_path, CONFIG['GENERALIZATION_TRAIN_IDS'], CONFIG['GENERALIZATION_TEST_IDS']
    )
    if train_data is None or len(train_data) == 0: return

    # 初始化
    agent = SAC_Agent(CONFIG)
    if CONFIG['MODEL_TYPE'] == 'RNN':
        buffer = ReplayBuffer_RNN(CONFIG['BUFFER_SIZE'], CONFIG['BATCH_SIZE'], CONFIG['RNN_SEQUENCE_LENGTH'])
        state_fn = get_state_RNN; offset = CONFIG['RNN_SEQUENCE_LENGTH']
    else:
        buffer = ReplayBuffer_MLP(CONFIG['BUFFER_SIZE'], CONFIG['BATCH_SIZE'])
        state_fn = get_state_MLP; offset = 4

    print(f"Start Training... Max Episodes: {CONFIG['MAX_EPISODES']}")
    rewards = []
    
    for ep in range(CONFIG['MAX_EPISODES']):
        # 简单的 Episode 采样
        max_idx = len(train_data) - 300 - offset
        if max_idx < offset: break
        start_idx = np.random.randint(offset, max_idx)
        
        ep_data = train_data.iloc[start_idx : start_idx + 300 + 5].reset_index(drop=True)
        
        if CONFIG['MODEL_TYPE'] == 'RNN': agent.reset_rnn()
        
        state = state_fn(ep_data, offset-1)
        ep_reward = 0
        prev_pred = None
        
        for t in range(300):
            # 决策
            if CONFIG['MODEL_TYPE'] == 'RNN': action_state = state # (9,)
            else: action_state = state # (36,)
            
            action = agent.select_action(action_state)
            
            # 环境反馈
            idx_curr = offset - 1 + t
            actual = get_action_target(ep_data, idx_curr + 1)
            
            # 计算下一状态
            next_state = state_fn(ep_data, idx_curr + 1)
            
            # 奖励
            p_jerk = 0; a_jerk = 0 # 简化
            if CONFIG['USE_JERK_PENALTY']: # 粗略计算 jerk
                curr_vel = (state[3:6] if CONFIG['MODEL_TYPE']=='RNN' else state[30:33])
                curr_acc = (state[6:9] if CONFIG['MODEL_TYPE']=='RNN' else state[33:36])
                p_vel = (action - (state[0:3] if CONFIG['MODEL_TYPE']=='RNN' else state[27:30]))
                p_acc = p_vel - curr_vel
                p_jerk = p_acc - curr_acc
                a_jerk = get_jerk_target(ep_data, idx_curr+1)

            r = calculate_reward(action, actual, p_jerk, a_jerk, prev_pred, get_action_target(ep_data, idx_curr), 
                                 CONFIG['USE_JERK_PENALTY'], CONFIG['JERK_WEIGHT'])
            
            buffer.add(state, action, r, next_state, False)
            
            state = next_state
            ep_reward += r
            prev_pred = action
            
        # 更新网络
        if len(buffer) > CONFIG['BATCH_SIZE'] * 5:
            for _ in range(5): # 每回合更新5次
                batch = buffer.sample()
                if batch: agent.update(batch)
                
        rewards.append(ep_reward)
        if (ep+1) % 50 == 0: print(f"Episode {ep+1}: Reward {ep_reward:.2f}")

    # --- 评估 ---
    print("Training complete. Evaluating...")
    preds, actuals = [], []
    
    if CONFIG['MODEL_TYPE'] == 'RNN':
        start_idx = CONFIG['RNN_SEQUENCE_LENGTH']
        state_seq = [get_state_RNN(test_data, i) for i in range(start_idx)]
        agent.reset_rnn()
        # 预热 RNN hidden state
        for s in state_seq[:-1]:
             _ = agent.select_action(s, deterministic=True)
        curr_state = state_seq[-1]
    else:
        start_idx = 4
        curr_state = get_state_MLP(test_data, start_idx-1)
        
    for i in range(start_idx, len(test_data)):
        # 鲁棒性测试: 注入噪声
        noise = 0.0
        if CONFIG['ROBUSTNESS_NOISE_LEVEL'] > 0:
            noise = np.random.randn(*curr_state.shape) * CONFIG['ROBUSTNESS_NOISE_LEVEL']
            
        action = agent.select_action(curr_state + noise, deterministic=True)
        
        preds.append(action)
        actuals.append(get_action_target(test_data, i))
        
        # 开环更新 (使用真实数据作为下一状态)
        if CONFIG['MODEL_TYPE'] == 'RNN': curr_state = get_state_RNN(test_data, i)
        else: curr_state = get_state_MLP(test_data, i)

    # 计算指标
    preds = np.array(preds); actuals = np.array(actuals)
    # 反归一化
    p_real = np.zeros_like(preds); a_real = np.zeros_like(actuals)
    for i, name in enumerate(['joint1','joint2','joint3']):
        mu = data_info[name]['mu']; sigma = data_info[name]['sigma']
        p_real[:,i] = preds[:,i]*sigma + mu
        a_real[:,i] = actuals[:,i]*sigma + mu
        
    rmse = np.sqrt(np.mean((p_real - a_real)**2))
    print(f"Final RMSE (Actual Space): {rmse:.4f}")
    
    # 绘图
    plt.figure(figsize=(10,5))
    plt.plot(pd.Series(rewards).rolling(20).mean())
    plt.title(f"{CONFIG['EXPERIMENT_NAME']} Training Rewards")
    plt.savefig(f"{CONFIG['EXPERIMENT_NAME']}_rewards.png")
    
    plt.figure(figsize=(10,8))
    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.plot(p_real[:,i], 'r', label='Pred')
        plt.plot(a_real[:,i], 'b--', label='Actual')
        plt.title(f"Joint {i+1}")
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"{CONFIG['EXPERIMENT_NAME']}_predictions.png")
    print("Charts saved.")

if __name__ == "__main__":
    run_experiment()