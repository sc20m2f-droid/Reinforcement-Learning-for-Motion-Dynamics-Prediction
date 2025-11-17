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
# --- 终极融合配置 (FINAL FUSION CONFIGURATION) ---
# =============================================================================
CONFIG = {
    'EXPERIMENT_NAME': 'Final_Fusion_SACfD_Jerk', # 实验名称
    
    # --- 融合核心设置 ---
    'MODEL_TYPE': 'MLP',        # 保持 MLP，因为它在您的 Path 3 中表现最好
    'PREFILL_BUFFER': True,     # [来自 Path 1] 开启演示学习 (SACfD)
    'USE_JERK_PENALTY': True,   # [来自 Path 3] 开启平滑奖励 (Jerk Penalty)
    'JERK_WEIGHT': 0.001,         # 惩罚权重
    
    # --- 其他设置 ---
    'RNN_SEQUENCE_LENGTH': 10,    # (仅当 MODEL_TYPE='RNN' 时生效)
    'ROBUSTNESS_NOISE_LEVEL': 0.0, # 标准测试，不加额外噪声
    'GENERALIZATION_TRAIN_IDS': [0, 1, 2, 3, 4], # 使用全部数据训练
    'GENERALIZATION_TEST_IDS': [4],              # 仅用于代码逻辑，不影响训练集
    
    # --- 训练超参数 (保持不变) ---
    'MAX_EPISODES': 800,
    'BUFFER_SIZE': 50000,
    'BATCH_SIZE': 256,
    'HIDDEN_DIM1': 256,
    'HIDDEN_DIM2': 128,
    'GAMMA': 0.99, 'TAU': 0.005,
    'ACTOR_LR': 1e-4, 'CRITIC_LR': 1e-4, 'ALPHA_LR': 3e-4,
    'L2_LAMBDA': 1e-4, 'MAX_ACTION': 1.0,
}

# =============================================================================
# --- 核心逻辑 (Core Logic) ---
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Experiment: {CONFIG['EXPERIMENT_NAME']}")
print(f"Configuration: SACfD={CONFIG['PREFILL_BUFFER']}, Jerk_Penalty={CONFIG['USE_JERK_PENALTY']}")

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
        
    # Jerk 归一化
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
    
    # --- 修复: 添加缺失的 prefill 方法并使用正确的函数名 ---
    def prefill(self, data):
        print(f"Prefilling buffer with {len(data)-4} transitions...")
        for t in tqdm(range(3, len(data) - 1)):
            state = get_state_MLP(data, t); next_state = get_state_MLP(data, t + 1)
            # 使用 get_action_target 代替 get_action_from_row
            action = get_action_target(data, t+1)
            reward = 1.0; done = False
            self.add(state, action, reward, next_state, done)
        print("Buffer prefill complete.")

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
    
    # --- 修复: 确保 RNN buffer 也有 prefill 方法 ---
    def prefill(self, data):
        print(f"Prefilling RNN buffer with {len(data)-1} transitions...")
        for t in tqdm(range(len(data) - 1)):
            state = get_state_RNN(data, t); next_state = get_state_RNN(data, t + 1)
            action = get_action_target(data, t+1)
            reward = 1.0; done = False
            self.add(state, action, reward, next_state, done)
        print("RNN Buffer prefill complete.")

# --- 3. 网络定义 ---

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

class StochasticActor_RNN(nn.Module):
    def __init__(self, s_dim, h, rnn_h, a_dim, max_a=1.0):
        super().__init__()
        self.l1 = nn.Linear(s_dim, h); self.bn1 = nn.BatchNorm1d(h)
        self.lstm = nn.LSTM(h, rnn_h, batch_first=True)
        self.l2 = nn.Linear(rnn_h, h); self.bn2 = nn.BatchNorm1d(h)
        self.mu = nn.Linear(h, a_dim); self.log_std = nn.Linear(h, a_dim)
        self.max_a = max_a
    def forward(self, x, h=None, c=None):
        B, L, D = x.shape; x_flat = x.view(B*L, -1)
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
        B, L, _ = s.shape; s_flat = s.view(B*L, -1)
        s_emb = F.relu(self.sbn(self.sl1(s_flat))).view(B, L, -1)
        rnn_out, _ = self.lstm(s_emb); a_flat = a.view(B*L, -1)
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
            
        self.tc1.load_state_dict(self.c1.state_dict()); self.tc2.load_state_dict(self.c2.state_dict())
        self.a_opt = optim.Adam(self.actor.parameters(), lr=cfg['ACTOR_LR'], weight_decay=cfg['L2_LAMBDA'])
        self.c_opt = optim.Adam(list(self.c1.parameters())+list(self.c2.parameters()), lr=cfg['CRITIC_LR'], weight_decay=cfg['L2_LAMBDA'])
        self.target_entropy = -ACTION_DIM
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg['ALPHA_LR'])
        self.alpha = 1.0; self.h_state = None
        
    def select_action(self, state, deterministic=False):
        self.actor.eval()
        with torch.no_grad():
            if self.is_rnn:
                st = torch.FloatTensor(state).to(device).unsqueeze(0).unsqueeze(0) if state.ndim==1 else torch.FloatTensor(state).to(device).unsqueeze(0)
                mu, std, self.h_state = self.actor(st, self.h_state[0] if self.h_state else None, self.h_state[1] if self.h_state else None)
                mu, std = mu[:,-1,:], std[:,-1,:]
            else:
                st = torch.FloatTensor(state).to(device).unsqueeze(0)
                mu, std = self.actor(st)
            if deterministic: action = torch.tanh(mu)
            else: action = torch.tanh(Normal(mu, std).sample())
        self.actor.train(); return action.cpu().numpy()[0]

    def reset_rnn(self): self.h_state = None

    def update(self, batch):
        s, a, r, sn, d = batch
        with torch.no_grad():
            if self.is_rnn: next_a, next_log_p, _ = self.actor.sample(sn)
            else: next_a, next_log_p = self.actor.sample(sn)
            tq = torch.min(self.tc1(sn, next_a), self.tc2(sn, next_a))
            y = r + self.cfg['GAMMA'] * (1-d) * (tq - self.alpha * next_log_p)
        loss_c = F.mse_loss(self.c1(s,a), y) + F.mse_loss(self.c2(s,a), y)
        self.c_opt.zero_grad(); loss_c.backward(); self.c_opt.step()
        if self.is_rnn: pi, log_p, _ = self.actor.sample(s)
        else: pi, log_p = self.actor.sample(s)
        q_pi = torch.min(self.c1(s, pi), self.c2(s, pi))
        loss_a = (self.alpha * log_p - q_pi).mean()
        self.a_opt.zero_grad(); loss_a.backward(); self.a_opt.step()
        loss_alpha = -(self.log_alpha * (log_p.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(); loss_alpha.backward(); self.alpha_opt.step(); self.alpha = self.log_alpha.exp().item()
        with torch.no_grad():
            for p, tp in zip(self.c1.parameters(), self.tc1.parameters()): tp.data.copy_(self.cfg['TAU']*p.data+(1-self.cfg['TAU'])*tp.data)
            for p, tp in zip(self.c2.parameters(), self.tc2.parameters()): tp.data.copy_(self.cfg['TAU']*p.data+(1-self.cfg['TAU'])*tp.data)

# --- 5. 训练与评估主逻辑 ---

def run_experiment():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'motion_trajectory_dataset.csv')
    
    train_data, test_data, data_info = load_and_preprocess_data(
        csv_path, CONFIG['GENERALIZATION_TRAIN_IDS'], CONFIG['GENERALIZATION_TEST_IDS']
    )
    if train_data is None or len(train_data) == 0: return

    agent = SAC_Agent(CONFIG)
    if CONFIG['MODEL_TYPE'] == 'RNN':
        buffer = ReplayBuffer_RNN(CONFIG['BUFFER_SIZE'], CONFIG['BATCH_SIZE'], CONFIG['RNN_SEQUENCE_LENGTH'])
        state_fn = get_state_RNN; offset = CONFIG['RNN_SEQUENCE_LENGTH']
    else:
        buffer = ReplayBuffer_MLP(CONFIG['BUFFER_SIZE'], CONFIG['BATCH_SIZE'])
        state_fn = get_state_MLP; offset = 4

    # --- 关键点 1: 预填充 (SACfD) ---
    if CONFIG['PREFILL_BUFFER']:
        buffer.prefill(train_data)

    print(f"Start Training... Max Episodes: {CONFIG['MAX_EPISODES']}")
    rewards = []
    
    for ep in range(CONFIG['MAX_EPISODES']):
        max_idx = len(train_data) - 300 - offset
        if max_idx < offset: break
        start_idx = np.random.randint(offset, max_idx)
        ep_data = train_data.iloc[start_idx : start_idx + 300 + 5].reset_index(drop=True)
        
        if CONFIG['MODEL_TYPE'] == 'RNN': agent.reset_rnn()
        state = state_fn(ep_data, offset-1)
        ep_reward = 0; prev_pred = None
        
        for t in range(300):
            action = agent.select_action(state)
            
            idx_curr = offset - 1 + t
            actual = get_action_target(ep_data, idx_curr + 1)
            next_state = state_fn(ep_data, idx_curr + 1)
            
            # --- 关键点 2: Jerk Reward ---
            p_jerk = 0; a_jerk = 0
            if CONFIG['USE_JERK_PENALTY']:
                curr_vel = (state[3:6] if CONFIG['MODEL_TYPE']=='RNN' else state[30:33])
                curr_acc = (state[6:9] if CONFIG['MODEL_TYPE']=='RNN' else state[33:36])
                p_vel = (action - (state[0:3] if CONFIG['MODEL_TYPE']=='RNN' else state[27:30]))
                p_acc = p_vel - curr_vel
                p_jerk = p_acc - curr_acc
                a_jerk = get_jerk_target(ep_data, idx_curr+1)

            r = calculate_reward(action, actual, p_jerk, a_jerk, prev_pred, get_action_target(ep_data, idx_curr), 
                                 CONFIG['USE_JERK_PENALTY'], CONFIG['JERK_WEIGHT'])
            
            if CONFIG['MODEL_TYPE'] == 'MLP': buffer.add(state, action, r, next_state, False)
            else: buffer.add(state, action, r, next_state, False)
            
            state = next_state; ep_reward += r; prev_pred = action
            
        if len(buffer) > CONFIG['BATCH_SIZE'] * 5:
            for _ in range(5):
                batch = buffer.sample()
                if batch: agent.update(batch)
                
        rewards.append(ep_reward)
        if (ep+1) % 50 == 0: print(f"Episode {ep+1}: Reward {ep_reward:.2f}")

    print("Training complete. Evaluating...")
    preds, actuals = [], []
    
    if CONFIG['MODEL_TYPE'] == 'RNN':
        start_idx = CONFIG['RNN_SEQUENCE_LENGTH']
        state_seq = [get_state_RNN(test_data, i) for i in range(start_idx)]
        agent.reset_rnn()
        for s in state_seq[:-1]: _ = agent.select_action(s, deterministic=True)
        curr_state = state_seq[-1]
    else:
        start_idx = 4
        curr_state = get_state_MLP(test_data, start_idx-1)
        
    for i in range(start_idx, len(test_data)):
        action = agent.select_action(curr_state, deterministic=True)
        preds.append(action); actuals.append(get_action_target(test_data, i))
        if CONFIG['MODEL_TYPE'] == 'RNN': curr_state = get_state_RNN(test_data, i)
        else: curr_state = get_state_MLP(test_data, i)

    preds = np.array(preds); actuals = np.array(actuals)
    p_real = np.zeros_like(preds); a_real = np.zeros_like(actuals)
    for i, name in enumerate(['joint1','joint2','joint3']):
        mu = data_info[name]['mu']; sigma = data_info[name]['sigma']
        p_real[:,i] = preds[:,i]*sigma + mu; a_real[:,i] = actuals[:,i]*sigma + mu
        
    rmse = np.sqrt(np.mean((p_real - a_real)**2))
    print(f"Final RMSE (Actual Space): {rmse:.4f}")
    
    plt.figure(figsize=(10,5)); plt.plot(pd.Series(rewards).rolling(20).mean())
    plt.title(f"{CONFIG['EXPERIMENT_NAME']} Training Rewards"); plt.savefig(f"{CONFIG['EXPERIMENT_NAME']}_rewards.png")
    
    plt.figure(figsize=(10,8))
    for i in range(3):
        plt.subplot(3,1,i+1); plt.plot(p_real[:,i], 'r', label='Pred'); plt.plot(a_real[:,i], 'b--', label='Actual')
        plt.title(f"Joint {i+1}"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{CONFIG['EXPERIMENT_NAME']}_predictions.png")
    print("Charts saved.")

if __name__ == "__main__":
    run_experiment()