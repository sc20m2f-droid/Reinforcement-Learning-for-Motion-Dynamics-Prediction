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
import optuna # 引入 Optuna

# 设置 PyTorch 设备 (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#################################################################
# 數據處理輔助函數
#################################################################

def kalman_filter(data, Q, R):
    n = len(data)
    filtered = np.zeros(n)
    x_hat = data[0]
    P = 1.0
    filtered[0] = x_hat
    for k in range(1, n):
        x_hat_minus = x_hat
        P_minus = P + Q
        K = P_minus / (P_minus + R)
        x_hat = x_hat_minus + K * (data[k] - x_hat_minus)
        P = (1 - K) * P_minus
        filtered[k] = x_hat
    return filtered

def calculate_reward(predicted, actual, prev_predicted, prev_actual, is_terminal):
    mse = np.mean((predicted - actual)**2)
    reward = -mse
    if prev_predicted is not None and prev_actual is not None:
        pred_velocity = predicted - prev_predicted
        actual_velocity = actual - prev_actual
        velocity_mse = np.mean((pred_velocity - actual_velocity)**2)
        reward = reward - 0.2 * velocity_mse
        actual_direction = np.sign(actual - prev_actual)
        pred_direction = np.sign(predicted - prev_predicted)
        direction_reward = np.sum(actual_direction == pred_direction) / len(actual_direction)
        reward = reward + 0.1 * direction_reward
    if is_terminal:
        reward = reward * 2.0
    reward = reward / (1 + abs(reward))
    return reward

def get_initial_state(data):
    state = np.zeros(3 * 3 * 4)
    for t in range(4):
        idx = min(t, len(data) - 1)
        row = data.iloc[idx]
        offset = t * 9
        state[offset + 0] = row['joint1']
        state[offset + 1] = row['joint2']
        state[offset + 2] = row['joint3']
        state[offset + 3] = row['velocity1']
        state[offset + 4] = row['velocity2']
        state[offset + 5] = row['velocity3']
        state[offset + 6] = row['acceleration1']
        state[offset + 7] = row['acceleration2']
        state[offset + 8] = row['acceleration3']
    return state

def update_state(old_state, action):
    new_state = np.zeros_like(old_state)
    new_state[0:27] = old_state[9:36]
    prev_angles = old_state[27:30]
    prev_vels = old_state[30:33]
    new_state[27:30] = action
    dt = 1.0
    new_state[30:33] = (action - prev_angles) / dt
    new_state[33:36] = (new_state[30:33] - prev_vels) / dt
    return new_state

def polyak_update(target, source, tau):
    with torch.no_grad():
        for target_param, source_param in zip(target.state_dict().values(), source.state_dict().values()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

#################################################################
# 標準經驗回放緩衝區 (Standard Replay Buffer)
#################################################################

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

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

#################################################################
# 網路定義 (Network Definitions)
#################################################################

class DeterministicActor(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, max_action=1.0):
        super(DeterministicActor, self).__init__()
        self.bn_eps = 1e-6
        self.bn_momentum = 0.1
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1, eps=self.bn_eps, momentum=self.bn_momentum)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2, eps=self.bn_eps, momentum=self.bn_momentum)
        self.layer3 = nn.Linear(hidden_dim2, output_dim)
        self.max_action = max_action
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.layer1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.layer1.bias)
        nn.init.kaiming_normal_(self.layer2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.layer2.bias)
        nn.init.kaiming_normal_(self.layer3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.layer3.bias)

    def forward(self, x):
        h1 = torch.tanh(self.bn1(self.layer1(x)))
        h2 = torch.tanh(self.bn2(self.layer2(h1)))
        output = self.max_action * torch.tanh(self.layer3(h2)) 
        return output

class StochasticActor(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, max_action=1.0):
        super(StochasticActor, self).__init__()
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
        nn.init.zeros_(self.layer1.bias)
        nn.init.kaiming_normal_(self.layer2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.layer2.bias)
        nn.init.kaiming_normal_(self.mean_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.mean_layer.bias)
        nn.init.kaiming_normal_(self.log_std_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.log_std_layer.bias)

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

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim=1):
        super(Critic, self).__init__()
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
        nn.init.zeros_(self.layer1.bias)
        nn.init.kaiming_normal_(self.layer2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.layer2.bias)
        nn.init.kaiming_normal_(self.layer3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.layer3.bias)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        h1 = torch.tanh(self.bn1(self.layer1(x)))
        h2 = torch.tanh(self.bn2(self.layer2(h1)))
        q_value = self.layer3(h2)
        return q_value

#################################################################
# 演算法 Agent 定義 (Algorithm Agent Definitions)
#################################################################

class DDPG_Agent:
    def __init__(self, state_dim, action_dim, params):
        self.action_dim = action_dim
        self.max_action = params['max_action']
        self.gamma = params['gamma']
        self.tau = params['tau']
        hidden1, hidden2 = params['hidden_dim1'], params['hidden_dim2']
        
        self.actor = DeterministicActor(state_dim, hidden1, hidden2, action_dim, self.max_action).to(device)
        self.critic = Critic(state_dim + action_dim, hidden1, hidden2, 1).to(device)
        self.target_actor = DeterministicActor(state_dim, hidden1, hidden2, action_dim, self.max_action).to(device)
        self.target_critic = Critic(state_dim + action_dim, hidden1, hidden2, 1).to(device)
        
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.SGD(self.actor.parameters(), lr=params['actor_lr'], momentum=0.9, weight_decay=params['l2_lambda'])
        self.critic_optimizer = optim.SGD(self.critic.parameters(), lr=params['critic_lr'], momentum=0.9, weight_decay=params['l2_lambda'])

    def select_action(self, state, exploration_noise):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).detach().cpu().numpy()[0]
        self.actor.train()
        noise = exploration_noise * np.random.randn(self.action_dim)
        action = (action + noise).clip(-self.max_action, self.max_action)
        return action

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        self.critic.train()
        self.target_actor.eval()
        self.target_critic.eval()
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            y = rewards + self.gamma * (1.0 - dones) * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.critic.eval() 
        self.actor.train()
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        polyak_update(self.target_actor, self.actor, self.tau)
        polyak_update(self.target_critic, self.critic, self.tau)
        return actor_loss.item(), critic_loss.item()

class TD3_Agent:
    def __init__(self, state_dim, action_dim, params):
        self.action_dim = action_dim
        self.max_action = params['max_action']
        self.gamma = params['gamma']
        self.tau = params['tau']
        self.policy_delay = params['policy_delay']
        self.policy_noise = params['policy_noise']
        self.noise_clip = params['noise_clip']
        hidden1, hidden2 = params['hidden_dim1'], params['hidden_dim2']
        
        self.actor = DeterministicActor(state_dim, hidden1, hidden2, action_dim, self.max_action).to(device)
        self.target_actor = DeterministicActor(state_dim, hidden1, hidden2, action_dim, self.max_action).to(device)
        self.critic_1 = Critic(state_dim + action_dim, hidden1, hidden2, 1).to(device)
        self.critic_2 = Critic(state_dim + action_dim, hidden1, hidden2, 1).to(device)
        self.target_critic_1 = Critic(state_dim + action_dim, hidden1, hidden2, 1).to(device)
        self.target_critic_2 = Critic(state_dim + action_dim, hidden1, hidden2, 1).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        
        self.actor_optimizer = optim.SGD(self.actor.parameters(), lr=params['actor_lr'], momentum=0.9, weight_decay=params['l2_lambda'])
        self.critic_optimizer = optim.SGD(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), 
            lr=params['critic_lr'], momentum=0.9, weight_decay=params['l2_lambda']
        )
        self.update_counter = 0

    def select_action(self, state, exploration_noise):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).detach().cpu().numpy()[0]
        self.actor.train()
        noise = exploration_noise * np.random.randn(self.action_dim)
        action = (action + noise).clip(-self.max_action, self.max_action)
        return action

    def update(self, batch):
        self.update_counter += 1
        states, actions, rewards, next_states, dones = batch
        
        self.critic_1.train()
        self.critic_2.train()
        self.target_actor.eval()
        self.target_critic_1.eval()
        self.target_critic_2.eval()
        
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.target_actor(next_states) + noise).clamp(-self.max_action, self.max_action)
            target_q1 = self.target_critic_1(next_states, next_actions)
            target_q2 = self.target_critic_2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            y = rewards + self.gamma * (1.0 - dones) * target_q
        
        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)
        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = torch.tensor(0.0)
        if self.update_counter % self.policy_delay == 0:
            self.critic_1.eval()
            self.actor.train()
            actor_actions = self.actor(states)
            actor_loss = -self.critic_1(states, actor_actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            polyak_update(self.target_actor, self.actor, self.tau)
            polyak_update(self.target_critic_1, self.critic_1, self.tau)
            polyak_update(self.target_critic_2, self.critic_2, self.tau)
        
        return actor_loss.item(), critic_loss.item()

class SAC_Agent:
    def __init__(self, state_dim, action_dim, params):
        self.action_dim = action_dim
        self.max_action = params['max_action']
        self.gamma = params['gamma']
        self.tau = params['tau']
        hidden1, hidden2 = params['hidden_dim1'], params['hidden_dim2']
        
        self.actor = StochasticActor(state_dim, hidden1, hidden2, action_dim, self.max_action).to(device)
        self.critic_1 = Critic(state_dim + action_dim, hidden1, hidden2, 1).to(device)
        self.critic_2 = Critic(state_dim + action_dim, hidden1, hidden2, 1).to(device)
        self.target_critic_1 = Critic(state_dim + action_dim, hidden1, hidden2, 1).to(device)
        self.target_critic_2 = Critic(state_dim + action_dim, hidden1, hidden2, 1).to(device)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        
        self.actor_optimizer = optim.SGD(self.actor.parameters(), lr=params['actor_lr'], momentum=0.9, weight_decay=params['l2_lambda'])
        self.critic_optimizer = optim.SGD(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), 
            lr=params['critic_lr'], momentum=0.9, weight_decay=params['l2_lambda']
        )
        
        self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.SGD([self.log_alpha], lr=params['alpha_lr'], momentum=0.9)
        self.alpha = self.log_alpha.exp().item()

    def select_action(self, state, exploration_noise=0.0):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action, _ = self.actor.sample(state_tensor)
        self.actor.train()
        return action.detach().cpu().numpy()[0]

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        self.critic_1.train()
        self.critic_2.train()
        self.target_critic_1.eval()
        self.target_critic_2.eval()
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
        
        self.actor.train()
        self.critic_1.eval()
        self.critic_2.eval()
        
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
        
        polyak_update(self.target_critic_1, self.critic_1, self.tau)
        polyak_update(self.target_critic_2, self.critic_2, self.tau)
        
        return actor_loss.item(), critic_loss.item()

#################################################################
# 訓練與評估 (Training and Evaluation)
#################################################################

def run_training(algorithm, params, data, data_info):
    """
    运行一次完整的训练 (Run one full training)
    """
    state_dim = 3 * 3 * 4
    action_dim = 3
    
    if algorithm == "DDPG":
        agent = DDPG_Agent(state_dim, action_dim, params)
    elif algorithm == "TD3":
        agent = TD3_Agent(state_dim, action_dim, params)
    elif algorithm == "SAC":
        agent = SAC_Agent(state_dim, action_dim, params)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
        
    replay_buffer = ReplayBuffer(params['buffer_size'], params['batch_size'])
    
    max_episodes = params['max_episodes']
    data_length = len(data)
    max_steps = min(300, data_length - 5)
    
    exploration_noise = params['exploration_noise']
    
    episode_rewards = []
    episode_mse = []
    
    for episode in range(max_episodes):
        max_start_idx = len(data) - max_steps - 4
        if max_start_idx < 4:
            break
        
        start_idx = np.random.randint(3, max_start_idx) 
        current_data = data.iloc[start_idx : start_idx + max_steps + 4].reset_index(drop=True)
        state = get_initial_state(current_data)
        episode_reward = 0
        episode_total_mse = 0
        step_count = 0
        prev_predicted = None
        
        for step in range(max_steps):
            step_count += 1
            noise = exploration_noise if algorithm in ["DDPG", "TD3"] else 0.0
            action = agent.select_action(state, noise)
            
            actual_idx = 4 + step
            done = (step == max_steps - 1)
            
            actual_row = current_data.iloc[actual_idx]
            actual_joints = actual_row[['joint1', 'joint2', 'joint3']].values
            prev_actual_row = current_data.iloc[actual_idx - 1]
            prev_actual = prev_actual_row[['joint1', 'joint2', 'joint3']].values
            
            reward = calculate_reward(action, actual_joints, prev_predicted, prev_actual, done)
            next_state = update_state(state, action)
            replay_buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_total_mse += np.mean((action - actual_joints)**2)
            prev_predicted = action
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_mse.append(episode_total_mse / step_count)
        
        if len(replay_buffer) >= 5 * params['batch_size']:
            updates = 5 if algorithm in ["DDPG", "TD3"] else 1
            for _ in range(updates):
                batch = replay_buffer.sample()
                if batch is not None:
                    agent.update(batch)

        exploration_noise = max(params['min_noise'], exploration_noise * params['noise_decay'])
        
        if (episode + 1) % 50 == 0:
            print(f'[{algorithm}] Episode: {episode+1}/{max_episodes}, Reward: {episode_reward:.2f}, MSE: {episode_mse[-1]:.4f}')
            
    test_ratio = 0.2
    test_start_idx = int(np.floor(len(data) * (1 - test_ratio)))
    test_data = data.iloc[test_start_idx:].reset_index(drop=True)
    test_size = len(test_data)

    state = get_initial_state(test_data)
    predicted_trajectory = np.zeros((test_size, action_dim))
    actual_trajectory = np.zeros((test_size, action_dim))

    for i in range(test_size):
        action = agent.select_action(state, 0.0) 
        predicted_trajectory[i, :] = action
        actual_trajectory[i, :] = [test_data.loc[i, 'joint1'], 
                                   test_data.loc[i, 'joint2'], 
                                   test_data.loc[i, 'joint3']]
        state = update_state(state, action)

    predicted_actual = np.zeros_like(predicted_trajectory)
    actual_actual = np.zeros_like(actual_trajectory)
    for i, joint_name in enumerate(['joint1', 'joint2', 'joint3']):
        sigma = data_info[joint_name]['sigma']
        mu = data_info[joint_name]['mu']
        predicted_actual[:, i] = predicted_trajectory[:, i] * sigma + mu
        actual_actual[:, i] = actual_trajectory[:, i] * sigma + mu

    mse_actual = np.mean((predicted_actual - actual_actual)**2)
    rmse_actual = np.sqrt(mse_actual)
    mae_actual = np.mean(np.abs(predicted_actual - actual_actual))
    
    print(f'[{algorithm}] Test Results (Actual Angle Space): RMSE: {rmse_actual:.4f}, MAE: {mae_actual:.4f}')
    
    test_results = {'rmse': rmse_actual, 'mae': mae_actual}
    training_history = {'rewards': episode_rewards, 'mse': episode_mse}
    
    return training_history, test_results

#################################################################
# Optuna 超參數優化 (Hyperparameter Optimization)
#################################################################

GLOBAL_DATA = None
GLOBAL_DATA_INFO = None

def objective(trial):
    params = {
        'max_episodes': 300,
        'buffer_size': 50000,
        'max_action': 1.0,
        'exploration_noise': 0.1,
        'noise_decay': 0.995,
        'min_noise': 0.01,
        'l2_lambda': 1e-4,
        'policy_delay': 2,
        'policy_noise': 0.2,
        'noise_clip': 0.5,
        'gamma': trial.suggest_categorical('gamma', [0.98, 0.99, 0.995]),
        'tau': trial.suggest_loguniform('tau', 1e-3, 1e-2),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
        'hidden_dim1': trial.suggest_categorical('hidden_dim1', [128, 256]),
        'hidden_dim2': trial.suggest_categorical('hidden_dim2', [64, 128, 256]),
        'actor_lr': trial.suggest_loguniform('actor_lr', 1e-4, 5e-3),
        'critic_lr': trial.suggest_loguniform('critic_lr', 1e-4, 5e-3),
        'alpha_lr': trial.suggest_loguniform('alpha_lr', 1e-5, 1e-3),
    }

    print(f"\nOptuna Trial: {trial.number}, Params: {params}")

    try:
        _, test_results = run_training(
            algorithm="SAC", 
            params=params, 
            data=GLOBAL_DATA, 
            data_info=GLOBAL_DATA_INFO
        )
        return test_results['rmse'] 
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise optuna.exceptions.TrialPruned()


#################################################################
# 主函數 (Main Function)
#################################################################

def main():
    global GLOBAL_DATA, GLOBAL_DATA_INFO

    # === 路径修复 (Path Fix) ===
    # 获去 .py 脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 组合 CSV 文件的完整路径
    csv_path = os.path.join(script_dir, 'motion_trajectory_dataset.csv')
    # === 修复结束 ===

    # %% 1. 數據加載與預處理 (Data Loading & Preprocessing)
    try:
        data = pd.read_csv(csv_path) # 使用修复后的路径
    except FileNotFoundError:
        print(f"Error: File not found at '{csv_path}'.") 
        print("Please make sure 'motion_trajectory_dataset.csv' is in the same directory as the script.")
        return
        
    print('Original data sample:') 
    print(data.head())
    
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
    for name in feature_names:
        mu = data[name].mean()
        sigma = data[name].std()
        if sigma < 1e-6: sigma = 1.0
        data[name] = (data[name] - mu) / sigma
        data_info[name] = {'mu': mu, 'sigma': sigma}
    
    GLOBAL_DATA = data
    GLOBAL_DATA_INFO = data_info
    
    # --- 步骤 2: (可選) 運行超參數優化 ---
    RUN_OPTUNA = False # 保持为 False 以跳过优化
    
    if RUN_OPTUNA:
        print("="*50)
        print("Starting Optuna Hyperparameter Optimization (Target: SAC)...") 
        print("="*50)
        
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPEsampler())
        study.enqueue_trial({
            'gamma': 0.99, 'tau': 0.005, 'batch_size': 256,
            'hidden_dim1': 256, 'hidden_dim2': 128,
            'actor_lr': 0.001, 'critic_lr': 0.001, 'alpha_lr': 0.0003
        })
        study.optimize(objective, n_trials=50) 

        print("\nOptuna optimization finished!")
        print(f"Best RMSE: {study.best_value:.4f}")
        print("Best Parameters:")
        print(study.best_params)
        
        best_sac_params = study.best_params
        
    else:
        print("Skipping Optuna optimization, using default parameters.")
        best_sac_params = {
            'gamma': 0.99, 'tau': 0.005, 'batch_size': 256,
            'hidden_dim1': 256, 'hidden_dim2': 128,
            'actor_lr': 0.001, 'critic_lr': 0.001, 'alpha_lr': 0.0003
        }

    # --- 步骤 3: 使用定義好的參數訓練所有演算法並進行比較 ---
    
    default_params = {
        'max_episodes': 800,
        'buffer_size': 50000,
        'batch_size': 256,
        'hidden_dim1': 256,
        'hidden_dim2': 128,
        'gamma': 0.99,
        'tau': 0.005,
        'actor_lr': 0.0005,
        'critic_lr': 0.001,
        'l2_lambda': 1e-4,
        'max_action': 1.0,
        'exploration_noise': 0.1,
        'noise_decay': 0.995,
        'min_noise': 0.01,
        'policy_delay': 2,
        'policy_noise': 0.2,
        'noise_clip': 0.5,
        'alpha_lr': best_sac_params.get('alpha_lr', 0.0003),
    }

    sac_params = default_params.copy()
    sac_params.update(best_sac_params)
    sac_params['max_episodes'] = 800
    
    all_histories = {}
    all_test_results = {}
    
    print("\n" + "="*50)
    print("Starting DDPG training...")
    print("="*50)
    start_time = time.time()
    hist_ddpg, test_ddpg = run_training("DDPG", default_params, data, data_info)
    all_histories['DDPG'] = hist_ddpg
    all_test_results['DDPG'] = test_ddpg
    print(f"DDPG training time: {time.time() - start_time:.2f} seconds")

    print("\n" + "="*50)
    print("Starting TD3 training...")
    print("="*50)
    start_time = time.time()
    hist_td3, test_td3 = run_training("TD3", default_params, data, data_info)
    all_histories['TD3'] = hist_td3
    all_test_results['TD3'] = test_td3
    print(f"TD3 training time: {time.time() - start_time:.2f} seconds")

    print("\n" + "="*50)
    print("Starting SAC training (with best params)...")
    print("="*50)
    start_time = time.time()
    hist_sac, test_sac = run_training("SAC", sac_params, data, data_info)
    all_histories['SAC'] = hist_sac
    all_test_results['SAC'] = test_sac
    print(f"SAC training time: {time.time() - start_time:.2f} seconds")

    # --- 步骤 4: 生成比較圖表 ---
    print("\n" + "="*50)
    print("All training complete. Generating comparison charts...")
    print("="*50)
    
    plt.figure(figsize=(12, 6))
    colors = {'DDPG': 'b', 'TD3': 'g', 'SAC': 'r'}
    for algo, history in all_histories.items():
        smooth_rewards = pd.Series(history['rewards']).rolling(20, min_periods=1).mean()
        plt.plot(smooth_rewards, label=algo, color=colors[algo])
        
    plt.title('Algorithm Comparison: Smoothed Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig("comparison_training_rewards.png")
    
    plt.figure(figsize=(8, 5))
    algos = list(all_test_results.keys())
    rmses = [res['rmse'] for res in all_test_results.values()]
    
    bars = plt.bar(algos, rmses, color=[colors[a] for a in algos])
    plt.ylabel('RMSE (Actual Angle Space)')
    plt.title('Algorithm Comparison: Final Test RMSE')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center')
        
    plt.savefig("comparison_final_rmse.png")

    print("Charts saved as 'comparison_training_rewards.png' and 'comparison_final_rmse.png'") 
    # plt.show()

if __name__ == "__main__":
    main()