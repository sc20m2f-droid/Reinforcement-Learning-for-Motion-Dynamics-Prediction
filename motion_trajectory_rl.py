import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# 设置 PyTorch 设备 (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#################################################################
# 辅助函数 (从 MATLAB 移植)
#################################################################

def kalman_filter(data, Q, R):
    """
    MATLAB kalman_filter 函数的 Python 实现
    """
    n = len(data)
    filtered = np.zeros(n)
    x_hat = data[0]
    P = 1.0  # 初始协方差

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
    """
    MATLAB calculate_reward 函数的 Python 实现
    """
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
        terminal_weight = 2.0
        reward = reward * terminal_weight
    
    reward = reward / (1 + abs(reward))
    return reward

def get_initial_state(data):
    """
    从数据中获取初始状态 (4 个时间步)
    """
    state = np.zeros(3 * 3 * 4)  # 3 关节 × 3 特征 × 4 时间步
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
    """
    通过滚动窗口并添加新动作来更新状态
    """
    new_state = np.zeros_like(old_state)
    # 将 t=1, 2, 3 的数据转移到 t=0, 1, 2
    new_state[0:27] = old_state[9:36]
    
    # 获取最近的(t=3)角度和速度
    prev_angles = old_state[27:30]
    prev_vels = old_state[30:33]
    
    # 在 t=3 的位置添加新数据 (来自 action)
    new_state[27] = action[0]  # joint1
    new_state[28] = action[1]  # joint2
    new_state[29] = action[2]  # joint3
    
    dt = 1.0
    # 计算新速度 (at t=3)
    new_state[30] = (action[0] - prev_angles[0]) / dt
    new_state[31] = (action[1] - prev_angles[1]) / dt
    new_state[32] = (action[2] - prev_angles[2]) / dt
    
    # 计算新加速度 (at t=3)
    new_state[33] = (new_state[30] - prev_vels[0]) / dt
    new_state[34] = (new_state[31] - prev_vels[1]) / dt
    new_state[35] = (new_state[32] - prev_vels[2]) / dt
    
    return new_state

def polyak_update(target, source, tau):
    """
    Polyak (软) 更新目标网络
    """
    with torch.no_grad():
        for target_param, source_param in zip(target.state_dict().values(), source.state_dict().values()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


#################################################################
# 强化学习组件 (PyTorch)
#################################################################

class ReplayBuffer:
    """
    优先经验回放 (PER) 缓冲区
    """
    def __init__(self, buffer_size, batch_size, alpha=0.6):
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.batch_size = batch_size
        self.alpha = alpha
        self.idx = 0
        self.count = 0
        self.buffer_size = buffer_size

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        
        # 新经验给予最大优先级
        max_priority = np.max(self.priorities) if self.count > 0 else 1.0
        
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.priorities[self.idx] = max_priority
            self.count += 1
        else:
            self.buffer[self.idx] = experience
            self.priorities[self.idx] = max_priority
        
        self.idx = (self.idx + 1) % self.buffer_size

    def sample(self, beta=0.4):
        if self.count < self.batch_size:
            return None, None, None
        
        prios = self.priorities[:self.count]
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(self.count, self.batch_size, p=probs)
        batch = [self.buffer[i] for i in indices]
        
        weights = (self.count * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return batch, indices, weights
    
    def update_priorities(self, indices, priorities):
        # 这里的 priorities 已经是 1D 数组 (scalar 列表)
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = np.abs(prio) + 1e-6  # Epsilon

    def __len__(self):
        return self.count

class Actor(nn.Module):
    """
    Actor 网络 (DDPG)
    """
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(Actor, self).__init__()
        # 匹配 MATLAB 的 BatchNorm 参数
        self.bn_eps = 1e-6
        self.bn_momentum = 0.1 # MATLAB: 0.9 * old + 0.1 * new
        
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1, eps=self.bn_eps, momentum=self.bn_momentum)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2, eps=self.bn_eps, momentum=self.bn_momentum)
        self.layer3 = nn.Linear(hidden_dim2, output_dim)
        self.init_weights()

    def init_weights(self):
        # 匹配 MATLAB 的 He 初始化 (randn * sqrt(2/input_dim))
        nn.init.kaiming_normal_(self.layer1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.layer1.bias)
        nn.init.kaiming_normal_(self.layer2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.layer2.bias)
        nn.init.kaiming_normal_(self.layer3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.layer3.bias)

    def forward(self, x):
        h1 = torch.tanh(self.bn1(self.layer1(x)))
        h2 = torch.tanh(self.bn2(self.layer2(h1)))
        output = torch.tanh(self.layer3(h2))  # 输出范围 [-1, 1]
        return output

class Critic(nn.Module):
    """
    Critic 网络 (DDPG)
    """
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
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
        q_value = self.layer3(h2)  # Q-value (unbounded)
        return q_value


#################################################################
# 主脚本
#################################################################

def main():
    
    # %% 1. 数据加载与预处理
    try:
        data = pd.read_csv('motion_trajectory_dataset.csv')
    except FileNotFoundError:
        print("错误: 找不到 'motion_trajectory_dataset.csv'。请确保文件在同一目录下。")
        return
        
    print('原始数据示例:')
    print(data.head())

    # 异常值去除 (3σ 准则)
    for joint in ['joint1', 'joint2', 'joint3']:
        mu = data[joint].mean()
        sigma = data[joint].std()
        outlier_idx = np.abs(data[joint] - mu) > 3 * sigma
        
        if outlier_idx.sum() > 0:
            for i in np.where(outlier_idx)[0]:
                if i == 0:
                    data.loc[i, joint] = data.loc[i + 1, joint]
                elif i == len(data) - 1:
                    data.loc[i, joint] = data.loc[i - 1, joint]
                else:
                    data.loc[i, joint] = (data.loc[i - 1, joint] + data.loc[i + 1, joint]) / 2

    # 处理缺失值 (线性插值)
    data.interpolate(method='linear', inplace=True)

    # 数据可视化 - 原始关节角度数据
    plt.figure()
    plt.plot(data['timestep'], data['joint1'], 'r', linewidth=1.5, label='关节1')
    plt.plot(data['timestep'], data['joint2'], 'g', linewidth=1.5, label='关节2')
    plt.plot(data['timestep'], data['joint3'], 'b', linewidth=1.5, label='关节3')
    plt.title('原始关节角度数据')
    plt.xlabel('时间步')
    plt.ylabel('关节角度')
    plt.legend()
    plt.grid(True)
    # plt.show() # 取消注释以显示图表

    # 卡尔曼滤波降噪
    data['joint1'] = kalman_filter(data['joint1'].values, 0.01, 0.1)
    data['joint2'] = kalman_filter(data['joint2'].values, 0.01, 0.12)
    data['joint3'] = kalman_filter(data['joint3'].values, 0.01, 0.09)

    # 计算速度和加速度
    dt = 1.0
    data['velocity1'] = np.concatenate(([0], np.diff(data['joint1']) / dt))
    data['velocity2'] = np.concatenate(([0], np.diff(data['joint2']) / dt))
    data['velocity3'] = np.concatenate(([0], np.diff(data['joint3']) / dt))

    data['acceleration1'] = np.concatenate(([0], np.diff(data['velocity1']) / dt))
    data['acceleration2'] = np.concatenate(([0], np.diff(data['velocity2']) / dt))
    data['acceleration3'] = np.concatenate(([0], np.diff(data['velocity3']) / dt))

    # 特征归一化
    feature_names = ['joint1', 'joint2', 'joint3', 'velocity1', 'velocity2', 'velocity3',
                     'acceleration1', 'acceleration2', 'acceleration3']
    data_info = {}

    for name in feature_names:
        mu = data[name].mean()
        sigma = data[name].std()
        if sigma < 1e-6:
            sigma = 1.0
        data[name] = (data[name] - mu) / sigma
        data_info[name] = {'mu': mu, 'sigma': sigma}

    # %% 数据可视化增强
    plt.figure(figsize=(10, 10))
    plt.subplot(3, 1, 1)
    plt.hist(data['joint1'], bins=30)
    plt.title('Joint 1 Angle Distribution')
    plt.xlabel('Normalized Angle')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.hist(data['joint2'], bins=30)
    plt.title('Joint 2 Angle Distribution')
    plt.xlabel('Normalized Angle')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.hist(data['joint3'], bins=30)
    plt.title('Joint 3 Angle Distribution')
    plt.xlabel('Normalized Angle')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.suptitle('Distribution of Joint Angles')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show() # 取消注释以显示图表

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(data['timestep'], data['velocity1'], 'r', linewidth=1, label='Joint 1')
    plt.plot(data['timestep'], data['velocity2'], 'g', linewidth=1, label='Joint 2')
    plt.plot(data['timestep'], data['velocity3'], 'b', linewidth=1, label='Joint 3')
    plt.title('Joint Velocities Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Velocity')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(data['timestep'], data['acceleration1'], 'r', linewidth=1, label='Joint 1')
    plt.plot(data['timestep'], data['acceleration2'], 'g', linewidth=1, label='Joint 2')
    plt.plot(data['timestep'], data['acceleration3'], 'b', linewidth=1, label='Joint 3')
    plt.title('Joint Accelerations Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.grid(True)
    plt.suptitle('Joint Velocities and Accelerations')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show() # 取消注释以显示图表

    # %% 2. 强化学习环境设置
    state_dim = 3 * 3 * 4  # 3 关节 × 3 特征 × 4 时间步 (36)
    action_dim = 3

    # %% 4. DDPG 算法实现
    
    # 超参数
    actor_lr = 0.0005
    critic_lr = 0.001
    gamma = 0.99
    tau = 0.002
    buffer_size = 50000
    batch_size = 128
    max_episodes = 800

    data_length = len(data)
    # 确保有足够数据 (4步初始 + 1步预测)
    max_steps = min(300, data_length - 5) 

    exploration_noise = 0.1
    noise_decay = 0.995
    min_noise = 0.01

    alpha = 0.6  # PER alpha
    beta = 0.4   # PER beta
    beta_increment = (1.0 - beta) / max_episodes
    
    # L2 惩罚 (来自 MATLAB 脚本，应用于 SGD)
    l2_lambda = 1e-4

    # 初始化网络
    actor = Actor(state_dim, 128, 64, action_dim).to(device)
    critic = Critic(state_dim + action_dim, 128, 64, 1).to(device)
    
    target_actor = Actor(state_dim, 128, 64, action_dim).to(device)
    target_critic = Critic(state_dim + action_dim, 128, 64, 1).to(device)

    # 复制参数到目标网络
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    
    # 设置目标网络为评估模式
    target_actor.eval()
    target_critic.eval()
    
    # 初始化优化器
    actor_optimizer = optim.SGD(actor.parameters(), lr=actor_lr, momentum=0.9, weight_decay=l2_lambda)
    critic_optimizer = optim.SGD(critic.parameters(), lr=critic_lr, momentum=0.9, weight_decay=l2_lambda)

    # 初始化经验回放缓冲区
    replay_buffer = ReplayBuffer(buffer_size, batch_size, alpha)

    # 训练指标
    episode_rewards = []
    episode_mse = []
    actor_losses = []
    critic_losses = []
    
    print("开始 DDPG 训练...")

    # 训练循环
    for episode in range(max_episodes):
        # 确保起始索引选择合理 (至少需要 4 步初始状态)
        max_start_idx = len(data) - max_steps - 4
        if max_start_idx < 4:
            print("数据集太短，无法训练。")
            break
        
        # Python 0-indexed, randint [low, high)
        start_idx = np.random.randint(3, max_start_idx) 
        current_data = data.iloc[start_idx : start_idx + max_steps + 4].reset_index(drop=True)
        
        state = get_initial_state(current_data)
        
        episode_reward = 0
        episode_total_mse = 0
        step_count = 0
        prev_predicted = None
        
        # 在收集数据/获取动作时，网络应处于评估模式 (.eval())
        actor.eval()
        critic.eval()

        for step in range(max_steps):
            step_count += 1
            
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
                action = actor(state_tensor).detach().cpu().numpy()[0]
            
            # 添加探索噪声
            noise = exploration_noise * np.random.randn(action_dim)
            action = (action + noise).clip(-1.0, 1.0)
            
            # --- 环境交互 ---
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
        
        # --- 网络更新 ---
        if len(replay_buffer) >= 5 * batch_size:
            batch, indices, weights = replay_buffer.sample(beta)
            
            if batch is not None:
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
                actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
                rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device).unsqueeze(1)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
                dones = torch.tensor(np.array(dones), dtype=torch.float32).to(device).unsqueeze(1)
                weights = torch.tensor(np.array(weights), dtype=torch.float32).to(device).unsqueeze(1)
                
                # --- Critic 更新 ---
                actor.train() 
                critic.train()
                target_actor.eval()
                target_critic.eval()
                
                with torch.no_grad():
                    next_actions = target_actor(next_states)
                    target_q = target_critic(next_states, next_actions)
                    y = rewards + gamma * (1.0 - dones) * target_q
                
                current_q = critic(states, actions)
                
                td_errors = y - current_q
                
                critic_loss = (weights * (td_errors ** 2)).mean()
                
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                
                # =================================================================
                # === 代码修正 (DeprecationWarning) ===
                # 使用 .flatten() 将 (batch_size, 1) 的 2D 阵列转换为 (batch_size,) 的 1D 阵列
                replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy().flatten())
                # =================================================================
                
                critic_losses.append(critic_loss.item())
                
                # --- Actor 更新 ---
                critic.eval() 
                actor.train()

                actor_actions = actor(states)
                actor_loss = -critic(states, actor_actions)
                actor_loss = (weights * actor_loss).mean()
                
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                
                actor_losses.append(actor_loss.item())

                # --- 更新目标网络 ---
                polyak_update(target_actor, actor, tau)
                polyak_update(target_critic, critic, tau)

        # 更新噪声和 beta
        exploration_noise = max(min_noise, exploration_noise * noise_decay)
        beta = min(1.0, beta + beta_increment)
        
        if (episode + 1) % 10 == 0:
            print(f'回合: {episode+1}/{max_episodes}, 奖励: {episode_reward:.2f}, '
                  f'MSE: {episode_mse[-1]:.4f}, 噪声: {exploration_noise:.3f}')

    # %% 6. 训练过程可视化
    print("训练完成。正在绘制训练指标...")
    plt.figure(figsize=(15, 10))
    
    def plot_metric(subplot_idx, data, title, color, smooth_color, label):
        plt.subplot(2, 2, subplot_idx)
        plt.plot(data, color=color, linewidth=1, label=f'Raw {label}')
        plt.title(title)
        plt.xlabel('Episode')
        plt.ylabel(label)
        plt.grid(True)
        # 使用 pandas rolling mean 进行平滑
        smooth_data = pd.Series(data).rolling(10, min_periods=1).mean()
        plt.plot(smooth_data, color=smooth_color, linewidth=1.5, label=f'Smoothed {label}')
        plt.legend()

    plot_metric(1, episode_rewards, 'Training Rewards Over Episodes', 'b', 'r', 'Reward')
    plot_metric(2, episode_mse, 'Mean Squared Error Over Episodes', 'g', 'm', 'MSE')
    plot_metric(3, actor_losses, 'Actor Loss Over Episodes', 'c', 'k', 'Loss')
    plot_metric(4, critic_losses, 'Critic Loss Over Episodes', 'y', 'r', 'Loss')

    plt.suptitle('Training Metrics')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("training_metrics.png")
    # plt.show()

    # %% 5. 模型评估
    print("开始模型评估...")
    test_ratio = 0.2
    test_start_idx = int(np.floor(len(data) * (1 - test_ratio)))
    test_data = data.iloc[test_start_idx:].reset_index(drop=True)
    test_size = len(test_data)

    state = get_initial_state(test_data)
    predicted_trajectory = np.zeros((test_size, action_dim))
    actual_trajectory = np.zeros((test_size, action_dim))

    # 设置 actor 为评估模式 (关闭 BatchNorm 更新)
    actor.eval()

    for i in range(test_size):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
            action = actor(state_tensor).detach().cpu().numpy()[0]
        
        predicted_trajectory[i, :] = action
        actual_trajectory[i, :] = [test_data.loc[i, 'joint1'], 
                                   test_data.loc[i, 'joint2'], 
                                   test_data.loc[i, 'joint3']]
        
        # 使用模型自己的预测来更新下一步的状态
        state = update_state(state, action)

    prediction_errors = np.abs(predicted_trajectory - actual_trajectory)

    # 反归一化
    predicted_actual = np.zeros_like(predicted_trajectory)
    actual_actual = np.zeros_like(actual_trajectory)

    for i, joint_name in enumerate(['joint1', 'joint2', 'joint3']):
        sigma = data_info[joint_name]['sigma']
        mu = data_info[joint_name]['mu']
        predicted_actual[:, i] = predicted_trajectory[:, i] * sigma + mu
        actual_actual[:, i] = actual_trajectory[:, i] * sigma + mu

    # 计算指标 (归一化空间)
    mse_norm = np.mean((predicted_trajectory - actual_trajectory)**2)
    rmse_norm = np.sqrt(mse_norm)
    mae_norm = np.mean(np.abs(predicted_trajectory - actual_trajectory))

    # 计算指标 (实际角度空间)
    mse_actual = np.mean((predicted_actual - actual_actual)**2)
    rmse_actual = np.sqrt(mse_actual)
    mae_actual = np.mean(np.abs(predicted_actual - actual_actual))

    print('\n测试结果 (归一化空间):')
    print(f'MSE: {mse_norm:.4f}')
    print(f'RMSE: {rmse_norm:.4f}')
    print(f'MAE: {mae_norm:.4f}')

    print('\n测试结果 (实际角度空间):')
    print(f'MSE: {mse_actual:.4f}')
    print(f'RMSE: {rmse_actual:.4f}')
    print(f'MAE: {mae_actual:.4f}')

    # 绘制评估结果
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    plt.plot(predicted_actual[:, 0], 'r', linewidth=1.5, label='Predicted')
    plt.plot(actual_actual[:, 0], 'b--', linewidth=1.5, label='Actual')
    plt.title('Joint 1 Angle Prediction vs Actual')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(predicted_actual[:, 1], 'r', linewidth=1.5, label='Predicted')
    plt.plot(actual_actual[:, 1], 'b--', linewidth=1.5, label='Actual')
    plt.title('Joint 2 Angle Prediction vs Actual')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(predicted_actual[:, 2], 'r', linewidth=1.5, label='Predicted')
    plt.plot(actual_actual[:, 2], 'b--', linewidth=1.5, label='Actual')
    plt.title('Joint 3 Angle Prediction vs Actual')
    plt.legend()
    plt.grid(True)
    plt.suptitle('Motion Trajectory Prediction Results (Actual Angle Space)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("prediction_results.png")
    # plt.show()

    # 绘制误差分析
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(prediction_errors[:, 0], 'r', linewidth=1, label='Joint 1')
    plt.plot(prediction_errors[:, 1], 'g', linewidth=1, label='Joint 2')
    plt.plot(prediction_errors[:, 2], 'b', linewidth=1, label='Joint 3')
    plt.title('Prediction Errors Over Time (Normalized)')
    plt.xlabel('Timestep')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    mean_errors = [np.mean(prediction_errors[:, 0]), 
                   np.mean(prediction_errors[:, 1]), 
                   np.mean(prediction_errors[:, 2])]
    plt.bar(['Joint 1', 'Joint 2', 'Joint 3'], mean_errors)
    plt.title('Average Prediction Error per Joint (Normalized)')
    plt.xlabel('Joint')
    plt.ylabel('Average Absolute Error')
    plt.grid(True)
    plt.suptitle('Prediction Error Analysis')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("prediction_error_analysis.png")
    # plt.show()
    
    print("\n所有图表已保存为 .png 文件。")
    print("如果您在 Jupyter 环境中，请取消注释 plt.show() 以显示图表。")


if __name__ == "__main__":
    main()