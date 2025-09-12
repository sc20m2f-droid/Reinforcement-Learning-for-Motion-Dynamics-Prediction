%% 基于强化学习的运动轨迹建模 

% 清理工作区并关闭所有图形窗口
clear; close all; clc;

%% 1. 数据加载与预处理

% 读取CSV数据文件
data = readtable('motion_trajectory_dataset.csv');  
% 数据探索 - 显示前几行数据
disp('原始数据示例:');
disp(head(data));

% 数据清洗
% 处理缺失值 - 线性插值
data = fillmissing(data, 'linear');

% 数据可视化 - 关节角度随时间变化 (保留原始可视化)
figure;
plot(data.timestep, data.joint1, 'r', 'LineWidth', 1.5);
hold on;
plot(data.timestep, data.joint2, 'g', 'LineWidth', 1.5);
plot(data.timestep, data.joint3, 'b', 'LineWidth', 1.5);
title('原始关节角度数据');
xlabel('时间步');
ylabel('关节角度');
legend('关节1', '关节2', '关节3');
grid on;

% 数据降噪 - 使用移动平均滤波器
windowSize = 5; 
data.joint1 = smoothdata(data.joint1, 'movmean', windowSize);
data.joint2 = smoothdata(data.joint2, 'movmean', windowSize);
data.joint3 = smoothdata(data.joint3, 'movmean', windowSize);

% 特征工程 - 计算速度和加速度
dt = 1; % 假设时间步长为1
data.velocity1 = zeros(height(data), 1);
data.velocity2 = zeros(height(data), 1);
data.velocity3 = zeros(height(data), 1);
data.acceleration1 = zeros(height(data), 1);
data.acceleration2 = zeros(height(data), 1);
data.acceleration3 = zeros(height(data), 1);

for i = 2:height(data)
    data.velocity1(i) = (data.joint1(i) - data.joint1(i-1))/dt;
    data.velocity2(i) = (data.joint2(i) - data.joint2(i-1))/dt;
    data.velocity3(i) = (data.joint3(i) - data.joint3(i-1))/dt;
    
    data.acceleration1(i) = (data.velocity1(i) - data.velocity1(i-1))/dt;
    data.acceleration2(i) = (data.velocity2(i) - data.velocity2(i-1))/dt;
    data.acceleration3(i) = (data.velocity3(i) - data.velocity3(i-1))/dt;
end

% 数据归一化
data.joint1 = (data.joint1 - mean(data.joint1))/std(data.joint1);
data.joint2 = (data.joint2 - mean(data.joint2))/std(data.joint2);
data.joint3 = (data.joint3 - mean(data.joint3))/std(data.joint3);

%% 新增可视化1：关节角度分布直方图 (英文图例)
figure;
subplot(3,1,1);
histogram(data.joint1, 30);
title('Joint 1 Angle Distribution');
xlabel('Normalized Angle');
ylabel('Frequency');
grid on;

subplot(3,1,2);
histogram(data.joint2, 30);
title('Joint 2 Angle Distribution');
xlabel('Normalized Angle');
ylabel('Frequency');
grid on;

subplot(3,1,3);
histogram(data.joint3, 30);
title('Joint 3 Angle Distribution');
xlabel('Normalized Angle');
ylabel('Frequency');
grid on;
sgtitle('Distribution of Joint Angles');

%% 新增可视化2：速度和加速度对比图 (英文图例)
figure;
subplot(2,1,1);
plot(data.timestep, data.velocity1, 'r', 'LineWidth', 1);
hold on;
plot(data.timestep, data.velocity2, 'g', 'LineWidth', 1);
plot(data.timestep, data.velocity3, 'b', 'LineWidth', 1);
title('Joint Velocities Over Time');
xlabel('Timestep');
ylabel('Velocity');
legend('Joint 1', 'Joint 2', 'Joint 3');
grid on;

subplot(2,1,2);
plot(data.timestep, data.acceleration1, 'r', 'LineWidth', 1);
hold on;
plot(data.timestep, data.acceleration2, 'g', 'LineWidth', 1);
plot(data.timestep, data.acceleration3, 'b', 'LineWidth', 1);
title('Joint Accelerations Over Time');
xlabel('Timestep');
ylabel('Acceleration');
legend('Joint 1', 'Joint 2', 'Joint 3');
grid on;
sgtitle('Joint Velocities and Accelerations');

%% 2. 强化学习环境设置

% 定义状态空间维度 (当前关节角度 + 速度 + 加速度 + 历史3个时间步)
state_dim = 3*4; % 3个关节*(角度+速度+加速度) + 3个历史时间步

% 定义动作空间维度 (预测下一个时间步的3个关节角度)
action_dim = 3;

%% 3. 神经网络模型与经验回放缓冲区函数定义

% 初始化经验回放缓冲区
function buffer = replay_buffer_init(buffer_size, batch_size)
    buffer.buffer_size = buffer_size;
    buffer.batch_size = batch_size;
    buffer.buffer = cell(buffer_size, 1);
    buffer.idx = 1;
    buffer.count = 0;
end

% 向经验回放缓冲区添加经验
function buffer = replay_buffer_add(buffer, state, action, reward, next_state, done)
    experience.state = state;
    experience.action = action;
    experience.reward = reward;
    experience.next_state = next_state;
    experience.done = done;
    
    buffer.buffer{buffer.idx} = experience;
    buffer.idx = mod(buffer.idx, buffer.buffer_size) + 1;
    buffer.count = min(buffer.count + 1, buffer.buffer_size);
end

% 从经验回放缓冲区采样
function batch = replay_buffer_sample(buffer)
    batch = [];
    if buffer.count < buffer.batch_size
        return;
    end
    
    indices = randi([1, buffer.count], 1, buffer.batch_size);
    batch = cell(buffer.batch_size, 1);
    
    for i = 1:buffer.batch_size
        batch{i} = buffer.buffer{indices(i)};
    end
end

% 初始化行动者网络
function actor = actor_network_init(input_dim, hidden_dim, output_dim, learning_rate)
    actor.weights{1} = randn(hidden_dim, input_dim) * sqrt(2/input_dim);
    actor.weights{2} = randn(output_dim, hidden_dim) * sqrt(2/hidden_dim);
    
    actor.biases{1} = zeros(hidden_dim, 1);
    actor.biases{2} = zeros(output_dim, 1);
    
    actor.learning_rate = learning_rate;
end

% 行动者网络前向传播
function output = actor_forward(actor, x)
    h = tanh(actor.weights{1} * x + actor.biases{1});
    output = tanh(actor.weights{2} * h + actor.biases{2}); % 输出在[-1,1]范围内
end

% 更新行动者网络
function actor = actor_update(actor, states, actions, critic, batch_size)
    for i = 1:batch_size
        state = states{i};
        action = actions{i};
        
        % 前向传播
        h = tanh(actor.weights{1} * state + actor.biases{1});
        output = tanh(actor.weights{2} * h + actor.biases{2});
        
        % 计算梯度
        d_output = -critic_q_gradient(critic, state, output);
        d_output = d_output .* (1 - output.^2); % tanh导数
        
        d_h = actor.weights{2}' * d_output;
        d_h = d_h .* (1 - h.^2); % tanh导数
        
        % 更新权重和偏置
        actor.weights{2} = actor.weights{2} - actor.learning_rate * (d_output * h');
        actor.biases{2} = actor.biases{2} - actor.learning_rate * d_output;
        
        actor.weights{1} = actor.weights{1} - actor.learning_rate * (d_h * state');
        actor.biases{1} = actor.biases{1} - actor.learning_rate * d_h;
    end
end

% 初始化评论家网络
function critic = critic_network_init(input_dim, hidden_dim, output_dim, learning_rate)
    critic.weights{1} = randn(hidden_dim, input_dim) * sqrt(2/input_dim);
    critic.weights{2} = randn(output_dim, hidden_dim) * sqrt(2/hidden_dim);
    
    critic.biases{1} = zeros(hidden_dim, 1);
    critic.biases{2} = zeros(output_dim, 1);
    
    critic.learning_rate = learning_rate;
end

% 计算Q值
function q_value = critic_q_value(critic, state, action)
    x = [state; action];
    h = tanh(critic.weights{1} * x + critic.biases{1});
    q_value = critic.weights{2} * h + critic.biases{2};
end

% 计算Q值关于动作的梯度
function grad = critic_q_gradient(critic, state, action)
    x = [state; action];
    h = tanh(critic.weights{1} * x + critic.biases{1});
    
    d_h = critic.weights{2}' * 1; % 假设输出是标量
    d_h = d_h .* (1 - h.^2); % tanh导数
    
    grad = critic.weights{1}(:, end-length(action)+1:end)' * d_h;
end

% 更新评论家网络
function critic = critic_update(critic, states, actions, rewards, next_states, dones, target_actor, target_critic, gamma, batch_size)
    for i = 1:batch_size
        state = states{i};
        action = actions{i};
        reward = rewards{i};
        next_state = next_states{i};
        done = dones{i};
        
        % 计算目标Q值
        next_action = actor_forward(target_actor, next_state);
        target_q = critic_q_value(target_critic, next_state, next_action);
        y = reward + gamma * (1 - done) * target_q;
        
        % 计算当前Q值
        current_q = critic_q_value(critic, state, action);
        
        % 计算误差
        error = y - current_q;
        
        % 前向传播
        x = [state; action];
        h = tanh(critic.weights{1} * x + critic.biases{1});
        
        % 计算梯度
        d_output = -error;
        d_h = critic.weights{2}' * d_output;
        d_h = d_h .* (1 - h.^2); % tanh导数
        
        % 更新权重和偏置
        critic.weights{2} = critic.weights{2} - critic.learning_rate * (d_output * h');
        critic.biases{2} = critic.biases{2} - critic.learning_rate * d_output;
        
        critic.weights{1} = critic.weights{1} - critic.learning_rate * (d_h * x');
        critic.biases{1} = critic.biases{1} - critic.learning_rate * d_h;
    end
end

% 定义奖励函数
function reward = calculate_reward(predicted, actual)
    % 主要奖励: 负的均方误差
    mse = mean((predicted - actual).^2);
    reward = -mse;
    
    % 添加平滑度惩罚
    if length(predicted) > 1
        velocity_change = diff(predicted);
        reward = reward - 0.1*mean(velocity_change.^2); % 平滑度惩罚系数
    end
end

%% 4. DDPG算法实现

% DDPG参数设置
actor_lr = 0.001;    % 行动者学习率
critic_lr = 0.002;   % 评论家学习率
gamma = 0.99;        % 折扣因子
tau = 0.005;         % 软更新参数
buffer_size = 10000; % 回放缓冲区大小
batch_size = 64;     % 批量大小
max_episodes = 500;  % 最大训练回合数
max_steps = 200;     % 每回合最大步数
exploration_noise = 0.1; % 探索噪声

% 初始化网络
actor = actor_network_init(state_dim, 64, action_dim, actor_lr);
critic = critic_network_init(state_dim + action_dim, 64, 1, critic_lr);

target_actor = actor_network_init(state_dim, 64, action_dim, actor_lr);
target_critic = critic_network_init(state_dim + action_dim, 64, 1, critic_lr);

% 初始化目标网络 (与主网络相同)
target_actor.weights = actor.weights;
target_actor.biases = actor.biases;
target_critic.weights = critic.weights;
target_critic.biases = critic.biases;

% 初始化回放缓冲区
replay_buffer = replay_buffer_init(buffer_size, batch_size);

% 存储训练奖励用于可视化
episode_rewards = zeros(max_episodes, 1);

% 训练循环
for episode = 1:max_episodes
    % 重置环境 (在这里是重置数据指针)
    state = get_initial_state(data); % 自定义函数获取初始状态
    
    episode_reward = 0;
    
    for step = 1:max_steps
        % 选择动作 (添加探索噪声)
        action = actor_forward(actor, state);
        noise = exploration_noise * randn(size(action));
        action = action + noise;
        action = min(max(action, -1), 1); % 限制在[-1,1]范围内
        
        % 执行动作 (预测下一个状态)
        [next_state, reward, done] = step_environment(data, state, action); % 自定义函数
        
        % 存储经验
        replay_buffer = replay_buffer_add(replay_buffer, state, action, reward, next_state, done);
        
        % 更新状态
        state = next_state;
        episode_reward = episode_reward + reward;
        
        % 如果回合结束，跳出循环
        if done
            break;
        end
    end
    
    % 存储奖励
    episode_rewards(episode) = episode_reward;
    
    % 从回放缓冲区采样并更新网络
    batch = replay_buffer_sample(replay_buffer);
    if ~isempty(batch)
        states = cellfun(@(x) x.state, batch, 'UniformOutput', false);
        actions = cellfun(@(x) x.action, batch, 'UniformOutput', false);
        rewards = cellfun(@(x) x.reward, batch, 'UniformOutput', false);
        next_states = cellfun(@(x) x.next_state, batch, 'UniformOutput', false);
        dones = cellfun(@(x) x.done, batch, 'UniformOutput', false);
        
        % 更新评论家
        critic = critic_update(critic, states, actions, rewards, next_states, dones, target_actor, target_critic, gamma, batch_size);
        
        % 更新行动者
        actor = actor_update(actor, states, actions, critic, batch_size);
        
        % 软更新目标网络
        for i = 1:length(target_actor.weights)
            target_actor.weights{i} = tau * actor.weights{i} + (1 - tau) * target_actor.weights{i};
            target_actor.biases{i} = tau * actor.biases{i} + (1 - tau) * target_actor.biases{i};
        end
        
        for i = 1:length(target_critic.weights)
            target_critic.weights{i} = tau * critic.weights{i} + (1 - tau) * target_critic.weights{i};
            target_critic.biases{i} = tau * critic.biases{i} + (1 - tau) * target_critic.biases{i};
        end
    end
    
    % 打印训练进度
    fprintf('回合: %d, 奖励: %.2f\n', episode, episode_reward);
end

%% 新增可视化3：训练奖励曲线 (英文图例)
figure;
plot(episode_rewards, 'b', 'LineWidth', 1);
title('Training Rewards Over Episodes');
xlabel('Episode');
ylabel('Total Reward');
grid on;
% 添加滑动平均曲线使趋势更清晰
smoothed_rewards = smoothdata(episode_rewards, 'movmean', 10);
hold on;
plot(smoothed_rewards, 'r', 'LineWidth', 1.5);
legend('Raw Reward', 'Smoothed Reward');

%% 5. 模型评估
% 测试数据准备
test_data = data(end-100:end, :); % 假设最后100个样本是测试数据
test_size = height(test_data); % 获取测试数据的行数

% 初始化状态
state = get_initial_state(test_data);
predicted_trajectory = zeros(test_size, action_dim);
actual_trajectory = zeros(test_size, action_dim);
prediction_errors = zeros(test_size, action_dim); % 存储每个关节的预测误差

for i = 1:test_size
    % 使用训练好的行动者网络预测动作
    action = actor_forward(actor, state);
    
    % 存储预测、实际值和误差
    predicted_trajectory(i, :) = action';
    actual_trajectory(i, :) = [test_data.joint1(i), test_data.joint2(i), test_data.joint3(i)];
    prediction_errors(i, :) = abs(predicted_trajectory(i, :) - actual_trajectory(i, :));
    
    % 更新状态
    state = update_state(state, action, test_data, i);
end

% 计算评估指标
mse = mean(mean((predicted_trajectory - actual_trajectory).^2));
rmse = sqrt(mse);
mae = mean(mean(abs(predicted_trajectory - actual_trajectory)));

fprintf('测试结果:\n');
fprintf('MSE: %.4f\n', mse);
fprintf('RMSE: %.4f\n', rmse);
fprintf('MAE: %.4f\n', mae);

% 可视化预测结果与实际结果 (保留原始可视化，新增英文图例版本)
figure;
subplot(3,1,1);
plot(predicted_trajectory(:,1), 'r', 'LineWidth', 1.5);
hold on;
plot(actual_trajectory(:,1), 'b--', 'LineWidth', 1.5);
title('Joint 1 Angle Prediction vs Actual');
legend('Predicted', 'Actual');
grid on;

subplot(3,1,2);
plot(predicted_trajectory(:,2), 'r', 'LineWidth', 1.5);
hold on;
plot(actual_trajectory(:,2), 'b--', 'LineWidth', 1.5);
title('Joint 2 Angle Prediction vs Actual');
legend('Predicted', 'Actual');
grid on;

subplot(3,1,3);
plot(predicted_trajectory(:,3), 'r', 'LineWidth', 1.5);
hold on;
plot(actual_trajectory(:,3), 'b--', 'LineWidth', 1.5);
title('Joint 3 Angle Prediction vs Actual');
legend('Predicted', 'Actual');
grid on;
sgtitle('Motion Trajectory Prediction Results');

%% 新增可视化4：预测误差分析 (英文图例)
figure;
subplot(2,1,1);
plot(prediction_errors(:,1), 'r', 'LineWidth', 1);
hold on;
plot(prediction_errors(:,2), 'g', 'LineWidth', 1);
plot(prediction_errors(:,3), 'b', 'LineWidth', 1);
title('Prediction Errors Over Time');
xlabel('Timestep');
ylabel('Absolute Error');
legend('Joint 1', 'Joint 2', 'Joint 3');
grid on;

subplot(2,1,2);
bar([mean(prediction_errors(:,1)), mean(prediction_errors(:,2)), mean(prediction_errors(:,3))]);
title('Average Prediction Error per Joint');
xlabel('Joint');
ylabel('Average Absolute Error');
set(gca, 'XTick', 1:3, 'XTickLabel', {'Joint 1', 'Joint 2', 'Joint 3'});
grid on;
sgtitle('Prediction Error Analysis');

%% 新增可视化5：状态空间特征相关性分析 (英文图例)
% 提取状态特征样本用于相关性分析
feature_samples = [];
for i = 1:min(100, height(data))
    state = get_initial_state(data(i:end, :));
    feature_samples = [feature_samples; state'];
end

% 计算特征相关性
corr_matrix = corr(feature_samples);

% 可视化相关性热图
figure;
heatmap(corr_matrix, 'Title', 'State Feature Correlation Matrix', ...
        'XLabel', 'Feature Index', 'YLabel', 'Feature Index');
colorbar;

%% 辅助函数定义

function state = get_initial_state(data)
    % 获取初始状态 (当前关节角度+速度+加速度+历史3个时间步)
    state = zeros(3*4, 1); % 3个关节*(角度+速度+加速度) + 3个历史时间步
    
    % 填充初始状态
    state(1) = data.joint1(1);
    state(2) = data.joint2(1);
    state(3) = data.joint3(1);
    
    state(4) = data.velocity1(1);
    state(5) = data.velocity2(1);
    state(6) = data.velocity3(1);
    
    state(7) = data.acceleration1(1);
    state(8) = data.acceleration2(1);
    state(9) = data.acceleration3(1);
    
    % 历史数据 (使用初始值填充)
    if height(data) >= 2
        state(10) = data.joint1(min(2, height(data)));
        state(11) = data.joint2(min(2, height(data)));
        state(12) = data.joint3(min(2, height(data)));
    else
        state(10:12) = state(1:3); % 如果数据不足，重复当前值
    end
end

function [next_state, reward, done] = step_environment(data, state, action)
    % 模拟环境步骤
    % 获取当前时间步索引 (从状态历史中推断)
    current_step = find(data.joint1 == state(1), 1);
    if isempty(current_step) || current_step >= height(data)
        current_step = randi([1, height(data)-1]);
    end
    
    % 预测下一个关节角度
    predicted_joints = action'; 
    
    % 实际下一个关节角度 (从数据中获取)
    actual_joints = [data.joint1(current_step+1), data.joint2(current_step+1), data.joint3(current_step+1)];
    
    % 计算奖励
    reward = calculate_reward(predicted_joints, actual_joints);
    
    % 构建下一个状态
    next_state = zeros(size(state));
    
    % 更新当前关节角度
    next_state(1:3) = predicted_joints';
    
    % 更新速度 (当前角度变化率)
    next_state(4:6) = (predicted_joints' - state(1:3)) / 1; % dt=1
    
    % 更新加速度 (速度变化率)
    next_state(7:9) = (next_state(4:6) - state(4:6)) / 1; % dt=1
    
    % 更新历史数据 (滑动窗口)
    next_state(10:12) = state(1:3);  % 将当前状态移至历史
    
    % 判断是否结束
    done = (current_step + 1) >= height(data);
end

function new_state = update_state(old_state, action, data, current_idx)
    % 更新状态
    new_state = zeros(size(old_state));
    
    % 更新当前关节角度
    new_state(1:3) = action;
    
    % 更新速度
    if current_idx > 1
        prev_joints = [data.joint1(current_idx-1); data.joint2(current_idx-1); data.joint3(current_idx-1)];
        new_state(4:6) = (action - prev_joints) / 1; % dt=1
    else
        new_state(4:6) = old_state(4:6); % 初始速度保持不变
    end
    
    % 更新加速度
    new_state(7:9) = (new_state(4:6) - old_state(4:6)) / 1; % dt=1
    
    % 更新历史数据 (滑动窗口)
    new_state(10:12) = old_state(1:3);  % 将当前状态移至历史
end
