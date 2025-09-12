%% 基于强化学习的运动轨迹建模

% 清理工作区并关闭所有图形窗口
clear; close all; clc;

%% 1. 数据加载与预处理
data = readtable('motion_trajectory_dataset.csv');  
disp('原始数据示例:');
disp(head(data));

% 异常值去除（3σ准则）
for joint = {'joint1', 'joint2', 'joint3'}
    joint_name = joint{1};
    mu = mean(data.(joint_name));
    sigma = std(data.(joint_name));
    outlier_idx = abs(data.(joint_name) - mu) > 3*sigma;
    if sum(outlier_idx) > 0
        for i = find(outlier_idx)'
            if i == 1
                data.(joint_name)(i) = data.(joint_name)(i+1);
            elseif i == height(data)
                data.(joint_name)(i) = data.(joint_name)(i-1);
            else
                data.(joint_name)(i) = (data.(joint_name)(i-1) + data.(joint_name)(i+1))/2;
            end
        end
    end
end

% 处理缺失值
data = fillmissing(data, 'linear');

% 数据可视化 - 原始关节角度数据
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

% 卡尔曼滤波降噪
function filtered = kalman_filter(data, Q, R)
    n = length(data);
    filtered = zeros(n, 1);
    x_hat = data(1);
    P = 1; % 初始协方差
    
    for k = 2:n
        x_hat_minus = x_hat;
        P_minus = P + Q;
        K = P_minus / (P_minus + R);
        x_hat = x_hat_minus + K * (data(k) - x_hat_minus);
        P = (1 - K) * P_minus;
        filtered(k) = x_hat;
    end
    filtered(1) = data(1);
end

data.joint1 = kalman_filter(data.joint1, 0.01, 0.1);
data.joint2 = kalman_filter(data.joint2, 0.01, 0.12);
data.joint3 = kalman_filter(data.joint3, 0.01, 0.09);

% 计算速度和加速度
dt = 1;
data.velocity1 = [0; diff(data.joint1)/dt];
data.velocity2 = [0; diff(data.joint2)/dt];
data.velocity3 = [0; diff(data.joint3)/dt];

data.acceleration1 = [0; diff(data.velocity1)/dt];
data.acceleration2 = [0; diff(data.velocity2)/dt];
data.acceleration3 = [0; diff(data.velocity3)/dt];

% 特征归一化
feature_names = {'joint1', 'joint2', 'joint3', 'velocity1', 'velocity2', 'velocity3', ...
                 'acceleration1', 'acceleration2', 'acceleration3'};
data_info = struct();

for name = feature_names
    mu = mean(data.(name{1}));
    sigma = std(data.(name{1}));
    if sigma < 1e-6
        sigma = 1;
    end
    data.(name{1}) = (data.(name{1}) - mu) / sigma;
    data_info.(name{1}).mu = mu;
    data_info.(name{1}).sigma = sigma;
end

%% 数据可视化增强
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
state_dim = 3*3*4; % 3关节 × 3特征 × 4时间步
action_dim = 3;

%% 3. 神经网络模型与经验回放缓冲区函数定义
function buffer = replay_buffer_init(buffer_size, batch_size)
    buffer.buffer_size = buffer_size;
    buffer.batch_size = batch_size;
    buffer.buffer = cell(buffer_size, 1);
    buffer.idx = 1;
    buffer.count = 0;
end

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

function [batch, indices, weights] = replay_buffer_sample(buffer, beta)
    batch = [];
    indices = [];
    weights = [];
    
    if buffer.count < buffer.batch_size
        return;
    end
    
    if isfield(buffer, 'priorities') && ~isempty(buffer.priorities)
        priorities = buffer.priorities(1:buffer.count);
        probabilities = priorities / sum(priorities);
        indices = randsample(buffer.count, buffer.batch_size, true, probabilities);
        weights = (buffer.count * probabilities(indices)).^(-beta);
        weights = weights / max(weights);
    else
        indices = randi([1, buffer.count], 1, buffer.batch_size);
        weights = ones(1, buffer.batch_size);
    end
    
    batch = cell(buffer.batch_size, 1);
    for i = 1:buffer.batch_size
        batch{i} = buffer.buffer{indices(i)};
    end
end

function buffer = replay_buffer_update_priorities(buffer, indices, priorities)
    if ~isfield(buffer, 'priorities')
        buffer.priorities = zeros(buffer.buffer_size, 1);
    end
    
    for i = 1:length(indices)
        buffer.priorities(indices(i)) = priorities(i) + 1e-6;
    end
end

function actor = actor_network_init(input_dim, hidden_dim1, hidden_dim2, output_dim, learning_rate)
    actor.weights{1} = randn(hidden_dim1, input_dim) * sqrt(2/input_dim);
    actor.weights{2} = randn(hidden_dim2, hidden_dim1) * sqrt(2/hidden_dim1);
    actor.weights{3} = randn(output_dim, hidden_dim2) * sqrt(2/hidden_dim2);
    
    actor.biases{1} = zeros(hidden_dim1, 1);
    actor.biases{2} = zeros(hidden_dim2, 1);
    actor.biases{3} = zeros(output_dim, 1);
    
    actor.bn_mean{1} = zeros(hidden_dim1, 1);
    actor.bn_var{1} = ones(hidden_dim1, 1);
    actor.bn_beta{1} = zeros(hidden_dim1, 1);
    actor.bn_gamma{1} = ones(hidden_dim1, 1);
    
    actor.bn_mean{2} = zeros(hidden_dim2, 1);
    actor.bn_var{2} = ones(hidden_dim2, 1);
    actor.bn_beta{2} = zeros(hidden_dim2, 1);
    actor.bn_gamma{2} = ones(hidden_dim2, 1);
    
    actor.learning_rate = learning_rate;
    actor.training = true;
end

function [output, h1, h2] = actor_forward(actor, x)
    z1 = actor.weights{1} * x + actor.biases{1};
    
    if actor.training
        mu = mean(z1, 2);
        sigma = std(z1, 0, 2) + 1e-6;
        actor.bn_mean{1} = 0.9 * actor.bn_mean{1} + 0.1 * mu;
        actor.bn_var{1} = 0.9 * actor.bn_var{1} + 0.1 * sigma.^2;
        z1_norm = (z1 - mu) ./ sqrt(sigma.^2 + 1e-6);
    else
        z1_norm = (z1 - actor.bn_mean{1}) ./ sqrt(actor.bn_var{1} + 1e-6);
    end
    z1_bn = actor.bn_gamma{1} .* z1_norm + actor.bn_beta{1};
    h1 = tanh(z1_bn);
    
    z2 = actor.weights{2} * h1 + actor.biases{2};
    
    if actor.training
        mu = mean(z2, 2);
        sigma = std(z2, 0, 2) + 1e-6;
        actor.bn_mean{2} = 0.9 * actor.bn_mean{2} + 0.1 * mu;
        actor.bn_var{2} = 0.9 * actor.bn_var{2} + 0.1 * sigma.^2;
        z2_norm = (z2 - mu) ./ sqrt(sigma.^2 + 1e-6);
    else
        z2_norm = (z2 - actor.bn_mean{2}) ./ sqrt(actor.bn_var{2} + 1e-6);
    end
    z2_bn = actor.bn_gamma{2} .* z2_norm + actor.bn_beta{2};
    h2 = tanh(z2_bn);
    
    output = tanh(actor.weights{3} * h2 + actor.biases{3});
end

function [actor, loss] = actor_update(actor, states, actions, critic, batch_size, weights)
    loss = 0;
    if nargin < 6
        weights = ones(1, batch_size);
    end
    
    grad_weights1 = zeros(size(actor.weights{1}));
    grad_biases1 = zeros(size(actor.biases{1}));
    grad_weights2 = zeros(size(actor.weights{2}));
    grad_biases2 = zeros(size(actor.biases{2}));
    grad_weights3 = zeros(size(actor.weights{3}));
    grad_biases3 = zeros(size(actor.biases{3}));
    
    for i = 1:batch_size
        state = states{i};
        action = actions{i};
        
        [output, h1, h2] = actor_forward(actor, state);
        
        d_output = -critic_q_gradient(critic, state, output);
        d_output = d_output .* (1 - output.^2);
        
        d_weights3 = weights(i) * d_output * h2';
        d_biases3 = weights(i) * d_output;
        
        d_h2 = actor.weights{3}' * d_output;
        d_h2 = d_h2 .* (1 - h2.^2);
        
        d_weights2 = weights(i) * d_h2 * h1';
        d_biases2 = weights(i) * d_h2;
        
        d_h1 = actor.weights{2}' * d_h2;
        d_h1 = d_h1 .* (1 - h1.^2);
        
        d_weights1 = weights(i) * d_h1 * state';
        d_biases1 = weights(i) * d_h1;
        
        grad_weights1 = grad_weights1 + d_weights1;
        grad_biases1 = grad_biases1 + d_biases1;
        grad_weights2 = grad_weights2 + d_weights2;
        grad_biases2 = grad_biases2 + d_biases2;
        grad_weights3 = grad_weights3 + d_weights3;
        grad_biases3 = grad_biases3 + d_biases3;
        
        loss = loss + weights(i) * (-critic_q_value(critic, state, output));
    end
    
    grad_weights1 = grad_weights1 / batch_size;
    grad_biases1 = grad_biases1 / batch_size;
    grad_weights2 = grad_weights2 / batch_size;
    grad_biases2 = grad_biases2 / batch_size;
    grad_weights3 = grad_weights3 / batch_size;
    grad_biases3 = grad_biases3 / batch_size;
    
    loss = loss / batch_size;
    
    l2_lambda = 1e-4;
    loss = loss + l2_lambda * (sum(grad_weights1(:).^2) + sum(grad_weights2(:).^2) + sum(grad_weights3(:).^2));
    
    if ~isfield(actor, 'momentum')
        actor.momentum.w1 = 0;
        actor.momentum.b1 = 0;
        actor.momentum.w2 = 0;
        actor.momentum.b2 = 0;
        actor.momentum.w3 = 0;
        actor.momentum.b3 = 0;
        actor.momentum.beta = 0.9;
    end
    
    actor.momentum.w1 = actor.momentum.beta * actor.momentum.w1 + (1 - actor.momentum.beta) * grad_weights1;
    actor.momentum.b1 = actor.momentum.beta * actor.momentum.b1 + (1 - actor.momentum.beta) * grad_biases1;
    actor.momentum.w2 = actor.momentum.beta * actor.momentum.w2 + (1 - actor.momentum.beta) * grad_weights2;
    actor.momentum.b2 = actor.momentum.beta * actor.momentum.b2 + (1 - actor.momentum.beta) * grad_biases2;
    actor.momentum.w3 = actor.momentum.beta * actor.momentum.w3 + (1 - actor.momentum.beta) * grad_weights3;
    actor.momentum.b3 = actor.momentum.beta * actor.momentum.b3 + (1 - actor.momentum.beta) * grad_biases3;
    
    actor.weights{1} = actor.weights{1} - actor.learning_rate * actor.momentum.w1;
    actor.biases{1} = actor.biases{1} - actor.learning_rate * actor.momentum.b1;
    actor.weights{2} = actor.weights{2} - actor.learning_rate * actor.momentum.w2;
    actor.biases{2} = actor.biases{2} - actor.learning_rate * actor.momentum.b2;
    actor.weights{3} = actor.weights{3} - actor.learning_rate * actor.momentum.w3;
    actor.biases{3} = actor.biases{3} - actor.learning_rate * actor.momentum.b3;
end

function critic = critic_network_init(input_dim, hidden_dim1, hidden_dim2, output_dim, learning_rate)
    critic.weights{1} = randn(hidden_dim1, input_dim) * sqrt(2/input_dim);
    critic.weights{2} = randn(hidden_dim2, hidden_dim1) * sqrt(2/hidden_dim1);
    critic.weights{3} = randn(output_dim, hidden_dim2) * sqrt(2/hidden_dim2);
    
    critic.biases{1} = zeros(hidden_dim1, 1);
    critic.biases{2} = zeros(hidden_dim2, 1);
    critic.biases{3} = zeros(output_dim, 1);
    
    critic.bn_mean{1} = zeros(hidden_dim1, 1);
    critic.bn_var{1} = ones(hidden_dim1, 1);
    critic.bn_beta{1} = zeros(hidden_dim1, 1);
    critic.bn_gamma{1} = ones(hidden_dim1, 1);
    
    critic.bn_mean{2} = zeros(hidden_dim2, 1);
    critic.bn_var{2} = ones(hidden_dim2, 1);
    critic.bn_beta{2} = zeros(hidden_dim2, 1);
    critic.bn_gamma{2} = ones(hidden_dim2, 1);
    
    critic.learning_rate = learning_rate;
    critic.training = true;
end

function q_value = critic_q_value(critic, state, action)
    x = [state; action];
    
    z1 = critic.weights{1} * x + critic.biases{1};
    
    if critic.training
        mu = mean(z1, 2);
        sigma = std(z1, 0, 2) + 1e-6;
        critic.bn_mean{1} = 0.9 * critic.bn_mean{1} + 0.1 * mu;
        critic.bn_var{1} = 0.9 * critic.bn_var{1} + 0.1 * sigma.^2;
        z1_norm = (z1 - mu) ./ sqrt(sigma.^2 + 1e-6);
    else
        z1_norm = (z1 - critic.bn_mean{1}) ./ sqrt(critic.bn_var{1} + 1e-6);
    end
    z1_bn = critic.bn_gamma{1} .* z1_norm + critic.bn_beta{1};
    h1 = tanh(z1_bn);
    
    z2 = critic.weights{2} * h1 + critic.biases{2};
    
    if critic.training
        mu2 = mean(z2, 2);
        sigma2 = std(z2, 0, 2) + 1e-6;
        critic.bn_mean{2} = 0.9 * critic.bn_mean{2} + 0.1 * mu2;
        critic.bn_var{2} = 0.9 * critic.bn_var{2} + 0.1 * sigma2.^2;
        z2_norm = (z2 - mu2) ./ sqrt(sigma2.^2 + 1e-6);
    else
        z2_norm = (z2 - critic.bn_mean{2}) ./ sqrt(critic.bn_var{2} + 1e-6);
    end
    z2_bn = critic.bn_gamma{2} .* z2_norm + critic.bn_beta{2};
    h2 = tanh(z2_bn);
    
    q_value = critic.weights{3} * h2 + critic.biases{3};
end

function grad = critic_q_gradient(critic, state, action)
    x = [state; action];
    
    z1 = critic.weights{1} * x + critic.biases{1};
    if critic.training
        mu = mean(z1, 2);
        sigma = std(z1, 0, 2) + 1e-6;
        z1_norm = (z1 - mu) ./ sqrt(sigma.^2 + 1e-6);
    else
        % 使用存储的方差计算sigma
        sigma = sqrt(critic.bn_var{1} + 1e-6);
        z1_norm = (z1 - critic.bn_mean{1}) ./ sigma;
    end
    z1_bn = critic.bn_gamma{1} .* z1_norm + critic.bn_beta{1};
    h1 = tanh(z1_bn);
    dh1_dz1 = (1 - h1.^2) .* critic.bn_gamma{1} ./ sigma;
    
    z2 = critic.weights{2} * h1 + critic.biases{2};
    if critic.training
        mu2 = mean(z2, 2);
        sigma2 = std(z2, 0, 2) + 1e-6;
        z2_norm = (z2 - mu2) ./ sqrt(sigma2.^2 + 1e-6);
    else
        sigma2 = sqrt(critic.bn_var{2} + 1e-6);
        z2_norm = (z2 - critic.bn_mean{2}) ./ sigma2;
    end
    z2_bn = critic.bn_gamma{2} .* z2_norm + critic.bn_beta{2};
    h2 = tanh(z2_bn);
    dh2_dz2 = (1 - h2.^2) .* critic.bn_gamma{2} ./ sigma2;
    
    d_output = 1;
    dh2 = critic.weights{3}' * d_output;
    dz2 = dh2 .* dh2_dz2;
    dh1 = critic.weights{2}' * dz2;
    dz1 = dh1 .* dh1_dz1;
    
    action_grad_weights = critic.weights{1}(:, end-length(action)+1:end);
    grad = action_grad_weights' * dz1;
end

function reward = calculate_reward(predicted, actual, prev_predicted, prev_actual, is_terminal)
    mse = mean((predicted - actual).^2);
    reward = -mse;
    
    if ~isempty(prev_predicted) && ~isempty(prev_actual)
        pred_velocity = predicted - prev_predicted;
        actual_velocity = actual - prev_actual;
        velocity_mse = mean((pred_velocity - actual_velocity).^2);
        reward = reward - 0.2 * velocity_mse;
        
        actual_direction = sign(actual - prev_actual);
        pred_direction = sign(predicted - prev_predicted);
        direction_reward = sum(actual_direction == pred_direction) / length(actual_direction);
        reward = reward + 0.1 * direction_reward;
    end
    
    if is_terminal
        terminal_weight = 2.0;
        reward = reward * terminal_weight;
    end
    
    reward = reward / (1 + abs(reward));
end

%% 4. DDPG算法实现
actor_lr = 0.0005;
critic_lr = 0.001;
gamma = 0.99;
tau = 0.002;
buffer_size = 50000;
batch_size = 128;
max_episodes = 800;

% 根据实际数据长度动态调整max_steps
data_length = height(data);
max_steps = min(300, data_length - 5); % 确保有足够数据，避免越界

exploration_noise = 0.1;
noise_decay = 0.995;
min_noise = 0.01;

alpha = 0.6;
beta = 0.4;
beta_increment = 0.001;
max_beta = 1.0;

actor = actor_network_init(state_dim, 128, 64, action_dim, actor_lr);
critic = critic_network_init(state_dim + action_dim, 128, 64, 1, critic_lr);

target_actor = actor_network_init(state_dim, 128, 64, action_dim, actor_lr);
target_critic = critic_network_init(state_dim + action_dim, 128, 64, 1, critic_lr);

target_actor.weights = actor.weights;
target_actor.biases = actor.biases;
target_actor.bn_mean = actor.bn_mean;
target_actor.bn_var = actor.bn_var;
target_actor.bn_beta = actor.bn_beta;
target_actor.bn_gamma = actor.bn_gamma;

target_critic.weights = critic.weights;
target_critic.biases = critic.biases;
target_critic.bn_mean = critic.bn_mean;
target_critic.bn_var = critic.bn_var;
target_critic.bn_beta = critic.bn_beta;
target_critic.bn_gamma = critic.bn_gamma;

replay_buffer = replay_buffer_init(buffer_size, batch_size);

episode_rewards = zeros(max_episodes, 1);
episode_mse = zeros(max_episodes, 1);
actor_losses = zeros(max_episodes, 1);
critic_losses = zeros(max_episodes, 1);

% 训练循环
for episode = 1:max_episodes
    % 确保起始索引选择合理，有足够数据
    max_start_idx = height(data) - max_steps - 4;
    if max_start_idx < 4
        max_start_idx = 4; % 防止数据过短时出错
    end
    start_idx = randi([4, max_start_idx]);
    current_data = data(start_idx:start_idx+max_steps-1, :);
    state = get_initial_state(current_data);
    
    episode_reward = 0;
    episode_total_mse = 0;
    step_count = 0;
    prev_predicted = [];
    prev_actual = [];
    
    for step = 1:max_steps
        step_count = step_count + 1;
        
        action = actor_forward(actor, state);
        noise = exploration_noise * randn(size(action));
        action = action + noise;
        action = min(max(action, -1), 1);
        
        [next_state, reward, done, mse] = step_environment(current_data, state, action, prev_predicted, prev_actual, step == max_steps);
        
        replay_buffer = replay_buffer_add(replay_buffer, state, action, reward, next_state, done);
        
        state = next_state;
        episode_reward = episode_reward + reward;
        episode_total_mse = episode_total_mse + mse;
        prev_predicted = action;
        
        % 添加严格的索引边界检查
        if step + 1 <= height(current_data)
            prev_actual = [current_data.joint1(step+1), current_data.joint2(step+1), current_data.joint3(step+1)];
        else
            % 当索引超出范围时，使用最后一个可用值
            prev_actual = [current_data.joint1(end), current_data.joint2(end), current_data.joint3(end)];
        end
        
        if done
            break;
        end
    end
    
    episode_rewards(episode) = episode_reward;
    episode_mse(episode) = episode_total_mse / step_count;
    
    if replay_buffer.count >= 5*batch_size
        [batch, indices, weights] = replay_buffer_sample(replay_buffer, beta);
        
        if ~isempty(batch)
            states = cellfun(@(x) x.state, batch, 'UniformOutput', false);
            actions = cellfun(@(x) x.action, batch, 'UniformOutput', false);
            rewards = cellfun(@(x) x.reward, batch, 'UniformOutput', false);
            next_states = cellfun(@(x) x.next_state, batch, 'UniformOutput', false);
            dones = cellfun(@(x) x.done, batch, 'UniformOutput', false);
            
            [critic, critic_loss, td_errors] = critic_update(critic, states, actions, rewards, next_states, dones, ...
                                                             target_actor, target_critic, gamma, batch_size, weights);
            critic_losses(episode) = critic_loss;
            
            replay_buffer = replay_buffer_update_priorities(replay_buffer, indices, abs(td_errors).^alpha);
            
            critic.training = false;
            [actor, actor_loss] = actor_update(actor, states, actions, critic, batch_size, weights);
            critic.training = true;
            actor_losses(episode) = actor_loss;
            
            polyak_update(target_actor, actor, tau);
            polyak_update(target_critic, critic, tau);
        end
    end
    
    exploration_noise = max(min_noise, exploration_noise * noise_decay);
    beta = min(max_beta, beta + beta_increment);
    
    if mod(episode, 10) == 0
        fprintf('回合: %d, 奖励: %.2f, MSE: %.4f, 探索噪声: %.3f\n', ...
                episode, episode_reward, episode_mse(episode), exploration_noise);
    end
end

%% 训练过程可视化
figure;
subplot(2,2,1);
plot(episode_rewards, 'b', 'LineWidth', 1);
title('Training Rewards Over Episodes');
xlabel('Episode');
ylabel('Total Reward');
grid on;
smoothed_rewards = smoothdata(episode_rewards, 'movmean', 10);
hold on;
plot(smoothed_rewards, 'r', 'LineWidth', 1.5);
legend('Raw Reward', 'Smoothed Reward');

subplot(2,2,2);
plot(episode_mse, 'g', 'LineWidth', 1);
title('Mean Squared Error Over Episodes');
xlabel('Episode');
ylabel('MSE');
grid on;
smoothed_mse = smoothdata(episode_mse, 'movmean', 10);
hold on;
plot(smoothed_mse, 'm', 'LineWidth', 1.5);
legend('Raw MSE', 'Smoothed MSE');

subplot(2,2,3);
plot(actor_losses, 'c', 'LineWidth', 1);
title('Actor Loss Over Episodes');
xlabel('Episode');
ylabel('Loss');
grid on;
smoothed_actor = smoothdata(actor_losses, 'movmean', 10);
hold on;
plot(smoothed_actor, 'k', 'LineWidth', 1.5);
legend('Raw Loss', 'Smoothed Loss');

subplot(2,2,4);
plot(critic_losses, 'y', 'LineWidth', 1);
title('Critic Loss Over Episodes');
xlabel('Episode');
ylabel('Loss');
grid on;
smoothed_critic = smoothdata(critic_losses, 'movmean', 10);
hold on;
plot(smoothed_critic, 'r', 'LineWidth', 1.5);
legend('Raw Loss', 'Smoothed Loss');

sgtitle('Training Metrics');

%% 5. 模型评估
test_ratio = 0.2;
test_start_idx = floor(height(data) * (1 - test_ratio));
test_data = data(test_start_idx:end, :);
test_size = height(test_data);

state = get_initial_state(test_data);
predicted_trajectory = zeros(test_size, action_dim);
actual_trajectory = zeros(test_size, action_dim);
prediction_errors = zeros(test_size, action_dim);

actor.training = false;

for i = 1:test_size
    action = actor_forward(actor, state);
    
    predicted_trajectory(i, :) = action';
    actual_trajectory(i, :) = [test_data.joint1(i), test_data.joint2(i), test_data.joint3(i)];
    prediction_errors(i, :) = abs(predicted_trajectory(i, :) - actual_trajectory(i, :));
    
    state = update_state(state, action, test_data, i);
end

predicted_actual = predicted_trajectory;
actual_actual = actual_trajectory;

for i = 1:3
    joint_name = sprintf('joint%d', i);
    predicted_actual(:, i) = predicted_actual(:, i) * data_info.(joint_name).sigma + data_info.(joint_name).mu;
    actual_actual(:, i) = actual_actual(:, i) * data_info.(joint_name).sigma + data_info.(joint_name).mu;
end

mse = mean(mean((predicted_trajectory - actual_trajectory).^2));
rmse = sqrt(mse);
mae = mean(mean(abs(predicted_trajectory - actual_trajectory)));

actual_mse = mean(mean((predicted_actual - actual_actual).^2));
actual_rmse = sqrt(actual_mse);
actual_mae = mean(mean(abs(predicted_actual - actual_actual)));

fprintf('\n测试结果 (归一化空间):\n');
fprintf('MSE: %.4f\n', mse);
fprintf('RMSE: %.4f\n', rmse);
fprintf('MAE: %.4f\n', mae);

fprintf('\n测试结果 (实际角度空间):\n');
fprintf('MSE: %.4f\n', actual_mse);
fprintf('RMSE: %.4f\n', actual_rmse);
fprintf('MAE: %.4f\n', actual_mae);

figure;
subplot(3,1,1);
plot(predicted_actual(:,1), 'r', 'LineWidth', 1.5);
hold on;
plot(actual_actual(:,1), 'b--', 'LineWidth', 1.5);
title('Joint 1 Angle Prediction vs Actual');
legend('Predicted', 'Actual');
grid on;

subplot(3,1,2);
plot(predicted_actual(:,2), 'r', 'LineWidth', 1.5);
hold on;
plot(actual_actual(:,2), 'b--', 'LineWidth', 1.5);
title('Joint 2 Angle Prediction vs Actual');
legend('Predicted', 'Actual');
grid on;

subplot(3,1,3);
plot(predicted_actual(:,3), 'r', 'LineWidth', 1.5);
hold on;
plot(actual_actual(:,3), 'b--', 'LineWidth', 1.5);
title('Joint 3 Angle Prediction vs Actual');
legend('Predicted', 'Actual');
grid on;
sgtitle('Motion Trajectory Prediction Results (Actual Angle Space)');

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

%% 辅助函数定义
function target = polyak_update(target, source, tau)
    for i = 1:length(target.weights)
        target.weights{i} = tau * source.weights{i} + (1 - tau) * target.weights{i};
        target.biases{i} = tau * source.biases{i} + (1 - tau) * target.biases{i};
    end
    
    if isfield(target, 'bn_mean')
        for i = 1:length(target.bn_mean)
            target.bn_mean{i} = tau * source.bn_mean{i} + (1 - tau) * target.bn_mean{i};
            target.bn_var{i} = tau * source.bn_var{i} + (1 - tau) * target.bn_var{i};
            target.bn_beta{i} = tau * source.bn_beta{i} + (1 - tau) * target.bn_beta{i};
            target.bn_gamma{i} = tau * source.bn_gamma{i} + (1 - tau) * target.bn_gamma{i};
        end
    end
end

function [critic, loss, td_errors] = critic_update(critic, states, actions, rewards, next_states, dones, target_actor, target_critic, gamma, batch_size, weights)
    loss = 0;
    td_errors = zeros(batch_size, 1);
    if nargin < 9
        weights = ones(1, batch_size);
    end
    
    grad_weights1 = zeros(size(critic.weights{1}));
    grad_biases1 = zeros(size(critic.biases{1}));
    grad_weights2 = zeros(size(critic.weights{2}));
    grad_biases2 = zeros(size(critic.biases{2}));
    grad_weights3 = zeros(size(critic.weights{3}));
    grad_biases3 = zeros(size(critic.biases{3}));
    
    for i = 1:batch_size
        state = states{i};
        action = actions{i};
        reward = rewards{i};
        next_state = next_states{i};
        done = dones{i};
        
        next_action = actor_forward(target_actor, next_state);
        target_q = critic_q_value(target_critic, next_state, next_action);
        y = reward + gamma * (1 - done) * target_q;
        
        current_q = critic_q_value(critic, state, action);
        
        td_error = y - current_q;
        td_errors(i) = td_error;
        
        weighted_error = weights(i) * td_error;
        
        x = [state; action];
        
        z1 = critic.weights{1} * x + critic.biases{1};
        if critic.training
            mu = mean(z1, 2);
            sigma = std(z1, 0, 2) + 1e-6;
            z1_norm = (z1 - mu) ./ sqrt(sigma.^2 + 1e-6);
        else
            sigma = sqrt(critic.bn_var{1} + 1e-6);
            z1_norm = (z1 - critic.bn_mean{1}) ./ sigma;
        end
        z1_bn = critic.bn_gamma{1} .* z1_norm + critic.bn_beta{1};
        h1 = tanh(z1_bn);
        dh1_dz1 = (1 - h1.^2) .* critic.bn_gamma{1} ./ sigma;
        
        z2 = critic.weights{2} * h1 + critic.biases{2};
        if critic.training
            mu2 = mean(z2, 2);
            sigma2 = std(z2, 0, 2) + 1e-6;
            z2_norm = (z2 - mu2) ./ sqrt(sigma2.^2 + 1e-6);
        else
            sigma2 = sqrt(critic.bn_var{2} + 1e-6);
            z2_norm = (z2 - critic.bn_mean{2}) ./ sigma2;
        end
        z2_bn = critic.bn_gamma{2} .* z2_norm + critic.bn_beta{2};
        h2 = tanh(z2_bn);
        dh2_dz2 = (1 - h2.^2) .* critic.bn_gamma{2} ./ sigma2;
        
        d_output = -weighted_error;
        
        d_weights3 = d_output * h2';
        d_biases3 = d_output;
        
        d_h2 = critic.weights{3}' * d_output;
        d_z2 = d_h2 .* dh2_dz2;
        
        d_weights2 = d_z2 * h1';
        d_biases2 = d_z2;
        
        d_h1 = critic.weights{2}' * d_z2;
        d_z1 = d_h1 .* dh1_dz1;
        
        d_weights1 = d_z1 * x';
        d_biases1 = d_z1;
        
        grad_weights1 = grad_weights1 + d_weights1;
        grad_biases1 = grad_biases1 + d_biases1;
        grad_weights2 = grad_weights2 + d_weights2;
        grad_biases2 = grad_biases2 + d_biases2;
        grad_weights3 = grad_weights3 + d_weights3;
        grad_biases3 = grad_biases3 + d_biases3;
        
        loss = loss + 0.5 * weights(i) * (td_error^2);
    end
    
    grad_weights1 = grad_weights1 / batch_size;
    grad_biases1 = grad_biases1 / batch_size;
    grad_weights2 = grad_weights2 / batch_size;
    grad_biases2 = grad_biases2 / batch_size;
    grad_weights3 = grad_weights3 / batch_size;
    grad_biases3 = grad_biases3 / batch_size;
    
    loss = loss / batch_size;
    
    l2_lambda = 1e-4;
    loss = loss + l2_lambda * (sum(grad_weights1(:).^2) + sum(grad_weights2(:).^2) + sum(grad_weights3(:).^2));
    
    if ~isfield(critic, 'momentum')
        critic.momentum.w1 = 0;
        critic.momentum.b1 = 0;
        critic.momentum.w2 = 0;
        critic.momentum.b2 = 0;
        critic.momentum.w3 = 0;
        critic.momentum.b3 = 0;
        critic.momentum.beta = 0.9;
    end
    
    critic.momentum.w1 = critic.momentum.beta * critic.momentum.w1 + (1 - critic.momentum.beta) * grad_weights1;
    critic.momentum.b1 = critic.momentum.beta * critic.momentum.b1 + (1 - critic.momentum.beta) * grad_biases1;
    critic.momentum.w2 = critic.momentum.beta * critic.momentum.w2 + (1 - critic.momentum.beta) * grad_weights2;
    critic.momentum.b2 = critic.momentum.beta * critic.momentum.b2 + (1 - critic.momentum.beta) * grad_biases2;
    critic.momentum.w3 = critic.momentum.beta * critic.momentum.w3 + (1 - critic.momentum.beta) * grad_weights3;
    critic.momentum.b3 = critic.momentum.beta * critic.momentum.b3 + (1 - critic.momentum.beta) * grad_biases3;
    
    critic.weights{1} = critic.weights{1} - critic.learning_rate * critic.momentum.w1;
    critic.biases{1} = critic.biases{1} - critic.learning_rate * critic.momentum.b1;
    critic.weights{2} = critic.weights{2} - critic.learning_rate * critic.momentum.w2;
    critic.biases{2} = critic.biases{2} - critic.learning_rate * critic.momentum.b2;
    critic.weights{3} = critic.weights{3} - critic.learning_rate * critic.momentum.w3;
    critic.biases{3} = critic.biases{3} - critic.learning_rate * critic.momentum.b3;
end

function state = get_initial_state(data)
    state = zeros(3*3*4, 1);
    
    for t = 0:3
        idx = min(t+1, height(data));
        
        state(t*9 + 1) = data.joint1(idx);
        state(t*9 + 2) = data.joint2(idx);
        state(t*9 + 3) = data.joint3(idx);
        
        state(t*9 + 4) = data.velocity1(idx);
        state(t*9 + 5) = data.velocity2(idx);
        state(t*9 + 6) = data.velocity3(idx);
        
        state(t*9 + 7) = data.acceleration1(idx);
        state(t*9 + 8) = data.acceleration2(idx);
        state(t*9 + 9) = data.acceleration3(idx);
    end
end

function [next_state, reward, done, mse] = step_environment(data, state, action, prev_predicted, prev_actual, is_terminal)
    current_joint1 = state(1);
    current_idx = find(abs(data.joint1 - current_joint1) < 1e-3, 1);
    
    if isempty(current_idx) || current_idx >= height(data)
        current_idx = randi([1, height(data)-1]);
    end
    
    predicted_joints = action'; 
    
    % 确保next_idx不超出数据范围
    next_idx = min(current_idx + 1, height(data));
    actual_joints = [data.joint1(next_idx), data.joint2(next_idx), data.joint3(next_idx)];
    
    mse = mean((predicted_joints - actual_joints).^2);
    
    reward = calculate_reward(predicted_joints, actual_joints, prev_predicted, prev_actual, is_terminal);
    
    next_state = zeros(size(state));
    next_state(1:27) = state(10:36);
    
    next_state(28) = predicted_joints(1);
    next_state(29) = predicted_joints(2);
    next_state(30) = predicted_joints(3);
    
    current_angles = state(1:3);
    next_state(31) = (predicted_joints(1) - current_angles(1)) / 1;
    next_state(32) = (predicted_joints(2) - current_angles(2)) / 1;
    next_state(33) = (predicted_joints(3) - current_angles(3)) / 1;
    
    current_velocities = state(4:6);
    next_state(34) = (next_state(31) - current_velocities(1)) / 1;
    next_state(35) = (next_state(32) - current_velocities(2)) / 1;
    next_state(36) = (next_state(33) - current_velocities(3)) / 1;
    
    done = (current_idx + 1) >= height(data) || is_terminal;
end

function new_state = update_state(old_state, action, data, current_idx)
    new_state = zeros(size(old_state));
    
    new_state(1:27) = old_state(10:36);
    
    new_state(28) = action(1);
    new_state(29) = action(2);
    new_state(30) = action(3);
    
    current_angles = old_state(1:3);
    new_state(31) = (action(1) - current_angles(1)) / 1;
    new_state(32) = (action(2) - current_angles(2)) / 1;
    new_state(33) = (action(3) - current_angles(3)) / 1;
    
    current_velocities = old_state(4:6);
    new_state(34) = (new_state(31) - current_velocities(1)) / 1;
    new_state(35) = (new_state(32) - current_velocities(2)) / 1;
    new_state(36) = (new_state(33) - current_velocities(3)) / 1;
end
