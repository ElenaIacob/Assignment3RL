import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import pandas as pd
import matplotlib.pyplot as plt
from test_script import QNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0005)
        self.memory = deque(maxlen=50000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update_freq = 500
        self.steps_done = 0

    def select_action(self, state):
        self.steps_done += 1
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)
        q_values = self.policy_net(states).gather(1, actions).squeeze()
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def shaped_reward(state, original_reward):
    x, x_dot, theta, theta_dot = state
    angle_reward = (np.pi / 2 - abs(theta)) / (np.pi / 2)
    position_reward = (2.4 - abs(x)) / 2.4
    velocity_penalty = -0.1 * abs(x_dot) - 0.1 * abs(theta_dot)
    return original_reward + 3.0 * angle_reward + 1.0 * position_reward + velocity_penalty

def train_dqn(episodes=300):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    rewards_per_episode = []
    total_steps = 0
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    for episode in range(episodes):
        env.unwrapped.length = random.uniform(0.4, 1.8)
        state = env.reset()[0]
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, __ = env.step(action)
            reward = shaped_reward(next_state, reward)
            agent.store_transition(state, action, reward, next_state, done)
            if total_steps % 4 == 0:
                agent.optimize()
            state = next_state
            total_reward += reward
            total_steps += 1
            if total_steps % agent.target_update_freq == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
        if episode % 20 == 0:
            avg = np.mean(rewards_per_episode[-20:])
            print(f"Episode {episode}/{episodes} | Avg reward: {avg:.1f} | ε: {agent.epsilon:.3f}")
    model_path = os.path.join(results_dir, "Random_reshaped_reward_training.pth")
    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"Training complete. Weights saved to {model_path}")
    df = pd.DataFrame({"Episode": np.arange(len(rewards_per_episode)), "TotalReward": rewards_per_episode})
    excel_path = os.path.join(results_dir, "Random_reshaped_reward.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Saved training rewards to {excel_path}")
    plt.figure(figsize=(8, 5))
    plt.plot(rewards_per_episode, label="Total Reward per Episode", alpha=0.8)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward Shaping DQN Training Performance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "Random_reshaped_reward_plot.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved training plot as {plot_path}")
    return rewards_per_episode

if __name__ == "__main__":
    train_dqn(episodes=300)