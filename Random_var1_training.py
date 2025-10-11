import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
from test_script import QNetwork

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)  
        self.memory = deque(maxlen=100000) 
        self.batch_size = 64  
        self.gamma = 0.99  
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995 
        self.target_update_freq = 10  

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
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

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

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


def train_dqn(episodes=200): 
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    
    for _ in range(2000): 
        state = env.reset()[0]
        env.unwrapped.length = random.uniform(0.4, 1.8)
        done = False
        steps = 0
        while not done and steps < 500:
            action = env.action_space.sample()
            next_state, reward, done, _, __ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            steps += 1
    
    rewards_per_episode = []

    for episode in range(episodes):
        env.unwrapped.length = random.uniform(0.4, 1.8)
        state = env.reset()[0]
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, __ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            
           
            for _ in range(4):
                agent.optimize()
            
            state = next_state
            total_reward += reward

        if episode % 10 == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
        
        if episode % 20 == 0:
            avg = np.mean(rewards_per_episode[-20:])
            print(f"Ep {episode}/{episodes} | Avg: {avg:.1f} | ε: {agent.epsilon:.3f}")
    
    os.makedirs("weights", exist_ok=True)
    torch.save(agent.policy_net.state_dict(), "weights/dqn_cartpole_test.pth")
    return rewards_per_episode


if __name__ == "__main__":
    train_dqn(episodes=200)
