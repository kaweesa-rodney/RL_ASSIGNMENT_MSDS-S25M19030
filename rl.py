import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import csv
import os
import matplotlib.pyplot as plt

import random
np.random.seed(42)
random.seed(42)

# ============================================================
# 1. Inventory Environment (SME Inventory Optimization)
# ============================================================

class InventoryEnv:
    """
    Simple SME inventory environment.
    State = current inventory (0–10)
    Actions = order 0, 1, or 2 units
    """

    def __init__(
        self,
        max_inventory=10,
        holding_cost=2,
        stockout_penalty=35,
        sale_profit=10,
        order_cost=5,
        episode_length=60
    ):
        self.max_inventory = max_inventory
        self.holding_cost = holding_cost
        self.stockout_penalty = stockout_penalty
        self.sale_profit = sale_profit
        self.order_cost = order_cost
        self.episode_length = episode_length

        self.inventory = None
        self.day = None

        # Demand distribution: realistic for a small shop
        self.demand_probs = [0.1, 0.2, 0.4, 0.2, 0.1]
        self.demand_values = [0, 1, 2, 3, 4]

    def reset(self):
        self.inventory = 5  # start with moderate stock
        self.day = 0
        return self.inventory

    def step(self, action):
        # --- Action: place order ---
        order_qty = action   # 0, 1, 2
        order_cost = self.order_cost if order_qty > 0 else 0

        # Add ordered stock
        self.inventory = min(self.inventory + order_qty, self.max_inventory)

        # --- Demand happens ---
        demand = np.random.choice(self.demand_values, p=self.demand_probs)

        # Sales = min(inventory, demand)
        sales = min(self.inventory, demand)

        # Stockouts happen if demand > inventory
        lost_sales = max(0, demand - self.inventory)

        # Remove sold items
        self.inventory -= sales

        # --- Reward ---
        reward = 0
        reward += sales * self.sale_profit
        reward -= lost_sales * self.stockout_penalty
        reward -= self.inventory * self.holding_cost
        reward -= order_cost

        # --- Episode control ---
        self.day += 1
        done = self.day >= self.episode_length

        return self.inventory, reward, done, {
            "demand": demand,
            "sales": sales,
            "lost_sales": lost_sales,
            "order_qty": order_qty
        }


# ============================================================
# 2. Q-Learning Agent
# ============================================================

class QLearningAgent:
    def __init__(
        self,
        n_states=11,      # inventory 0–10
        n_actions=3,      # order 0,1,2
        alpha=0.05,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=300
    ):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def epsilon(self, episode):
        frac = min(episode / self.epsilon_decay, 1.0)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state, episode):
        if np.random.rand() < self.epsilon(episode):
            return np.random.randint(3)
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        best_next = 0 if done else np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error


# ============================================================
# 3. Training Loop
# ============================================================

def train(num_episodes=800):
    env = InventoryEnv()
    agent = QLearningAgent()

    all_rewards = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action = agent.select_action(state, ep)
            next_state, reward, done, info = env.step(action)

            agent.update(state, action, reward, next_state, done)

            state = next_state
            ep_reward += reward

        all_rewards.append(ep_reward)

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{num_episodes} - avg reward (last 50): {np.mean(all_rewards[-50:]):.2f}")

    return agent, all_rewards


# ============================================================
# 4. Baseline Heuristic (Threshold Rule)
# ============================================================

def baseline_policy(inventory):
    if inventory < 3:
        return 2  # order 2
    return 0      # otherwise order nothing


def run_baseline(episodes=200):
    env = InventoryEnv()

    rewards = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action = baseline_policy(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            ep_reward += reward

        rewards.append(ep_reward)

    return rewards


# ============================================================
# 5. Comparison
# ============================================================

def compare(agent_rewards, baseline_rewards):
    print("\n================ RL vs BASELINE ================")
    print(f"Baseline avg reward: {np.mean(baseline_rewards):.2f}")
    print(f"RL avg reward:       {np.mean(agent_rewards[-200:]):.2f}")
    print("================================================\n")


def print_learned_policy(agent):
    print("Inventory | Best action (order qty)")
    print("-------------------------------")
    for inv in range(0, 11):  # states 0..10
        best_action = np.argmax(agent.Q[inv])
        print(f"{inv:9d} | {best_action}")



# ============================================================
# 6. PLOTS (added here)
# ============================================================

def plot_learning_curve(rewards):
    plt.figure(figsize=(10,5))
    plt.plot(rewards)
    plt.title("RL Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()

def plot_moving_average(rewards, window=50):
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(10,5))
    plt.plot(moving_avg)
    plt.title(f"Moving Average Reward (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.grid(True)
    plt.show()

def plot_baseline_vs_rl(baseline_rewards, rl_rewards):
    plt.figure(figsize=(10,5))
    plt.boxplot([baseline_rewards, rl_rewards[-200:]], labels=["Baseline", "RL (last 200)"])
    plt.title("Baseline vs RL Performance")
    plt.ylabel("Episode Reward")
    plt.grid(True)
    plt.show()


import seaborn as sns

def plot_q_table(agent):
    plt.figure(figsize=(8, 6))
    sns.heatmap(agent.Q, annot=True, cmap="viridis", fmt=".1f")
    plt.title("Q-Table Heatmap\nRows = Inventory Level (0–10), Columns = Actions (0,1,2)")
    plt.xlabel("Action (Order Qty)")
    plt.ylabel("Inventory Level")
    plt.show()


def simulate_trajectory(agent, episode_length=60):
    env = InventoryEnv()
    
    # Create identical random demand sequence
    demand_seq = np.random.choice(env.demand_values, size=episode_length, p=env.demand_probs)
    
    # ----------- BASELINE TRAJECTORY -----------
    env = InventoryEnv()
    inventory_baseline = []
    state = env.reset()

    for d in demand_seq:
        action = baseline_policy(state)
        env.inventory = min(state + action, env.max_inventory)
        
        sales = min(env.inventory, d)
        env.inventory -= sales

        inventory_baseline.append(env.inventory)
        state = env.inventory

    # ----------- RL TRAJECTORY -----------
    env = InventoryEnv()
    inventory_rl = []
    state = env.reset()

    for d in demand_seq:
        action = np.argmax(agent.Q[state])  # greedy RL policy
        env.inventory = min(state + action, env.max_inventory)

        sales = min(env.inventory, d)
        env.inventory -= sales

        inventory_rl.append(env.inventory)
        state = env.inventory

    # ----------- PLOT BOTH TRAJECTORIES -----------
    plt.figure(figsize=(10, 5))
    plt.plot(inventory_baseline, label="Baseline", linewidth=2)
    plt.plot(inventory_rl, label="RL", linewidth=2)
    plt.xlabel("Day")
    plt.ylabel("Inventory Level")
    plt.title("Inventory Trajectory: Baseline vs RL")
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    agent, rewards = train()
    baseline = run_baseline()
    compare(rewards, baseline)
    print_learned_policy(agent)

    # ---- SHOW PLOTS ----
    plot_learning_curve(rewards)
    plot_moving_average(rewards)
    plot_baseline_vs_rl(baseline, rewards)

    #plot_q_table(agent)
    #simulate_trajectory(agent)