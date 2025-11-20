import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Fix seeds
np.random.seed(42)
random.seed(42)

# ============================================================
# ENVIRONMENT
# ============================================================

class InventoryEnv:
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

        self.demand_probs = [0.1, 0.2, 0.4, 0.2, 0.1]
        self.demand_values = [0, 1, 2, 3, 4]

    def reset(self):
        self.inventory = 5
        self.day = 0
        return self.inventory

    def step(self, action):
        order_qty = action
        order_cost = self.order_cost if order_qty > 0 else 0
        self.inventory = min(self.inventory + order_qty, self.max_inventory)

        demand = np.random.choice(self.demand_values, p=self.demand_probs)
        sales = min(self.inventory, demand)
        lost_sales = max(0, demand - self.inventory)
        self.inventory -= sales

        reward = (
            sales * self.sale_profit
            - lost_sales * self.stockout_penalty
            - self.inventory * self.holding_cost
            - order_cost
        )

        self.day += 1
        done = self.day >= self.episode_length
        return self.inventory, reward, done, {}

# ============================================================
# AGENT
# ============================================================

class QLearningAgent:
    def __init__(
        self,
        n_states=11,
        n_actions=3,
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
# TRAINING (UPDATED WITH STREAMLIT LOGGING)
# ============================================================

def train(num_episodes, alpha, gamma, epsilon_start, epsilon_end, epsilon_decay, log_placeholder=None):
    env = InventoryEnv()
    agent = QLearningAgent(
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay
    )

    rewards = []

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state, ep)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        rewards.append(total_reward)

        # LIVE LOGGING
        if log_placeholder and ep % 50 == 0:
            avg50 = np.mean(rewards[-50:])
            log_placeholder.text(
                f"Episode {ep}/{num_episodes} - avg reward (last 50): {avg50:.2f}"
            )

    return agent, rewards

# ============================================================
# BASELINE
# ============================================================

def baseline_policy(inv):
    return 2 if inv < 3 else 0

def run_baseline(episodes=200):
    env = InventoryEnv()
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        total = 0
        done = False
        while not done:
            action = baseline_policy(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total += reward
        rewards.append(total)
    return rewards

# ============================================================
# PLOTS
# ============================================================

def line_plot(data, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(data)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    return fig

def moving_avg_plot(rewards, window=50):
    avg = np.convolve(rewards, np.ones(window)/window, mode="valid")
    return line_plot(avg, f"Moving Avg Reward (window={window})", "Episode", "Reward")

def simulate_trajectory(agent):
    env = InventoryEnv()
    demand_seq = np.random.choice(env.demand_values, size=60, p=env.demand_probs)

    def run(policy_fn):
        env = InventoryEnv()
        inv = []
        state = env.reset()
        for d in demand_seq:
            action = policy_fn(state)
            env.inventory = min(state + action, env.max_inventory)
            sales = min(env.inventory, d)
            env.inventory -= sales
            inv.append(env.inventory)
            state = env.inventory
        return inv

    baseline_inv = run(baseline_policy)
    rl_inv = run(lambda s: np.argmax(agent.Q[s]))

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(baseline_inv, label="Baseline", linewidth=2)
    ax.plot(rl_inv, label="RL", linewidth=2)
    ax.set_title("Inventory Trajectory Comparison")
    ax.set_xlabel("Day")
    ax.set_ylabel("Inventory")
    ax.legend()
    ax.grid()
    return fig

# ============================================================
# STREAMLIT UI
# ============================================================

st.title("Health Facility Inventory Optimization with Reinforcement Learning")
st.write("Tune RL hyperparameters and track the agent's training progress.")

st.sidebar.header("🔧 RL Hyperparameters")

num_episodes = st.sidebar.slider("Training Episodes", 100, 3000, 800)
alpha = st.sidebar.slider("Learning Rate (alpha)", 0.01, 0.50, 0.05)
gamma = st.sidebar.slider("Discount Factor (gamma)", 0.50, 0.999, 0.99)
epsilon_start = st.sidebar.slider("Epsilon Start", 0.1, 1.0, 1.0)
epsilon_end = st.sidebar.slider("Epsilon End", 0.001, 0.2, 0.01)
epsilon_decay = st.sidebar.slider("Epsilon Decay", 50, 2000, 300)

if st.button("🚀 Train RL Agent"):
    st.write("Training in progress...")

    # Live log section
    log_placeholder = st.empty()

    agent, rewards = train(
        num_episodes,
        alpha,
        gamma,
        epsilon_start,
        epsilon_end,
        epsilon_decay,
        log_placeholder=log_placeholder
    )

    baseline_rewards = run_baseline()

    st.success("Training Complete!")

    st.subheader("Performance Summary")
    st.write(f"Baseline Avg Reward: **{np.mean(baseline_rewards):.2f}**")
    st.write(f"RL Avg Reward (last 200 episodes): **{np.mean(rewards[-200:]):.2f}**")

    st.pyplot(line_plot(rewards, "RL Learning Curve", "Episode", "Reward"))
    st.pyplot(moving_avg_plot(rewards))
    #st.pyplot(simulate_trajectory(agent))

    best_actions = {inv: int(np.argmax(agent.Q[inv])) for inv in range(11)}

    st.subheader("Inventory → Best Action (order quantity)")
    #st.json(best_actions)
    st.table(
        {"Inventory Level": list(best_actions.keys()),
        "Best Action": list(best_actions.values())}
    )