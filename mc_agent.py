import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from environment import Env

#Here we initialize everything the agent needs to learn and interact with the environment.
class MCAgent:
    def __init__(self, actions):
        self.width = 5
        self.height = 5
        self.actions = actions
        self.learning_rate = 0.05  # Increased for faster learning
        self.discount_factor = 0.95  # More future-focused
        self.epsilon = 0.2  # More exploration initially
        self.epsilon_decay = 0.99
        self.min_epsilon = 0.01
        self.samples = []
        self.value_table = defaultdict(float)
        self.episode_rewards = []
        self.success_count = 0
        self.failure_count = 0

#Save in sample list
    def save_sample(self, state, reward, done):
        self.samples.append([state, reward, done])

#Update after eps
    def update(self):
        G_t = 0
        visit_state = []
        total_reward = 0
        for reward in reversed(self.samples):
            state = str(reward[0])
            total_reward += reward[1]
            if state not in visit_state:
                visit_state.append(state)
                G_t = self.discount_factor * (reward[1] + G_t)
                value = self.value_table[state]
                self.value_table[state] = (value +
                                           self.learning_rate * (G_t - value))
        self.episode_rewards.append(total_reward) #store total reward
        self.samples.clear() #clear mem

        # Decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

#random action
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(self.actions))
        next_state = self.possible_next_state(state)
        return int(self.arg_max(next_state)) #Best action

#Tie-Max
    @staticmethod
    def arg_max(next_state):
        max_index_list = []
        max_value = next_state[0]
        for index, value in enumerate(next_state):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

#Simulate next state values - 4
    def possible_next_state(self, state):
        col, row = state
        next_state = [0.0] * 4
        if row != 0:
            next_state[0] = self.value_table[str([col, row - 1])]
        else:
            next_state[0] = self.value_table[str(state)]
        if row != self.height - 1:
            next_state[1] = self.value_table[str([col, row + 1])]
        else:
            next_state[1] = self.value_table[str(state)]
        if col != 0:
            next_state[2] = self.value_table[str([col - 1, row])]
        else:
            next_state[2] = self.value_table[str(state)]
        if col != self.width - 1:
            next_state[3] = self.value_table[str([col + 1, row])]
        else:
            next_state[3] = self.value_table[str(state)]
        return next_state

#Visualise heat map
    def render_value_heatmap(self, goal=None, trap=None):
        heatmap = np.zeros((self.height, self.width))
        for row in range(self.height):
            for col in range(self.width):
                heatmap[row, col] = self.value_table[str([col, row])]

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            heatmap,
            annot=True,
            cmap="RdYlGn",
            linewidths=0.5,
            linecolor='black',
            fmt=".2f"
        )
        plt.title("State Value Heatmap (Green = Better)", fontsize=20)
        plt.xlabel("Column", fontsize=12)
        plt.ylabel("Row", fontsize=12)

        if goal:
            plt.text(goal[0] + 0.5, goal[1] + 0.5, '', ha='center', va='center', fontsize=20)
        if trap:
            plt.text(trap[0] + 0.5, trap[1] + 0.5, '', ha='center', va='center', fontsize=20)

        plt.show()

#Line-chart
    def render_episode_rewards(self):
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards, color='cyan', linewidth=2)
        plt.title("Total Reward per Episode", fontsize=20, color='white')
        plt.xlabel("Episode", fontsize=20, color='white')
        plt.ylabel("Total Reward", fontsize=20, color='white')
        plt.xticks(color='white', fontsize=20)
        plt.yticks(color='white', fontsize=20)
        plt.grid(True, linestyle='--', alpha=1)
        plt.tight_layout()
        plt.show()


#Start env-agent
if __name__ == "__main__":
    env = Env()
    agent = MCAgent(actions=list(range(env.n_actions)))

    for episode in range(1000):
        state = env.reset()
        action = agent.get_action(state)
#Save evtg untl eps ends.
        while True:
            next_state, reward, done = env.step(action)
            agent.save_sample(next_state, reward, done)
            action = agent.get_action(next_state)
#chck s/f
            if done:
                print(f"Episode {episode + 1} finished. Reward: {reward}")
                if reward > 0:
                    agent.success_count += 1
                else:
                    agent.failure_count += 1
                agent.update()
                break

    print(f"\nTotal Success: {agent.success_count}, Failures: {agent.failure_count}")
    agent.render_episode_rewards()
    agent.render_value_heatmap(goal=[3, 2], trap=[2, 2])
