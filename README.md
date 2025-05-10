
# Monte Carlo Reinforcement Learning Agent in Gridworld

This project implements a Monte Carlo reinforcement learning (MC-RL) agent in a 5x5 Gridworld environment using Python and Tkinter. It visually demonstrates the learning process of the agent as it navigates toward a goal while avoiding traps, learning through interaction over episodes.

## 1. Overview

- The agent operates in a 5x5 grid.
- It learns by interacting with the environment and updating state values based on the returns it receives at the end of each episode.
- It uses a first-visit Monte Carlo method for learning.
- State values are visualized using heatmaps and line plots to show reward progress.

## 2. Files and Structure

### environment.py
- Creates a visual Tkinter grid of size 5x5.
- Initializes:
  - Agent (rectangle image)
  - Two traps (triangle images)
  - One goal (circle image)
- The agent moves based on actions: up, down, left, right.
- The environment returns:
  - A new state
  - A reward
  - A terminal flag (`done`)

### mc_agent.py
- Implements the Monte Carlo learning agent.
- Stores episode samples (`state`, `reward`, `done`).
- Uses epsilon-greedy policy for action selection:
  - Explores with probability ε
  - Exploits with probability 1-ε
- Updates state values using the formula:
  ```
  V(s) = V(s) + α * (G - V(s))
  ```
  where `G` is the return and `α` is the learning rate.
- Visualizes:
  - State value heatmap
  - Reward per episode line plot

## 3. Environment Details

- Grid: 5 rows x 5 columns
- Start Position: `[0, 0]`
- Goal: `[3, 2]`, reward = +100
- Traps:
  - `[2, 1]` and `[1, 2]`, reward = -100
- All other moves: reward = 0
- Agent can move in 4 directions: up, down, left, right
- Boundary checks prevent out-of-grid movement

## 4. Learning and Execution Details

- **Discount factor (γ):** 0.95
- **Learning rate (α):** 0.05
- **Epsilon (ε):** Starts at 0.2 and decays over time
- **Exploration decay:** Epsilon decays with each episode until it reaches a minimum of 0.01
- **Success count:** Incremented when agent reaches goal
- **Failure count:** Incremented when agent hits a trap

## 5. Visual Output

### Heatmap
- Shows learned state values.
- Color gradient (green = high value = preferred path).
- Helps understand where the agent prefers to move.

### Line Chart
- Shows total reward per episode.
- Gives an overview of how well the agent is learning over time.

## 6. How to Run

### Requirements
Install required Python packages:
```bash
pip install numpy matplotlib seaborn pillow
```

### Folder Structure
Ensure the following directory structure:
```
project/
│
├── mc_agent.py
├── environment.py
├── img/
│   ├── rectangle.png   # for agent
│   ├── triangle.png    # for traps
│   └── circle.png      # for goal
```

### Execution
Run the main script:
```bash
python mc_agent.py
```

## 7. Notes

- Uses **first-visit Monte Carlo control** for value updates.
- Values are stored in a `defaultdict`.
- Agent starts from `[0,0]` in every episode.
- Environment refreshes via `tkinter.update()` for visualization.

## 8. Example Outputs

- **Heatmap**: Green states are higher value; red are less preferred.
- **Episode Reward Graph**: Shows increasing or fluctuating reward as agent learns.

## 9. Keywords

Monte Carlo, Reinforcement Learning, Epsilon-Greedy, Gridworld, State Value Function, Exploration, Heatmap, Python, Tkinter, First-Visit MC, RL Visualization
