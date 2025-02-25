import gymnasium as gym
import numpy as np
import os
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
#from moviepy.editor import *

env = gym.make('MountainCar-v0', render_mode='rgb_array') # Create the environment.
video_folder="Videos"
successful_episode = None

# state, _ = env.reset() # Reset the environment and return the initial state and info dictionary.

BEST_MODEL_FILE = "best_q_table.npy"
best_score = float('-inf')

'''
print(env.observation_space.high)
print(env.observation_space.low)
[0.6  0.07] The first value 0.6 indicates the position of the flag, and the second value 0.07 indicates the velocity of the car.
[-1.2  -0.07] The first value -1.2 indicates the position of the car, and the second value -0.07 indicates the velocity of the car.
print(env.action_space.n) # Print the number of actions available in the environment.
3
Action 0: Push the car to the left.
Action 1: Do nothing (no action).
Action 2: Push the car to the right.
'''
LEARNING_RATE = 0.2    # Define the learning rate.                                                                                                                                 
DISCOUNT = 0.95 # Define the discount factor.                               
EPISODES = 25000 # Define the number of episodes to train the agent.
SHOW_EVERY = 3000 # Define the number of episodes to display the environment.

DISCRETE_OS_SIZE = [20, 20] # Define the number of buckets for each state variable.
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE # Define the size of each bucket.

# Exploration settings
epsilon = 1.0  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1 # starting epsilon value
END_EPSILON_DECAYING = EPISODES // 4 # ending epsilon value
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING) # decay value

# Q-table file path
#Q_TABLE_FILE = "q_table.npy"

# Load existing Q-table if available, else initialize randomly
if os.path.exists(BEST_MODEL_FILE):
    q_table = np.load(BEST_MODEL_FILE)
    print("Loaded existing Q-table.")
else:
    q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    # convert continuous state to discrete state
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int)) # we use this tuple to look up the 3 Q values for the available actions in the q-table

for episode in range(EPISODES):
    total_reward = 0
    state, info = env.reset()
    discrete_state = get_discrete_state(state)
    done = False
    frames = []

    render = episode % SHOW_EVERY == 0
    if render:
        print(f"Episode: {episode}")

    while not done:
        action = np.argmax(q_table[discrete_state]) if np.random.random() > epsilon else np.random.randint(0, env.action_space.n)
        new_state, reward, terminated, truncated, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        done = terminated or truncated

        frame = env.render()
        frames.append(frame)

        # Reward shaping to encourage movement
        if new_state[0] >= env.unwrapped.goal_position:
            reward = 50  # Reward for reaching the goal
            successful_episode = episode
            break
        else:
            reward += abs(new_state[1]) * 50  # Encourage velocity
        
        total_reward += reward

        # print(reward, new_state)
        # new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        # IF SIMULATION DID NOT END YET AFTER LAST STEP, UPDATE Q VALUE
        
        if not done:
            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])
            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]
            # New Q value
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q

        # Simulation ended (for any reason) - if goal position is achieved - update Q value with reward directly
        elif new_state[0] >= env.unwrapped.goal_position:
            q_table[discrete_state + (action,)] = 0
        
        discrete_state = new_discrete_state

        if render:
            env.render()
        
    if successful_episode:
        break

    # decay epsilon to reduce exploration over time
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon = max(0, epsilon - epsilon_decay_value)

    # Save best Q-table
    if total_reward > best_score:
        best_score = total_reward
        np.save(BEST_MODEL_FILE, q_table)
        print(f"New best model saved at Episode {episode}, Score: {best_score}")
# close the environment
env.close()

# Save video of the successful episode
if successful_episode:
    clip = ImageSequenceClip(frames, fps=30)
    clip.write_videofile(f"{video_folder}/success.mp4", codec="libx264")
    print("Saved successful episode as success.mp4")


env = gym.make('MountainCar-v0', render_mode='human')  # Use 'human' to see the agent

state, info = env.reset()
discrete_state = get_discrete_state(state)
done = False

while not done:
    action = np.argmax(q_table[discrete_state])  # Always pick the best action
    new_state, reward, terminated, truncated, _ = env.step(action)
    discrete_state = get_discrete_state(new_state)
    done = terminated or truncated
    env.render()  # Render in real-time


env.close()