# 🚗 MountainCar-v0 Q-Learning Agent

## 📝 Project Overview

This project implements Q-learning to train an agent to solve the MountainCar-v0 environment from OpenAI Gym. The agent learns to reach the flag by optimizing its movement using reinforcement learning techniques.

## 📌 Features

1. Q-learning with a discretized state space
2. Epsilon-greedy strategy for balancing exploration & exploitation
3. Reward shaping to encourage velocity and reaching the goal
4. Saves the best Q-table for improved learning
5. Generates a video of a successful episode using MoviePy

## 🚀 Installation

1️⃣ Clone the Repository
```
git clone https://github.com/Ajith-Kumar-Nelliparthi/MountainCarContinuous-v0.git
cd MountainCarContinous-v0
```

2️⃣ Create a Virtual Environment (Optional but recommended)
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3️⃣ Install Dependencies
```
pip install -r requirements.txt
```

## 🏁 How to Train the Agent

Run the following command to train the Q-learning agent:
```
python q.py
```
## Training Process

1. The agent learns through 25,000 episodes (default)
2. Updates the Q-table based on rewards
3. Saves the best performing model
4. Saves a video (success.mp4) of the best episode

## 🎥 Viewing the Learned Policy

After training, you can visualize the best model using:
```
python q.py --test
```
This will run the trained agent in the environment without learning, using the saved best_q_table.npy file.

## 🔧 Hyperparameters

You can adjust the training parameters in q.py:
```
LEARNING_RATE = 0.1    # Learning rate for Q-learning updates
DISCOUNT = 0.95        # Discount factor for future rewards
EPISODES = 25000       # Number of training episodes
SHOW_EVERY = 3000      # Render every N episodes
DISCRETE_OS_SIZE = [40, 40]  # Discretization of state space
```
## 📂 Project Structure
```
📁 MountainCar-Q-Learning
│── q.py                # Main training script
│── best_q_table.npy    # Saved Q-table (after training)
│── requirements.txt    # Dependencies
│── results/             # Folder for saving videos
```
## 📜 References

[OpenAI Gym](https://gymnasium.farama.org/environments/classic_control/mountain_car/)

[Q-learning Algorithm](https://en.wikipedia.org/wiki/Q-learning)

## 🤝 Contributing

Feel free to fork this repository and contribute with improvements!

## 📧 Contact

For any questions, reach out to: \

[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/Ajith532542840)\
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nelliparthi-ajith-233803262)\
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](nelliparthi123@gmail.com)

## 🌟 If you like this project, give it a star! ⭐

