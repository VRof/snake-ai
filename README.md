# Snake AI - Deep Q-Network Implementation

A high-performance Snake game AI that uses Deep Q-Learning (DQN) to train an intelligent agent. The project features parallel game training with real-time visualization of 13 simultaneous games.

![Snake AI Demo](https://img.shields.io/badge/AI-Deep%20Q--Learning-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

![](https://github.com/VRof/snake-ai/blob/main/demo.gif)

## 📺 YouTube Demo

[Watch On YouTube](https://www.youtube.com/watch?v=ZLUIE2cbBAw)

## 🎮 Features

- **Multi-Game Training**: 13 parallel Snake games running simultaneously
- **Deep Q-Network**: Advanced neural network with experience replay and target networks
- **Real-time Visualization**: Main game display with 12 mini training games
- **GPU Acceleration**: Optimized for CUDA-enabled GPUs with fallback to CPU
- **Performance Monitoring**: Live stats including scores, epsilon decay, and training metrics

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended for better performance)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/VRof/snake-ai
cd snake-ai
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv snake_ai_env
source snake_ai_env/bin/activate  # On Windows: snake_ai_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install tensorflow numpy pygame
```

4. **Run the Snake AI**
```bash
python main.py
```

## 📦 Project Structure

```
snake-ai/
├── main.py
├── agent.py         # DQN Agent implementation
├── game.py          # Snake game logic
├── config.py        # Game configuration constants
├── utils.py         # Rendering and utility functions
```

## 🧠 How It Works

### Deep Q-Network Architecture

The AI uses a Deep Q-Network with the following architecture:
- Input Layer: 11 state features
- Hidden Layers: 256 → 256 → 128 neurons (ReLU activation)
- Output Layer: 3 actions (straight, turn left, turn right)

### State Representation

The agent observes 11 key features:
- **Collision Detection**: Danger ahead, left, and right
- **Food Direction**: Relative position to food (x, y)
- **Current Direction**: One-hot encoded direction vector
- **Snake Length**: Normalized snake size
- **Food Distance**: Manhattan distance to food

### Training Features

- **Experience Replay**: Stores 100,000 experiences for stable learning
- **Target Network**: Separate target network updated every 500 steps
- **Epsilon-Greedy**: Exploration starts at 90%, decays to 1%
- **Reward System**: 
  - +10 for eating food
  - +1 for moving closer to food
  - -0.1 for moving away from food
  - -10 for collision/death
  - +0.01 for staying alive

**Happy Training! 🐍🤖**
