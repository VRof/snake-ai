import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class DQNAgent:
    def __init__(self, state_size=11, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.min_memory_to_train = 10000  # Minimum experiences before training
        self.epsilon = 0.9
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.learning_rate = 0.002
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_freq = 500
        self.train_freq = 5
        self.step_count = 0
        self.batch_size = 2048
        self.predict_model = self.model
        self.update_target_model()
        
    def _build_model(self):
        model = Sequential([
            Dense(256, input_dim=self.state_size, activation='relu'),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        self.predict_model = self.target_model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, states):
        if np.random.rand() <= self.epsilon:
            return [random.randrange(self.action_size) for _ in states]
        
        states_array = np.array(states)
        q_values = self.predict_model.predict(states_array, verbose=0, batch_size=len(states))
        return np.argmax(q_values, axis=1)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        target_q = self.target_model.predict(next_states, verbose=0, batch_size=self.batch_size)
        current_q = self.model.predict(states, verbose=0, batch_size=self.batch_size)
        
        max_next_q = np.max(target_q, axis=1)
        updated_q = rewards + 0.99 * max_next_q * (1 - dones)
        
        indices = np.arange(self.batch_size)
        current_q[indices, actions] = updated_q
        
        self.model.train_on_batch(states, current_q)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay