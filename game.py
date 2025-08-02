import numpy as np
import random
from config import GRID_WIDTH, GRID_HEIGHT, UP, RIGHT, DOWN, LEFT

class SnakeGame:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.snake = [(GRID_WIDTH//2, GRID_HEIGHT//2)]
        self.direction = RIGHT
        self.food = self._place_food()
        self.score = 0
        self.game_over = False
        self.steps = 0
        self.steps_since_food = 0
        self.prev_distance = self._get_food_distance()
        
    def _place_food(self):
        available_positions = {(x, y) for x in range(GRID_WIDTH) for y in range(GRID_HEIGHT)}
        available_positions -= set(self.snake)
        return random.choice(tuple(available_positions)) if available_positions else (0, 0)
    
    def _get_food_distance(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        return abs(food_x - head_x) + abs(food_y - head_y)
    
    def get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        return np.array([
            # Danger detection
            self._is_collision(0),   # straight
            self._is_collision(-1),  # left
            self._is_collision(1),   # right
            
            # Food direction
            (food_x - head_x) / GRID_WIDTH,
            (food_y - head_y) / GRID_HEIGHT,
            
            # Current direction
            float(self.direction == UP),
            float(self.direction == RIGHT), 
            float(self.direction == DOWN),
            float(self.direction == LEFT),
            
            # Snake length
            len(self.snake) / (GRID_WIDTH * GRID_HEIGHT),
            
            # Distance to food
            self._get_food_distance() / (GRID_WIDTH + GRID_HEIGHT)
        ], dtype=np.float32)
    
    def _is_collision(self, turn):
        new_dir = (self.direction + turn) % 4
        head_x, head_y = self.snake[0]
        
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][new_dir]
        new_x, new_y = head_x + dx, head_y + dy
        
        return float(
            new_x < 0 or new_x >= GRID_WIDTH or 
            new_y < 0 or new_y >= GRID_HEIGHT or 
            (new_x, new_y) in self.snake
        )
    
    def step(self, action):
        # Update direction
        if action == 1:  # right
            self.direction = (self.direction + 1) % 4
        elif action == 2:  # left
            self.direction = (self.direction - 1) % 4
        
        # Move snake
        head_x, head_y = self.snake[0]
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][self.direction]
        new_head = (head_x + dx, head_y + dy)
        
        # Check collisions
        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or 
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT or 
            new_head in self.snake):
            self.game_over = True
            return self.get_state(), -10, True
        
        self.snake.insert(0, new_head)
        self.steps += 1
        self.steps_since_food += 1
        
        # Food collection
        reward = 0
        if new_head == self.food:
            self.score += 1
            reward = 10
            self.food = self._place_food()
            self.prev_distance = self._get_food_distance()
            self.steps_since_food = 0
        else:
            self.snake.pop()
            new_distance = self._get_food_distance()
            reward = 1 if new_distance < self.prev_distance else -0.1
            self.prev_distance = new_distance
        
        # Early stopping if snake isn't making progress
        max_steps_without_food = 100 + 20 * self.score
        if self.steps_since_food > max_steps_without_food:
            self.game_over = True
            reward = -5
        
        # Small reward for staying alive
        reward += 0.01
        
        return self.get_state(), reward, self.game_over