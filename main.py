import pygame
import numpy as np
import tensorflow as tf
from collections import deque
import time
from config import GRID_SIZE, GRID_WIDTH, GRID_HEIGHT, NUM_PARALLEL_GAMES, WINDOW_WIDTH, WINDOW_HEIGHT
from game import SnakeGame
from agent import DQNAgent
from utils import *

# GPU configuration
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.optimizer.set_jit(True)
    print("🚀 GPU found and configured")
else:
    print("⚠️  No GPU found, using CPU")

# Initialize Pygame
pygame.init()

def main():
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption('Snake AI Demo')
    clock = pygame.time.Clock()
    
    # Minigame size - slightly smaller to fit everything
    MINI_SIZE = 11
    
    # Pre-render game surfaces
    main_surface = create_game_surface(GRID_SIZE, is_main=True)
    mini_surfaces = [create_game_surface(MINI_SIZE) for _ in range(12)]
    
    # Create games and agent
    games = [SnakeGame() for _ in range(NUM_PARALLEL_GAMES)]
    agent = DQNAgent()
    
    episode = 0
    max_score = 0
    running = True
    scores_last_10 = deque(maxlen=10)
    scores_last_100 = deque(maxlen=100)
    
    # Position calculations - 3 rows of 4 minigames
    main_x, main_y = 40, 40
    mini_game_width = GRID_WIDTH * MINI_SIZE
    mini_game_height = GRID_HEIGHT * MINI_SIZE
    games_per_row = 4
    
    # Calculate start position for minigames
    start_x = main_x + GRID_WIDTH * GRID_SIZE + 60
    start_y = main_y + 10
    
    mini_positions = []
    
    # Position minigames in 3 rows of 4 with reduced spacing
    for i in range(12):
        row = i // games_per_row
        col = i % games_per_row
        mini_positions.append((
            start_x + col * (mini_game_width + 20),
            start_y + row * (mini_game_height + 15),
            i + 1
        ))
    
    # Calculate panel top position to avoid overlapping minigames
    last_row_y = start_y + 2 * (mini_game_height + 15) + mini_game_height
    panel_top = last_row_y + 20
    
    # Ensure panel doesn't go off screen
    panel_top = min(panel_top, WINDOW_HEIGHT - 150)
    
    # For tracking performance
    last_log_time = time.time()
    last_step_count = 0
    steps_per_sec = 0
    
    while running:
        # Reset games
        for game in games:
            game.reset()
        
        while any(not game.game_over for game in games) and running:
            start_time = time.time()
            
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            if not running:
                break
            
            # Batch processing: Collect states for all active games
            active_games = []
            active_states = []
            for i, game in enumerate(games):
                if not game.game_over:
                    active_games.append(game)
                    active_states.append(game.get_state())
            
            # Batch prediction and action execution
            if active_states:
                actions = agent.act(active_states)
                
                for game, state, action in zip(active_games, active_states, actions):
                    next_state, reward, done = game.step(action)
                    agent.remember(state, action, reward, next_state, done)
                    agent.step_count += 1
            
            # Train periodically (with min_memory_to_train condition)
            if len(agent.memory) >= agent.min_memory_to_train and agent.step_count % agent.train_freq == 0:
                agent.replay()
            
            # Update target network
            if agent.step_count % agent.update_target_freq == 0:
                agent.update_target_model()
            
            # Rendering
            screen.fill(BLACK)
            
            # Draw main game
            draw_snake_game(screen, main_surface, games[0], main_x, main_y, GRID_SIZE)
            
            # Draw mini games
            for i, pos in enumerate(mini_positions):
                if i < len(games) - 1:
                    x, y, num = pos
                    draw_mini_game(screen, mini_surfaces[i], games[i+1], x, y, MINI_SIZE, num)
            
            # Performance metrics
            current_time = time.time()
            if current_time - last_log_time > 1:
                steps_per_sec = (agent.step_count - last_step_count) / (current_time - last_log_time)
                last_log_time = current_time
                last_step_count = agent.step_count
            
            # Draw info panel
            avg_score_100 = sum(scores_last_100) / len(scores_last_100) if scores_last_100 else 0
            draw_info_panel(screen, games, agent, episode, max_score, 
                           scores_last_10, avg_score_100, steps_per_sec, panel_top)
            
            pygame.display.flip()
            
            # Adaptive frame rate
            frame_time = time.time() - start_time
            target_fps = min(60, max(10, int(1.0 / max(0.01, frame_time))))
            clock.tick(target_fps)
        
        if not running:
            break
        
        # Update scores
        episode_scores = [game.score for game in games]
        main_score = games[0].score
        best_score = max(episode_scores)
        max_score = max(max_score, best_score)
        scores_last_10.append(main_score)
        scores_last_100.append(main_score)
        episode += 1
    
    pygame.quit()

if __name__ == "__main__":
    main()