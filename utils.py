import pygame
import numpy as np
from config import GRID_WIDTH, GRID_HEIGHT
import tensorflow as tf

# Colors
BLACK = (10, 10, 15)
DARK_BLUE = (20, 20, 40)
GREEN = (0, 255, 0)
RED = (255, 50, 50)
BLUE = (50, 150, 255)
WHITE = (240, 240, 255)
GRAY = (100, 100, 120)
PURPLE = (180, 70, 220)
YELLOW = (255, 255, 0)
CYAN = (0, 200, 200)

# Create gradient colors for snake body
def get_snake_color(index, length):
    ratio = index / max(1, length-1)
    r = int(255 * (1 - ratio))
    g = int(100 * ratio)
    b = int(255 * ratio)
    return (r, g, b)

# Pre-render game elements
def create_game_surface(cell_size, is_main=False):
    game_width = GRID_WIDTH * cell_size
    game_height = GRID_HEIGHT * cell_size
    surface = pygame.Surface((game_width, game_height))
    surface.fill(DARK_BLUE)
    border_color = WHITE if is_main else GRAY
    pygame.draw.rect(surface, border_color, surface.get_rect(), 2)
    return surface

def draw_snake_game(screen, surface, game, offset_x, offset_y, cell_size):
    """Draw snake game on pre-created surface"""
    # Clear surface
    surface.fill(DARK_BLUE)
    pygame.draw.rect(surface, WHITE, surface.get_rect(), 2)
    
    # Draw snake with gradient colors
    snake_length = len(game.snake)
    for i, (x, y) in enumerate(game.snake):
        rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
        if i == 0:  # Head
            pygame.draw.rect(surface, GREEN, rect)
            pygame.draw.rect(surface, WHITE, rect, 1)
        else:  # Body
            color = get_snake_color(i, snake_length)
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, (30, 30, 50), rect, 1)
    
    # Draw food
    food_rect = pygame.Rect(game.food[0] * cell_size, game.food[1] * cell_size, cell_size, cell_size)
    pygame.draw.rect(surface, RED, food_rect)
    pygame.draw.circle(surface, YELLOW, 
                      (food_rect.x + cell_size//2, food_rect.y + cell_size//2),
                      cell_size//3)
    
    # Blit to screen
    screen.blit(surface, (offset_x, offset_y))

def draw_mini_game(screen, surface, game, offset_x, offset_y, cell_size, game_num):
    # Clear surface
    surface.fill(DARK_BLUE)
    pygame.draw.rect(surface, GRAY, surface.get_rect(), 1)
    
    # Draw snake with simple gradient
    snake_length = len(game.snake)
    for i, (x, y) in enumerate(game.snake):
        rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
        if i == 0:
            pygame.draw.rect(surface, GREEN, rect)
        else:
            color = get_snake_color(i, snake_length)
            pygame.draw.rect(surface, color, rect)
    
    # Draw food
    food_rect = pygame.Rect(game.food[0] * cell_size, game.food[1] * cell_size, cell_size, cell_size)
    pygame.draw.rect(surface, RED, food_rect)
    
    # Blit to screen
    screen.blit(surface, (offset_x, offset_y))
    
    # Draw game number and score
    font = pygame.font.Font(None, 14)
    score_text = font.render(f'{game.score}', True, YELLOW)
    # Draw score in the top-left corner
    pygame.draw.rect(screen, (40, 40, 60), (offset_x, offset_y - 15, 30, 15))
    screen.blit(score_text, (offset_x + 5, offset_y - 10))

def draw_info_panel(screen, games, agent, episode, max_score, scores_last_10, avg_score_100, steps_per_sec, panel_top):
    panel_rect = pygame.Rect(0, panel_top, screen.get_width(), screen.get_height() - panel_top)
    pygame.draw.rect(screen, (25, 25, 40), panel_rect)
    pygame.draw.line(screen, PURPLE, (0, panel_rect.top), (screen.get_width(), panel_rect.top), 2)
    
    # Use smaller font
    try:
        font = pygame.font.Font(None, 18)
        title_font = pygame.font.Font(None, 22)
    except:
        font = pygame.font.SysFont('Arial', 16)
        title_font = pygame.font.SysFont('Arial', 20)
    
    # Draw title
    title = title_font.render(f"Snake AI - Episode {episode}", True, YELLOW)
    screen.blit(title, (screen.get_width() // 2 - title.get_width() // 2, panel_rect.top + 10))
    
    # Game info
    total_score = sum(g.score for g in games)
    active_games = sum(1 for g in games if not g.game_over)
    main_score = games[0].score
    
    # Column 1: Game stats
    col1 = [
        f"Main Score: {main_score}",
        f"Total Score: {total_score}",
        f"Max Score: {max_score}",
        f"Active Games: {active_games}/{len(games)}"
    ]
    
    # Column 2: Learning stats
    col2 = [
        f"Epsilon: {agent.epsilon:.4f}",
        f"Memory: {len(agent.memory)}/{agent.memory.maxlen}",
        f"Steps/sec: {steps_per_sec:.1f}",
        f"Batch Size: {agent.batch_size}"
    ]
    
    # Column 3: Performance stats
    col3 = [
        f"Avg(10): {sum(scores_last_10)/len(scores_last_10):.2f}" if scores_last_10 else "Avg(10): 0.00",
        f"Avg(100): {avg_score_100:.2f}" if avg_score_100 > 0 else "Avg(100): 0.00",
        f"Main Steps: {games[0].steps}",
        f"GPU: {'Active' if tf.config.list_physical_devices('GPU') else 'Inactive'}"
    ]
    
    col_width = screen.get_width() // 3
    for i, text in enumerate(col1):
        y_pos = panel_rect.top + 40 + i * 22
        txt = font.render(text, True, CYAN)
        screen.blit(txt, (50, y_pos))
    
    for i, text in enumerate(col2):
        y_pos = panel_rect.top + 40 + i * 22
        txt = font.render(text, True, CYAN)
        screen.blit(txt, (col_width + 50, y_pos))
    
    for i, text in enumerate(col3):
        y_pos = panel_rect.top + 40 + i * 22
        txt = font.render(text, True, CYAN)
        screen.blit(txt, (2 * col_width + 50, y_pos))