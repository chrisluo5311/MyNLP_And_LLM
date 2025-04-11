import pygame
import random
import sys
import os
import math
from datetime import datetime

pygame.init()
pygame.mixer.init()

# Constants
WIDTH, HEIGHT = 600, 400
CELL_SIZE = 20
FPS = 10
SCORE_FILE = "scores.txt"

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
BLUE = (0, 100, 255)

# Setup display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")

clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 25)
large_font = pygame.font.SysFont("Arial", 40)

# Load sounds
eat_sound = pygame.mixer.Sound("./sounds/eat.wav")
eat_sound.set_volume(0.3)
powerup_sound = pygame.mixer.Sound("sounds/powerup.wav")
powerup_sound.set_volume(0.5)
gameover_sound = pygame.mixer.Sound("sounds/gameover.wav")
gameover_sound.set_volume(0.5)

NUM_OBSTACLES = 5
POWER_UP_DURATION = 5000  # milliseconds
POWER_UP_INTERVAL = 15000  # spawn every 15s

SLOW_FPS = 5  # snake speed during power-up effect


def draw_snake(snake_body, power_up_active=False, pulse_phase=0.0):
    if power_up_active:
        # Brightness range: 180–255 instead of 1–255
        brightness = int(75 * math.sin(pulse_phase) + 180)
        color = (0, brightness, brightness)  # Cyan pulse
    else:
        color = GREEN

    for segment in snake_body:
        pygame.draw.rect(screen, color, pygame.Rect(segment[0], segment[1], CELL_SIZE, CELL_SIZE))


def draw_food(food_pos):
    pygame.draw.rect(screen, RED, pygame.Rect(food_pos[0], food_pos[1], CELL_SIZE, CELL_SIZE))

def draw_score(score, best_score, powerup_remaining=None):
    score_text = font.render(f"Score: {score}", True, WHITE)
    best_text = font.render(f"Best: {best_score}", True, RED)

    # Background box
    padding = 10
    box_width = max(score_text.get_width(), best_text.get_width()) + padding * 2
    box_height = score_text.get_height() + best_text.get_height() + padding * 3
    pygame.draw.rect(screen, GRAY, (5, 5, box_width, box_height), border_radius=6)

    screen.blit(score_text, (5 + padding, 5 + padding))
    screen.blit(best_text, (5 + padding, 5 + padding + score_text.get_height() + 5))

    # Draw power-up countdown (top-right)
    if powerup_remaining is not None and powerup_remaining > 0:
        timer_text = font.render(f"Power-Up: {powerup_remaining:.1f}s", True, (0, 255, 255))
        screen.blit(timer_text, (WIDTH - timer_text.get_width() - 10, 10))

def draw_border_glow(pulse_phase=0.0, thickness=5):
    brightness = int(75 * math.sin(pulse_phase) + 180)
    glow_color = (0, brightness, brightness)  # Cyan pulse

    pygame.draw.rect(screen, glow_color, pygame.Rect(0, 0, WIDTH, thickness))
    pygame.draw.rect(screen, glow_color, pygame.Rect(0, HEIGHT - thickness, WIDTH, thickness))
    pygame.draw.rect(screen, glow_color, pygame.Rect(0, 0, thickness, HEIGHT))
    pygame.draw.rect(screen, glow_color, pygame.Rect(WIDTH - thickness, 0, thickness, HEIGHT))


def generate_food(snake):
    score_box_width = 170   # Max width of score box
    score_box_height = 70   # Height of score box

    while True:
        x = random.randint(0, (WIDTH - CELL_SIZE) // CELL_SIZE) * CELL_SIZE
        y = random.randint(0, (HEIGHT - CELL_SIZE) // CELL_SIZE) * CELL_SIZE

        # Reject if inside the gray score box
        if x < score_box_width and y < score_box_height:
            continue

        if [x, y] not in snake:
            return [x, y]



def draw_button(text, x, y, w, h, color, hover_color, mouse_pos, selected=False):
    is_hovered = x < mouse_pos[0] < x + w and y < mouse_pos[1] < y + h
    if selected:
        bg_color = hover_color
    elif is_hovered:
        bg_color = tuple(min(255, c + 40) for c in color)  # light hover tint
    else:
        bg_color = color

    pygame.draw.rect(screen, bg_color, (x, y, w, h))
    label = font.render(text, True, WHITE)
    screen.blit(label, (x + (w - label.get_width()) // 2, y + (h - label.get_height()) // 2))
    return is_hovered



def save_score(score):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = (score, timestamp)
    scores = read_raw_scores()
    scores.append(entry)
    scores.sort(key=lambda x: x[0], reverse=True)
    scores = scores[:5]
    with open(SCORE_FILE, "w") as f:
        for s, t in scores:
            f.write(f"{s},{t}\n")


def read_raw_scores():
    if not os.path.exists(SCORE_FILE):
        return []
    with open(SCORE_FILE, "r") as f:
        lines = f.readlines()
    result = []
    for line in lines:
        try:
            parts = line.strip().split(",")
            score = int(parts[0])
            timestamp = parts[1] if len(parts) > 1 else ""
            result.append((score, timestamp))
        except:
            continue
    return result


def read_best_score():
    scores = read_raw_scores()
    return scores[0][0] if scores else 0


def history_screen():
    scores = read_raw_scores()
    selected_index = 0
    buttons = ["Back"]

    while True:
        mouse_pos = pygame.mouse.get_pos()
        screen.fill(BLACK)

        title = large_font.render("Top 5 Scores", True, WHITE)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 40))

        if scores:
            for i, (score, timestamp) in enumerate(scores):
                line = font.render(f"{i + 1}. {score} pts - {timestamp}", True, GRAY)
                screen.blit(line, (WIDTH // 2 - 150, 100 + i * 30))
        else:
            no_data = font.render("No scores recorded yet.", True, GRAY)
            screen.blit(no_data, (WIDTH // 2 - no_data.get_width() // 2, HEIGHT // 2))

        # Draw Back button
        y = HEIGHT - 70
        is_hovered = draw_button("Back", WIDTH // 2 - 100, y, 200, 50,
                                GRAY, BLUE, mouse_pos, selected=(selected_index == 0))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_RETURN, pygame.K_KP_ENTER]:
                    return
                elif event.key in [pygame.K_UP, pygame.K_DOWN]:
                    selected_index = 0  # only one button, toggle not needed but could be extended later

            elif event.type == pygame.MOUSEBUTTONDOWN and is_hovered:
                return

        clock.tick(15)



def game_over_screen(score):
    save_score(score)
    buttons = ["Play Again", "Main Menu", "Quit"]
    selected_index = 0
    gameover_sound.play()
    while True:
        mouse_pos = pygame.mouse.get_pos()
        screen.fill(BLACK)
        over_text = large_font.render(f"Game Over! Score: {score}", True, RED)
        screen.blit(over_text, (WIDTH // 2 - over_text.get_width() // 2, HEIGHT // 4))

        button_rects = []
        for i, label in enumerate(buttons):
            y = HEIGHT // 2 + i * 70
            is_selected = (i == selected_index)
            hovered = draw_button(label, WIDTH // 2 - 100, y, 200, 50,
                                GRAY, BLUE, mouse_pos, selected=is_selected)
            button_rects.append((label, hovered))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected_index = (selected_index - 1) % len(buttons)
                elif event.key == pygame.K_DOWN:
                    selected_index = (selected_index + 1) % len(buttons)
                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    choice = buttons[selected_index]
                    if choice == "Play Again":
                        return "RESTART"
                    elif choice == "Main Menu":
                        return "MENU"
                    elif choice == "Quit":
                        pygame.quit()
                        sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                for i, (label, hovered) in enumerate(button_rects):
                    if hovered:
                        if label == "Play Again":
                            return "RESTART"
                        elif label == "Main Menu":
                            return "MENU"
                        elif label == "Quit":
                            pygame.quit()
                            sys.exit()

        clock.tick(15)



def main_menu():
    buttons = ["Start Game", "AI Autoplay Mode", "History", "Quit"]
    selected_index = 0

    while True:
        mouse_pos = pygame.mouse.get_pos()
        screen.fill(BLACK)
        title = large_font.render("Snake Game", True, GREEN)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 50))
        # Made by JIDUNG, LO signature
        signature = font.render("Made by JIDUNG, LO", True, (180, 180, 180))
        screen.blit(signature, (
            WIDTH // 2 - signature.get_width() // 2,
            50 + title.get_height() + 5
        ))

        button_rects = []
        for i, label in enumerate(buttons):
            y = 150 + i * 70
            is_selected = (i == selected_index)
            color = BLUE if is_selected else GRAY
            hover_color = GREEN if is_selected else WHITE
            hovered = draw_button(label, WIDTH // 2 - 100, y, 200, 50, color, hover_color, mouse_pos)
            button_rects.append((label, hovered))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    selected_index = (selected_index + 1) % len(buttons)
                elif event.key == pygame.K_UP:
                    selected_index = (selected_index - 1) % len(buttons)
                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    choice = buttons[selected_index]
                    if choice == "Start Game":
                        return "START"
                    elif choice == "History":
                        history_screen()
                    elif choice == "AI Autoplay Mode":
                        return "AI"
                    elif choice == "Quit":
                        pygame.quit()
                        sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                for i, (label, hovered) in enumerate(button_rects):
                    if hovered:
                        if label == "Start Game":
                            return "START"
                        elif label == "History":
                            history_screen()
                        elif label == "AI Autoplay Mode":
                            return "AI"
                        elif label == "Quit":
                            pygame.quit()
                            sys.exit()

        clock.tick(15)

def generate_obstacles(snake, food):
    obstacles = []
    while len(obstacles) < NUM_OBSTACLES:
        x = random.randint(0, (WIDTH - CELL_SIZE) // CELL_SIZE) * CELL_SIZE
        y = random.randint(0, (HEIGHT - CELL_SIZE) // CELL_SIZE) * CELL_SIZE
        if [x, y] not in snake and [x, y] != food and not in_score_box(x, y):
            obstacles.append([x, y])
    return obstacles


def draw_obstacles(obstacles):
    for obs in obstacles:
        pygame.draw.rect(screen, (150, 75, 0), pygame.Rect(obs[0], obs[1], CELL_SIZE, CELL_SIZE))  # brown


def generate_power_up(entities_to_avoid):
    while True:
        x = random.randint(0, (WIDTH - CELL_SIZE) // CELL_SIZE) * CELL_SIZE
        y = random.randint(0, (HEIGHT - CELL_SIZE) // CELL_SIZE) * CELL_SIZE
        if [x, y] not in entities_to_avoid and not in_score_box(x, y):
            return [x, y]


def draw_power_up(pos):
    pygame.draw.rect(screen, (0, 255, 255), pygame.Rect(pos[0], pos[1], CELL_SIZE, CELL_SIZE))  # cyan

def in_score_box(x, y):
    score_box_width = 170
    score_box_height = 70
    return x < score_box_width and y < score_box_height


def game_loop(ai_mode=False):
    snake = [[100, 100], [80, 100], [60, 100]]
    direction = 'RIGHT'
    change_to = direction
    score = 0
    best_score = read_best_score()

    food = generate_food(snake)
    obstacles = generate_obstacles(snake, food)

    power_up = None
    power_up_active = False
    power_up_spawn_time = pygame.time.get_ticks() + POWER_UP_INTERVAL
    power_up_expire_time = 0
    pulse_phase = 0.0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and direction != 'DOWN':
                    change_to = 'UP'
                elif event.key == pygame.K_DOWN and direction != 'UP':
                    change_to = 'DOWN'
                elif event.key == pygame.K_LEFT and direction != 'RIGHT':
                    change_to = 'LEFT'
                elif event.key == pygame.K_RIGHT and direction != 'LEFT':
                    change_to = 'RIGHT'

        # AI: Simple Greedy Pathfinding toward food
        if ai_mode:
            head_x, head_y = snake[0]

            # All four possible directions
            directions = {
                'UP': (head_x, head_y - CELL_SIZE),
                'DOWN': (head_x, head_y + CELL_SIZE),
                'LEFT': (head_x - CELL_SIZE, head_y),
                'RIGHT': (head_x + CELL_SIZE, head_y)
            }

            # Rank directions based on proximity to food
            ranked = sorted(directions.items(), key=lambda d: (
                    abs(d[1][0] - food[0]) + abs(d[1][1] - food[1])
            ))

            # Check each direction in ranked order
            for dir_name, pos in ranked:
                if (
                        pos not in snake and
                        pos not in obstacles and
                        0 <= pos[0] < WIDTH and
                        0 <= pos[1] < HEIGHT
                ):
                    # Valid and safe direction
                    # Avoid reversing
                    if dir_name == 'UP' and direction != 'DOWN':
                        change_to = 'UP'
                        break
                    elif dir_name == 'DOWN' and direction != 'UP':
                        change_to = 'DOWN'
                        break
                    elif dir_name == 'LEFT' and direction != 'RIGHT':
                        change_to = 'LEFT'
                        break
                    elif dir_name == 'RIGHT' and direction != 'LEFT':
                        change_to = 'RIGHT'
                        break

        direction = change_to
        head_x, head_y = snake[0]
        if direction == 'UP':
            head_y -= CELL_SIZE
        elif direction == 'DOWN':
            head_y += CELL_SIZE
        elif direction == 'LEFT':
            head_x -= CELL_SIZE
        elif direction == 'RIGHT':
            head_x += CELL_SIZE

        new_head = [head_x, head_y]
        snake.insert(0, new_head)

        # Eat food
        if new_head == food:
            score += 1
            eat_sound.play()
            food = generate_food(snake + obstacles)
        else:
            snake.pop()

        # Eat power-up
        now = pygame.time.get_ticks()
        if power_up and new_head == power_up:
            score += 5
            powerup_sound.play()
            power_up = None
            power_up_active = True
            power_up_expire_time = now + POWER_UP_DURATION
            power_up_spawn_time = power_up_expire_time + POWER_UP_INTERVAL

        # Collision checks
        if (
            head_x < 0 or head_x >= WIDTH or
            head_y < 0 or head_y >= HEIGHT or
            new_head in snake[1:] or
            new_head in obstacles
        ):
            break

        # Handle power-up spawn timing
        if not power_up and now >= power_up_spawn_time:
            power_up = generate_power_up(snake + obstacles + [food])
            power_up_expire_time = now + POWER_UP_DURATION
        elif power_up and now >= power_up_expire_time:
            power_up = None
            # Next one comes after 15 seconds again
            power_up_spawn_time = now + POWER_UP_INTERVAL
        elif power_up_active and now >= power_up_expire_time:
            power_up_active = False

        # Draw everything
        screen.fill(BLACK)
        draw_snake(snake, power_up_active, pulse_phase)
        draw_food(food)
        draw_obstacles(obstacles)
        if power_up:
            draw_power_up(power_up)

        # Compute remaining power-up time
        if power_up_active:
            powerup_remaining = (power_up_expire_time - now) / 1000
            draw_border_glow(pulse_phase)
        else:
            powerup_remaining = None

        draw_score(score, best_score, powerup_remaining)
        pygame.display.update()

        # Advance the glow phase (for pulsing effect)
        pulse_phase += 0.15  # speed of pulse
        if pulse_phase > 2 * math.pi:
            pulse_phase = 0.0

        # Control game speed
        current_fps = SLOW_FPS if power_up_active else FPS
        clock.tick(current_fps)

    return game_over_screen(score)



def main():
    while True:
        choice = main_menu()
        if choice == "START":
            while True:
                result = game_loop()
                if result == "RESTART":
                    continue
                elif result == "MENU":
                    break
        elif choice == "AI":
            while True:
                result = game_loop(ai_mode=True)
                if result == "RESTART":
                    continue
                elif result == "MENU":
                    break


if __name__ == "__main__":
    main()
