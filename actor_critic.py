import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

pygame.init()
WIDTH, HEIGHT = 440, 440
GRID_SIZE = 40
ROWS, COLS = HEIGHT // GRID_SIZE, WIDTH // GRID_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Actor-Critic Dot Simulation")
clock = pygame.time.Clock()
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
start_position = (ROWS // 2, COLS // 2)
actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
safe_wall_actions = [1]
initial_epsilon = 0.2
wall_status = {'top': True, 'bottom': True, 'left': True, 'right': False}

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.gru = nn.GRU(128, 128, batch_first=True)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
        
    def forward(self, x, hidden=None):
        x = torch.relu(self.fc(x))
        x, hidden = self.gru(x.unsqueeze(1), hidden)
        x = x.squeeze(1)
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value, hidden

input_dim = 1
action_dim = 4
model = ActorCritic(input_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
gamma = 0.99

def reset_game():
    return list(start_position)

def step(state, action):
    delta_row, delta_col = actions[action]
    new_row = state[0] + delta_row
    new_col = state[1] + delta_col
    new_state = [new_row, new_col]
    if new_row < 0:
        hit_wall = 'top'
    elif new_row >= ROWS:
        hit_wall = 'bottom'
    elif new_col < 0:
        hit_wall = 'left'
    elif new_col >= COLS:
        hit_wall = 'right'
    else:
        hit_wall = None
    if hit_wall is not None:
        reward = -10 if wall_status[hit_wall] else 10
        return new_state, reward, True
    return new_state, 0, False

def draw_grid(state):
    screen.fill(WHITE)
    for i in range(ROWS):
        for j in range(COLS):
            rect = pygame.Rect(j * GRID_SIZE, i * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)
    pygame.draw.rect(screen, RED if wall_status['top'] else GREEN, (0, 0, WIDTH, GRID_SIZE))
    pygame.draw.rect(screen, RED if wall_status['bottom'] else GREEN, (0, HEIGHT - GRID_SIZE, WIDTH, GRID_SIZE))
    pygame.draw.rect(screen, RED if wall_status['left'] else GREEN, (0, 0, GRID_SIZE, HEIGHT))
    pygame.draw.rect(screen, RED if wall_status['right'] else GREEN, (WIDTH - GRID_SIZE, 0, GRID_SIZE, HEIGHT))
    pygame.draw.rect(screen, BLUE, (state[1] * GRID_SIZE, state[0] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
    pygame.display.flip()

def handle_wall_click(event):
    if event.type == pygame.MOUSEBUTTONDOWN:
        x, y = event.pos
        if y < GRID_SIZE:
            wall_status['top'] = not wall_status['top']
        elif y > HEIGHT - GRID_SIZE:
            wall_status['bottom'] = not wall_status['bottom']
        elif x < GRID_SIZE:
            wall_status['left'] = not wall_status['left']
        elif x > WIDTH - GRID_SIZE:
            wall_status['right'] = not wall_status['right']

def run_simulation():
    state = reset_game()
    state_tensor = torch.tensor([[1.0]], dtype=torch.float32)
    hidden = None
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        with torch.no_grad():
            action_probs, _, hidden = model(state_tensor, hidden)
            action = torch.argmax(action_probs).item()
        done = False
        while not done:
            state, reward, done = step(state, action)
            draw_grid(state)
            clock.tick(10)
            pygame.display.flip()
        state = reset_game()
        state_tensor = torch.tensor([[1.0]], dtype=torch.float32)
        hidden = None

def train():
    global initial_epsilon
    epsilon = initial_epsilon
    num_episodes = 1000
    for episode in range(num_episodes):
        state = reset_game()
        state_tensor = torch.tensor([[1.0]], dtype=torch.float32)
        hidden = None
        total_reward = 0
        done = False
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = random.choice(range(action_dim))
                log_prob = None
                state_value = None
            else:
                action_probs, state_value, hidden = model(state_tensor, hidden)
                if hidden is not None:
                    hidden = hidden.detach()
                action = torch.argmax(action_probs).item()
                log_prob = torch.log(action_probs[0, action])
            new_state, reward, done = step(state, action)
            total_reward += reward
            draw_grid(new_state)
            clock.tick(10)
            if done:
                pass
            state = new_state
            state_tensor = torch.tensor([[1.0]], dtype=torch.float32)
            if done and log_prob is not None:
                advantage = torch.tensor([reward], dtype=torch.float32) - state_value.squeeze()
                actor_loss = -log_prob * advantage.detach()
                critic_loss = advantage.pow(2)
                loss = actor_loss + critic_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        epsilon = max(0.05, epsilon * 0.995)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if y < GRID_SIZE:
                    wall_status['top'] = not wall_status['top']
                elif y > HEIGHT - GRID_SIZE:
                    wall_status['bottom'] = not wall_status['bottom']
                elif x < GRID_SIZE:
                    wall_status['left'] = not wall_status['left']
                elif x > WIDTH - GRID_SIZE:
                    wall_status['right'] = not wall_status['right']
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")
    run_simulation()

train()

