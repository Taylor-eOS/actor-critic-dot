import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import random

pygame.init()
WIDTH, HEIGHT = 400, 400
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

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
    def forward(self, x):
        x = torch.relu(self.fc(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value
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
        hit_wall = 0
    elif new_row >= ROWS:
        hit_wall = 2
    elif new_col < 0:
        hit_wall = 3
    elif new_col >= COLS:
        hit_wall = 1
    else:
        hit_wall = -1
    if hit_wall != -1:
        if hit_wall in safe_wall_actions:
            return new_state, 10, True
        else:
            return new_state, -10, True
    return new_state, 0, False

def draw_grid(state):
    screen.fill(WHITE)
    for i in range(ROWS):
        for j in range(COLS):
            rect = pygame.Rect(j * GRID_SIZE, i * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)
    pygame.draw.rect(screen, RED, (0, 0, WIDTH, GRID_SIZE))
    pygame.draw.rect(screen, RED, (0, HEIGHT - GRID_SIZE, WIDTH, GRID_SIZE))
    pygame.draw.rect(screen, RED, (0, 0, GRID_SIZE, HEIGHT))
    pygame.draw.rect(screen, GREEN, (WIDTH - GRID_SIZE, 0, GRID_SIZE, HEIGHT))
    pygame.draw.rect(screen, BLUE, (state[1] * GRID_SIZE, state[0] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
    pygame.display.flip()

def train():
    global initial_epsilon
    epsilon = initial_epsilon
    num_episodes = 1000
    for episode in range(num_episodes):
        state = reset_game()
        state_tensor = torch.tensor([[1.0]], dtype=torch.float32)
        if random.uniform(0, 1) < epsilon:
            action = random.choice(range(action_dim))
            log_prob = None
            state_value = None
        else:
            action_probs, state_value = model(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()
            log_prob = action_dist.log_prob(torch.tensor(action))
        done = False
        while not done:
            delta_row, delta_col = actions[action]
            state[0] += delta_row
            state[1] += delta_col
            draw_grid(state)
            clock.tick(10)
            if state[0] < 0 or state[0] >= ROWS or state[1] < 0 or state[1] >= COLS:
                if action in safe_wall_actions:
                    reward = 10
                else:
                    reward = -10
                done = True
                break
        if state[0] < 0 or state[0] >= ROWS or state[1] < 0 or state[1] >= COLS:
            if action in safe_wall_actions:
                reward = 10
            else:
                reward = -10
        else:
            reward = 0
        draw_grid(state)
        pygame.display.flip()
        if log_prob is not None:
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
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Epsilon: {epsilon:.4f}")
    run_simulation()

def run_simulation():
    state = reset_game()
    state_tensor = torch.tensor([[1.0]], dtype=torch.float32)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        with torch.no_grad():
            action_probs, _ = model(state_tensor)
            action = torch.argmax(action_probs).item()
        done = False
        while not done:
            delta_row, delta_col = actions[action]
            state[0] += delta_row
            state[1] += delta_col
            draw_grid(state)
            clock.tick(10)
            if state[0] < 0 or state[0] >= ROWS or state[1] < 0 or state[1] >= COLS:
                done = True
                break
        if state[0] < 0 or state[0] >= ROWS or state[1] < 0 or state[1] >= COLS:
            if action in safe_wall_actions:
                reward = 10
            else:
                reward = -10
        else:
            reward = 0
        draw_grid(state)
        pygame.display.flip()
        state = reset_game()
        state_tensor = torch.tensor([[1.0]], dtype=torch.float32)
train()
