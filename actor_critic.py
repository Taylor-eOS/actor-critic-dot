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
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
start_position = (ROWS // 2, COLS // 2)
actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
action_dim = 4
wall_status = {'top': True, 'bottom': True, 'left': True, 'right': False}

class CuriosityModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CuriosityModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        prediction = self.fc2(x)
        return prediction

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.curiosity = CuriosityModule(input_dim, hidden_dim)
    
    def forward(self, x, hidden=None):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.unsqueeze(1)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(1)
        x = F.relu(self.fc3(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

input_dim = 40  # 4 actions * 5 history + 5 rewards
model = ActorCritic(input_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
curiosity_optimizer = optim.Adam(model.curiosity.parameters(), lr=0.0005)
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

def draw_dot(state):
    screen.fill(WHITE)
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

def construct_history_vector(history_actions, history_rewards, history_length=5):
    history_vector = []
    for action in history_actions[-history_length:]:
        one_hot = [0, 0, 0, 0]
        one_hot[action] = 1
        history_vector.extend(one_hot)
    for reward in history_rewards[-history_length:]:
        history_vector.append(reward / 10.0)
    while len(history_vector) < input_dim:
        if len(history_vector) < 20:
            history_vector.extend([0, 0, 0, 0])
        else:
            history_vector.append(0.0)
    return history_vector

def run_simulation():
    state = reset_game()
    history_actions = []
    history_rewards = []
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            handle_wall_click(event)
        history_vector = construct_history_vector(history_actions, history_rewards)
        state_tensor = torch.tensor([history_vector], dtype=torch.float32)
        with torch.no_grad():
            action_probs, _ = model(state_tensor)
            action = torch.argmax(action_probs).item()
        done = False
        while not done:
            state, reward, done = step(state, action)
            history_actions.append(action)
            history_rewards.append(reward)
            draw_dot(state)
            clock.tick(10)
            pygame.display.flip()

def train():
    num_episodes = 2000
    history_length = 5
    for episode in range(num_episodes):
        state = reset_game()
        history_actions = []
        history_rewards = []
        done = False
        total_reward = 0
        history_vector = construct_history_vector(history_actions, history_rewards, history_length)
        state_tensor = torch.tensor([history_vector], dtype=torch.float32)
        action_probs, state_value = model(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample().item()
        new_state, reward, done = step(state, action)
        history_actions.append(action)
        history_rewards.append(reward)
        total_reward += reward
        while not done:
            new_state, reward, done = step(new_state, action)
            history_actions.append(action)
            history_rewards.append(reward)
            total_reward += reward
            draw_dot(new_state)
            clock.tick(10)
            pygame.display.flip()
        history_vector = construct_history_vector(history_actions, history_rewards, history_length)
        next_state_tensor = torch.tensor([history_vector], dtype=torch.float32)
        with torch.no_grad():
            _, next_state_value = model(next_state_tensor)
            target_value = torch.tensor([[reward]], dtype=torch.float32) if done else reward + gamma * next_state_value
        advantage = target_value - state_value
        actor_loss = -action_dist.log_prob(torch.tensor(action)) * advantage
        critic_loss = F.mse_loss(state_value, target_value)
        loss = actor_loss + critic_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        curiosity_loss = F.mse_loss(model.curiosity(state_tensor), torch.tensor([[reward]], dtype=torch.float32))
        curiosity_optimizer.zero_grad()
        curiosity_loss.backward()
        curiosity_optimizer.step()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            handle_wall_click(event)
        if (episode + 1) % 200 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}")
    run_simulation()

train()

