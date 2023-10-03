import torch
import torch.optim as optim
import torch.nn as nn
from env import TimeSeriesPredictorEnv
from lstm import LSTMModel
from gen import gen_values

# Hyperparameters
LOOK_BACK = 40
EPOCHS = 100
LR = 0.001
values = gen_values()

# Create gym environment
env = TimeSeriesPredictorEnv(values, LOOK_BACK)
model = LSTMModel(LOOK_BACK, 100, 8)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    state = env.reset()
    total_loss = 0
    while True:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = model(state_tensor)
        next_state, reward, done, _ = env.step(action.detach().numpy())
        target = torch.FloatTensor([env.data[env.current_step]])
        loss = loss_function(action, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        state = next_state

        if done:
            break
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")
