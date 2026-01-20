import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 

# --- CONFIGURATION ---
DATA_PATH = os.path.join('MP_Data')
ACTIONS = np.array(['chumreapsour', 'orkun', 'trov', 'nothing', 'howru', 'mineyte', 'deaf', 'soursdey', 'WC', 'i dont understand', 'zero']) 
EPOCHS = 50
BATCH_SIZE = 16

# --- 1. LOAD DATA ---
label_map = {label:num for num, label in enumerate(ACTIONS)}
sequences, labels = [], []

print("Loading data...")
for action in ACTIONS:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        print(f"ERROR: Folder {action} not found!")
        continue
        
    file_list = os.listdir(action_path)
    for file_name in file_list:
        res = np.load(os.path.join(action_path, file_name))
        sequences.append(res)
        labels.append(label_map[action])

X = np.array(sequences)
y = np.array(labels)

# Split: 5% of data is hidden from the model to test it later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

print(f"Training sequences: {len(X_train)}")
print(f"Testing sequences:  {len(X_test)}")

# --- 2. SETUP DEVICE (Universal Support) ---
if torch.cuda.is_available():
    device = torch.device('cuda')         # NVIDIA GPU (Windows/Linux)
elif torch.backends.mps.is_available():
    device = torch.device('mps')          # Apple Silicon GPU (Mac M1/M2/M3)
else:
    device = torch.device('cpu')          # No GPU found (Standard Laptop)

print(f"--- TRAINING ON: {device} ---") 

class KSLDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.tensor(x_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.long)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

train_loader = DataLoader(KSLDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

# --- 3. MODEL (Increased Complexity) ---
class KSLModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(KSLModel, self).__init__()
        # Increased hidden_size from 64 to 128 for better learning capacity
        self.lstm = nn.LSTM(input_size, 128, num_layers=3, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(128, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.relu(self.fc1(out))
        return self.fc2(out)

INPUT_SIZE = 258 
model = KSLModel(INPUT_SIZE, len(ACTIONS)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) 

# --- 4. TRAIN ---
print("\n--- STARTING TRAINING ---")
for epoch in range(EPOCHS):
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')

# --- 5. EVALUATE (TEST) ---
print("\n--- TESTING MODEL ---")
model.eval() # Switch to evaluation mode
with torch.no_grad():
    # Convert test data to tensor
    test_inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
    test_labels = y_test # Keep as numpy for sklearn

    # Get predictions
    outputs = model(test_inputs)
    _, predictions = torch.max(outputs, 1)
    
    # Move back to CPU for metric calculation
    predictions = predictions.cpu().numpy()
    
    acc = accuracy_score(test_labels, predictions)
    print(f"FINAL ACCURACY: {acc * 100:.2f}%")

# --- 6. SAVE ---
torch.save(model.state_dict(), 'ksl_model.pth')
print("Model saved as 'ksl_model.pth'")