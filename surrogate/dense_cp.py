import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CoordinateDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(8, 128)  # 8 entry → 64 hidden neurones
        self.fc2 = nn.Linear(128, 64)  # 64 → 32 neurones
        self.fc3 = nn.Linear(64, 32)  # 64 → 32 neurones
        self.fc4 = nn.Linear(32, 1)  # 32 → 1 output
        self.relu = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x)) # relu cause positive values
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x) # output fully connected layer cause regression
        return x


class Dense_slant_dist_pred:
    def __init__(self, data_csv="all_test_with_real.csv"):
        self.data_csv = data_csv
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

    def load_data(self):
        df = pd.read_csv(self.data_csv)
        df = df.dropna(subset=["slant_distance", "bbox_area"])

        X = df[['x_A_norm', 'y_A_norm', 'x_B_norm', 'y_B_norm', 
                'x_C_norm', 'y_C_norm', 'x_D_norm', 'y_D_norm']].values
        y = df['slant_distance_zscore'].values

        print(f"All images : {len(X)}")

        subset_size = int(0.9 * len(X))
        X, _, y, _ = train_test_split(X, y, train_size=subset_size, random_state=42)

        print(f"Using {subset_size} images for training.")

        # Split train/val/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

    def train_model(self, epochs=10, batch_size=32, lr=0.001):
        train_dataset = CoordinateDataset(self.X_train, self.y_train)
        val_dataset = CoordinateDataset(self.X_val, self.y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.model = MLP().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs.squeeze(), targets)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
        torch.save(self.model.state_dict(), "cp_surrogate.pt")

    def evaluate_model(self):
        self.model = MLP().to(device)
        self.model.load_state_dict(torch.load("cp_surrogate.pt"))  # Charger les poids sauvegardés

        test_dataset = CoordinateDataset(self.X_test, self.y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = self.model(inputs)
                loss = nn.MSELoss()(outputs.squeeze(), targets)
                total_loss += loss.item()

        print(f"Test Loss: {total_loss / len(test_loader)}")

    def predict(self, X):
        # convert one sample to tensor for prediction
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # add batch dimension
        self.model.eval()
        with torch.no_grad():
            output = self.model(X)
        return output.item()
    
    def plot(self):
        model = MLP().to(device)
        model.load_state_dict(torch.load("cp_surrogate.pt", weights_only=True)) 
        
        torch_xval = torch.tensor(self.X_val.astype("float32")) # convert to float32
        torch_yval = torch.tensor(self.y_val) 
        predictions = model(torch_xval.to(device)).to("cpu")

        # Improved visualization style
        plt.style.use("seaborn-v0_8-darkgrid")

        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_val, predictions.detach().numpy(), alpha=0.5, label="Predictions", color="blue")

        # Adding the identity line y = x for reference
        min_val = min(self.y_val.min(), predictions.min().item())
        max_val = max(self.y_val.max(), predictions.max().item())
        plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="red", label="y = x (Ideal)")

        plt.xlabel("True slant distance (normalized z-score)")
        plt.ylabel("Predicted slant distance (normalized z-score)")
        plt.title("Slant Distance Prediction: Model vs. Ground Truth")

        plt.legend()
        plt.savefig('cp_improved.png')
        plt.show()



mlp_model = Dense_slant_dist_pred()
print("Loading data...")
mlp_model.load_data()
print("Training model...")
#mlp_model.train_model(epochs=50, batch_size=32)
print("Evaluating model...")
mlp_model.evaluate_model()
mlp_model.plot()
