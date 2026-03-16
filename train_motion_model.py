import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from src.core.motion_predictor import MotionLSTM

def generate_synthetic_data(num_samples=1000, seq_len=20, pred_len=5, num_features=9):
    """
    Generate synthetic data mimicking human joint movements (sinusoidal waves).
    """
    total_len = seq_len + pred_len
    X = np.zeros((num_samples, seq_len, num_features), dtype=np.float32)
    y = np.zeros((num_samples, pred_len, num_features), dtype=np.float32)
    
    for i in range(num_samples):
        # Base frequency and phase for this sample
        freqs = np.random.uniform(0.05, 0.2, size=num_features)
        phases = np.random.uniform(0, 2 * np.pi, size=num_features)
        amps = np.random.uniform(10, 50, size=num_features)
        offsets = np.random.uniform(90, 150, size=num_features)
        
        # Generate the full time series for this sample
        t = np.arange(total_len)
        
        for j in range(num_features):
            wave = amps[j] * np.sin(2 * np.pi * freqs[j] * t + phases[j]) + offsets[j]
            # Add some noise
            noise = np.random.normal(0, 2, size=total_len)
            series = wave + noise
            
            X[i, :, j] = series[:seq_len]
            y[i, :, j] = series[seq_len:]
            
    return torch.tensor(X), torch.tensor(y)

def train_model():
    print("Generating synthetic dataset...")
    X_train, y_train = generate_synthetic_data(num_samples=5000, seq_len=20, pred_len=5, num_features=9)
    X_val, y_val = generate_synthetic_data(num_samples=1000, seq_len=20, pred_len=5, num_features=9)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = MotionLSTM(input_features=9, hidden_size=64, num_layers=2, output_features=9, pred_len=5).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    epochs = 50
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val.to(device))
                val_loss = criterion(val_outputs, y_val.to(device)).item()
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
    # Save the model
    os.makedirs('models', exist_ok=True)
    save_path = 'models/motion_lstm.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Weights saved to {save_path}")

if __name__ == "__main__":
    train_model()
