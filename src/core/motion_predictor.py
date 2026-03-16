import torch
import torch.nn as nn
import numpy as np

class MotionLSTM(nn.Module):
    def __init__(self, input_features=9, hidden_size=64, num_layers=2, output_features=9, pred_len=5):
        super(MotionLSTM, self).__init__()
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_features = output_features
        self.pred_len = pred_len
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_features, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_features * pred_len)
        
    def forward(self, x):
        # x is expected to be of shape (batch, seq_len, input_features)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # We only want the output from the last time step
        out = out[:, -1, :]
        
        # Pass through the linear layer
        out = self.fc(out)
        
        # Reshape to (batch_size, pred_len, output_features)
        out = out.view(-1, self.pred_len, self.output_features)
        return out

class MotionPredictorEngine:
    def __init__(self, model_path='models/motion_lstm.pth', seq_len=20, pred_len=5, num_features=9):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = MotionLSTM(input_features=num_features, 
                                hidden_size=64, 
                                num_layers=2, 
                                output_features=num_features, 
                                pred_len=pred_len).to(self.device)
        self.model.eval()
        
        # Try to load weights
        self.is_loaded = False
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.is_loaded = True
            print(f"[MotionPredictor] Successfully loaded weights from {model_path} onto {self.device}")
        except Exception as e:
            print(f"[MotionPredictor] Warning: Could not load weights from {model_path}. Predictor disabled. Error: {e}")
            
    def predict(self, history):
        """
        history: list or numpy array of the last seq_len frames of joint angles.
                 Shape expected: (seq_len, num_features)
        Returns: 
                 list of dictionaries representing the next pred_len frames, or None.
        """
        if not self.is_loaded or len(history) < self.seq_len:
            return None
            
        # Convert to numpy and then tensor
        hist_arr = np.array(history, dtype=np.float32)
        # expected internal shape: (20, 9)
        if hist_arr.shape != (self.seq_len, self.num_features):
            return None
            
        # Add batch dimension
        hist_tensor = torch.tensor(hist_arr).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred_tensor = self.model(hist_tensor)
            
        # Remove batch dimension and convert back to numpy
        pred_arr = pred_tensor.squeeze(0).cpu().numpy()
        
        return pred_arr.tolist()
