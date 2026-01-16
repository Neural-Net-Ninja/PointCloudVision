import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ==================== Model Definition ====================
class WetnessPredictor(nn.Module):
    def __init__(self, input_size=5):
        super(WetnessPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# ==================== Data Loading & Preprocessing ====================
def load_and_preprocess_data(csv_path):
    """Load training data and prepare features/labels"""
    df = pd.read_csv(csv_path)
    
    # Extract features: Intensity, Temperature, FruitID, ClusterID, and spatial coordinates
    feature_cols = ['Intensity', 'Temperature', 'FruitID', 'ClusterID', 'X']
    X = df[feature_cols].values
    y = df['Wetness'].values.reshape(-1, 1)
    
    # Normalize features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # Normalize labels
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)
    
    return X_scaled, y_scaled, scaler_X, scaler_y

# ==================== Training Function ====================
def train_model(model, train_loader, val_loader, epochs=100, learning_rate=0.001, device='cpu'):
    """Train the neural network"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=10, verbose=False)
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    train_losses = []
    val_losses = []
    
    print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Best Val Loss':<15}")
    print("-" * 53)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"{epoch+1:<8} {train_loss:<15.6f} {val_loss:<15.6f} {best_val_loss:<15.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model, train_losses, val_losses

# ==================== Prediction Function ====================
def predict_wetness(model, X_new, scaler_X, scaler_y, device='cpu'):
    """Make predictions on new data"""
    model.eval()
    
    # Normalize input
    X_new_scaled = scaler_X.transform(X_new)
    X_tensor = torch.tensor(X_new_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy()
    
    # Denormalize predictions
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    return y_pred.flatten()

# ==================== Main Execution ====================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # ========== STEP 1: Load and preprocess training data ==========
    print("=" * 60)
    print("STEP 1: Loading and preprocessing training data")
    print("=" * 60)
    
    X_train, y_train, scaler_X, scaler_y = load_and_preprocess_data('training_data.csv')
    
    # Split into train and validation sets
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Create PyTorch datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train_split, dtype=torch.float32),
        torch.tensor(y_train_split, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"Training samples: {len(X_train_split)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Feature dimensions: {X_train.shape[1]}\n")
    
    # ========== STEP 2: Create and train model ==========
    print("=" * 60)
    print("STEP 2: Training the model")
    print("=" * 60 + "\n")
    
    model = WetnessPredictor(input_size=5).to(device)
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        epochs=150, learning_rate=0.001, device=device
    )
    
    print("\n" + "=" * 60)
    print("STEP 3: Evaluating on training data")
    print("=" * 60)
    
    # Evaluate on full training set
    y_pred_train = predict_wetness(model, X_train, scaler_X, scaler_y, device)
    y_train_denorm = scaler_y.inverse_transform(y_train)
    
    mse = np.mean((y_pred_train - y_train_denorm.flatten())**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred_train - y_train_denorm.flatten()))
    
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"MSE: {mse:.6f}\n")
    
    # ========== STEP 4: Make predictions on new data ==========
    print("=" * 60)
    print("STEP 4: Making predictions on new data")
    print("=" * 60)
    
    # Load prediction data (without Wetness column)
    pred_data = pd.read_csv('prediction_data.csv')
    
    # Extract features in same order as training
    feature_cols = ['Intensity', 'Temperature', 'FruitID', 'ClusterID', 'X']
    X_pred = pred_data[feature_cols].values
    
    # Make predictions
    wetness_predictions = predict_wetness(model, X_pred, scaler_X, scaler_y, device)
    
    # Create output dataframe
    output_df = pred_data.copy()
    output_df['Wetness_Predicted'] = wetness_predictions
    
    # Save predictions
    output_df.to_csv('predictions_output.csv', index=False)
    
    print(f"\nPredictions made for {len(output_df)} samples")
    print("Results saved to 'predictions_output.csv'\n")
    print(output_df.head(10))
    
    # Save model
    torch.save(model.state_dict(), 'wetness_model.pth')
    print("\nModel saved as 'wetness_model.pth'")