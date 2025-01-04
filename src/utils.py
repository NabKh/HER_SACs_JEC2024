# -*- coding: utf-8 -*-
"""
Neural Network Approach for HER Activity Prediction
------------------------------------------------

This implementation demonstrates the machine learning approach for predicting 
Hydrogen Evolution Reaction (HER) activity based on material properties.

Author:  Dr. Nabil Khossossi
Email: n.khossossi@tudelft.nl
Institution: TU Delft | Technische Universiteit Delft 
Department:  MSE
Date: September 2024 

Training Parameters:
- Learning Rate: 0.001 (fixed)
- Epochs: 125
- Training Iterations: 60
- Loss Function: Mean Squared Error
- Performance Metric: Relative Deviation
"""

import numpy as np 
import torch 
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
import xlrd
from scipy import optimize

# ============================================================================
#                           Network Components
# ============================================================================

class Conv1d_same_padding(torch.nn.Module):
    """1D convolution layer with same padding."""
    def __init__(self, inplanes, planes, kernel_size, strides=1, dilation=1):
        super(Conv1d_same_padding, self).__init__() 
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation = dilation
        self.conv1d = torch.nn.Conv1d(inplanes, planes, kernel_size, strides, bias=False) 
        torch.nn.init.xavier_uniform_(self.conv1d.weight)

    def forward(self, x): 
        """Forward pass with same padding."""
        input_rows = x.size(2)
        out_rows = (input_rows + self.strides - 1) // self.strides
        padding_rows = max(0, (out_rows - 1) * self.strides + 
                         (self.kernel_size - 1) * self.dilation + 1 - input_rows) 
        x = F.pad(x, pad=(0, padding_rows), mode="constant") 
        return self.conv1d(x)

class Net(torch.nn.Module):
    """Neural network model for HER prediction."""
    def __init__(self): 
        super(Net, self).__init__() 
        X_data, _, _ = load_data()
        self.register_buffer('data', X_data)
        
        # Convolutional layers
        self.conv_0 = Conv1d_same_padding(4, 16, 1)
        self.conv_1 = Conv1d_same_padding(16, 16, 1)
        self.conv_2 = Conv1d_same_padding(16, 16, 1)
        self.conv_3 = Conv1d_same_padding(16, 16, 1)
        self.conv_4 = Conv1d_same_padding(16, 8, 1)
        
        # Dense layers
        self.dense_0 = torch.nn.Linear(40, 32, bias=False)
        self.dense_1 = torch.nn.Linear(32, 16, bias=False)
        self.dense_2 = torch.nn.Linear(16, 1, bias=False)
        
        # Initialize weights
        for layer in [self.dense_0, self.dense_1, self.dense_2]:
            torch.nn.init.xavier_uniform_(layer.weight)
        
    def forward(self, x):
        """Forward pass through the network."""
        # Reference samples (modify indices based on your reference samples)
        label_0 = self.data[3,:].reshape(1,1,5).float()
        label_1 = self.data[4,:].reshape(1,1,5).float()
        label_2 = self.data[13,:].reshape(1,1,5).float()
        net = torch.cat([x, label_0, label_1, label_2], dim=1)
        
        # Convolutional layers with batch normalization
        for conv in [self.conv_0, self.conv_1, self.conv_2, self.conv_3, self.conv_4]:
            net = F.relu(conv(net))
            
        # Dense layers with dropout
        net = torch.flatten(net)
        net = F.dropout(F.relu(self.dense_0(net)), p=0.2, training=self.training)
        net = F.dropout(F.relu(self.dense_1(net)), p=0.2, training=self.training)
        return self.dense_2(net)

# ============================================================================
#                           Error Metrics
# ============================================================================

class RelativeError(torch.nn.Module):
    """Relative deviation metric for model evaluation."""
    def __init__(self):
        super(RelativeError, self).__init__()
        
    def forward(self, pred, real):
        """Calculate relative deviation between predictions and real values."""
        return torch.mean(torch.abs(torch.sub(pred, real) / real))

# ============================================================================
#                           Data Loading
# ============================================================================

def load_data(file="./data/raw/base_data.xlsx"):
    """
    Load and process data from Excel file.
    
    Args:
        file (str): Path to Excel file containing the data
        
    Returns:
        tuple: Processed features (X_data), targets (y_data), and raw data lists
    """
    wb = xlrd.open_workbook(file)
    sheet = wb.sheet_by_index(0)
    
    # Number of elements (modify according to your dataset)
    num_elements = 29
    
    # Load features
    electronegativity = np.array([sheet.cell_value(i+1,1) for i in range(num_elements)])
    d_orbital_of_metal = np.array([sheet.cell_value(i+1,2) for i in range(num_elements)])
    group = np.array([sheet.cell_value(i+1,3) for i in range(num_elements)])
    radius_pm = np.array([sheet.cell_value(i+1,4) for i in range(num_elements)])
    first_ionization_energy = np.array([sheet.cell_value(i+1,5) for i in range(num_elements)])
    HER_activity = np.array([sheet.cell_value(i+1,6) for i in range(14)])  # Adjust range if needed
    
    # Stack features
    X_data = np.stack((electronegativity, d_orbital_of_metal,
                      group, radius_pm,
                      first_ionization_energy), axis=0).T
    X_data = torch.from_numpy(X_data).float()
    y_data = torch.from_numpy(HER_activity).float()
    
    return X_data, y_data, [electronegativity, d_orbital_of_metal,
                           group, radius_pm, first_ionization_energy, y_data]

def build_dataset():
    """
    Build training, testing, and prediction datasets.
    
    Returns:
        tuple: Training, testing, and prediction data
    """
    X_data, y_data, _ = load_data()
    
    # Define data splits (modify according to your needs)
    train_list = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    test_list = [4, 12]
    predict_list = list(range(18, 29))  # Adjust range if needed
    
    # Create datasets
    X_train = [X_data[i,:].reshape(1,1,5).float() for i in train_list]
    y_train = [y_data[i].reshape(1) for i in train_list]
    
    X_test = [X_data[i,:].reshape(1,1,5).float() for i in test_list]
    y_test = [y_data[i].reshape(1) for i in test_list]
    
    X_predict = [X_data[i,:].reshape(1,1,5).float() for i in predict_list]
    
    return X_train, y_train, X_test, y_test, X_predict

# ============================================================================
#                           Training Functions
# ============================================================================

def train_model():
    """
    Train the neural network model.
    
    Parameters:
    - Fixed learning rate: 0.001
    - Epochs: 125
    - Loss function: MSE
    - Performance metric: Relative deviation
    
    Returns:
        tuple: Training history and trained model
    """
    # Load data
    X_train, y_train, X_test, y_test, _ = build_dataset()
    
    # Initialize model and training parameters
    epochs = 125
    learning_rate = 0.001
    model = Net()
    criterion = torch.nn.MSELoss()  # Mean Squared Error loss
    rel_error = RelativeError()     # Relative deviation metric
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training records
    loss_rec_train, loss_rec_test = [], []
    error_rec_train, error_rec_test = [], []
    
    # Training loop
    for epoch in range(epochs):
        # Training
        train_loss, train_error = 0, 0
        model.train()
        for coord, labels in zip(X_train, y_train):
            optimizer.zero_grad()
            outputs = model(coord)
            loss = criterion(outputs, labels)
            error = rel_error(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_error += error.item()
            
        train_loss_mean = train_loss / len(X_train)
        train_error_mean = train_error / len(X_train)
        loss_rec_train.append(train_loss_mean)
        error_rec_train.append(train_error_mean)
        
        # Testing
        test_loss, test_error = 0, 0
        model.eval()
        with torch.no_grad():
            for coord, labels in zip(X_test, y_test):
                outputs = model(coord)
                loss = criterion(outputs, labels)
                error = rel_error(outputs, labels)
                test_loss += loss.item()
                test_error += error.item()
                
        test_loss_mean = test_loss / len(X_test)
        test_error_mean = test_error / len(X_test)
        loss_rec_test.append(test_loss_mean)
        error_rec_test.append(test_error_mean)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss_mean:.4f}, '
                  f'Test Loss: {test_loss_mean:.4f}')
    
    return loss_rec_train, error_rec_train, loss_rec_test, error_rec_test, model

# ============================================================================
#                           Visualization Functions
# ============================================================================

def plot_training_rst(error_rec_train, error_rec_test):
    """Plot training and testing relative errors."""
    plt.figure(figsize=(10,9))
    plt.plot(error_rec_train, "-o", linewidth=3, markersize=8)
    plt.plot(error_rec_test, "-o", linewidth=3, markersize=8)
    plt.ylim([0,1])
    plt.legend(["Train error", "Test error"], fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Relative Error", fontsize=18)
    plt.grid(which="both", linestyle="--", linewidth=2, axis='y')
    plt.show()
