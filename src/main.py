# -*- coding: utf-8 -*-
"""
Main script for HER Activity Prediction
-------------------------------------

This script implements the training and evaluation pipeline for HER activity prediction.
Training Parameters:
- Learning Rate: 0.001 (fixed)
- Epochs: 125
- Training Iterations: 60
- Loss Function: Mean Squared Error
- Performance Metric: Relative Deviation

Author: Dr. Nabil Khossossi
Email: n.khossossi@tudelft.nl
Institution: TU Delft | Technische Universiteit Delft | MSE
Date: September 2024 
"""

import os 
import shutil 
import torch
import numpy as np
from datetime import datetime
from utils import (build_dataset, train_model, plot_training_rst)

def create_run_directory():
    """Create directory for saving results with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f"./results/run_{timestamp}/"
    os.makedirs(save_path, exist_ok=True)
    return save_path

def save_model_predictions(save_path, elements, predictions, prefix=""):
    """Save predictions with element names."""
    with open(f"{save_path}{prefix}predictions.txt", "w") as f:
        f.write("Element\tPredicted_Value\n")
        for element, pred in zip(elements, predictions):
            f.write(f"{element}\t{pred[0]:.6f}\n")

def main():
    # Create save directory
    save_path = create_run_directory()
    print(f"\nResults will be saved in: {save_path}")

    # Load data
    print("\n1. Loading data...")
    X_train, y_train, X_test, y_test, X_predict = build_dataset()
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Prediction samples: {len(X_predict)}")

    # Training iterations
    print("\n2. Starting training iterations...")
    num_iterations = 60
    best_test_error = float('inf')
    best_model = None
    best_train_history = None

    # Define element names ZnZrNbRhCdHfOsHgTcScY
    elements_train = ["Mn", "Ti", "Fe", "V", "Co", "Cr", 
                     "Mo", "Ru", "Ni", "Ag", "Au", "Ta", "Re", "Ir", "W"]
    elements_test = ["Pd", "Pt"]
    elements_predict = ["Zn", "Zr", "Nb", "Rh", "Cd", "Hf", "Os", 
                    "Hg", "Tc", "Sc", "Y"]

    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")
        
        # Train model
        loss_train, error_train, loss_test, error_test, model = train_model()
        
        # Update best model
        if error_test[-1] < best_test_error:
            best_test_error = error_test[-1]
            best_model = model
            best_train_history = (loss_train, error_train, loss_test, error_test)
            print(f"New best model found! Test error: {best_test_error:.6f}")

    print("\n3. Training completed. Generating predictions...")

    # Generate predictions using best model
    best_model.eval()
    with torch.no_grad():
        # Training set predictions
        train_preds = []
        for coord in X_train:
            train_preds.append(best_model(coord).numpy())

        # Test set predictions
        test_preds = []
        for coord in X_test:
            test_preds.append(best_model(coord).numpy())

        # Predictions for new elements
        predictions = []
        for coord in X_predict:
            predictions.append(best_model(coord).numpy())

    # Save results
    print("\n4. Saving results...")
    
    # Save model
    torch.save(best_model.state_dict(), f"{save_path}best_model.pkl")
    
    # Save training history
    loss_train, error_train, loss_test, error_test = best_train_history
    np.savetxt(f"{save_path}training_losses.txt", 
               np.column_stack((loss_train, loss_test)))
    np.savetxt(f"{save_path}training_errors.txt", 
               np.column_stack((error_train, error_test)))

    # Save predictions
    save_model_predictions(save_path, elements_train, train_preds, "train_")
    save_model_predictions(save_path, elements_test, test_preds, "test_")
    save_model_predictions(save_path, elements_predict, predictions, "new_")

    # Generate and save plots
    print("\n5. Generating plots...")
    plot_training_rst(error_train, error_test)
    plt.savefig(f"{save_path}training_history.png", bbox_inches='tight', dpi=300)

    # Print final results
    print("\nFinal Results:")
    print(f"Best test error: {best_test_error:.6f}")
    
    print("\nTest Set Predictions:")
    for element, pred in zip(elements_test, test_preds):
        print(f"{element}: {pred[0]:.6f}")
    
    print("\nNew Elements Predictions:")
    for element, pred in zip(elements_predict, predictions):
        print(f"{element}: {pred[0]:.6f}")

    print(f"\nAll results have been saved in: {save_path}")

if __name__ == "__main__":
    main()
