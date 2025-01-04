# Few-Shot Learning in Screening Septuple-Atomic-Layer Supported Single-Atom Catalysts for Hydrogen Production

## Overview
This repository contains the implementation of the machine learning approach described in our paper "Few-shot learning for screening 2D Ga2CoS4- x supported single-atom catalysts for hydrogen production" for predicting Hydrogen Evolution Reaction (HER) activity. The model uses a combination of convolutional and dense neural networks to predict HER activity based on material properties.

## Citation
If you use this code in your research, please cite our paper:
```bibtex
@article{khossossi2025few,
  title={Few-shot learning for screening 2D Ga2CoS4- x supported single-atom catalysts for hydrogen production},
  author={Khossossi, Nabil and Dey, Poulumi},
  journal={Journal of Energy Chemistry},
  volume={100},
  pages={665--673},
  year={2025},
  publisher={Elsevier}
}
```

## Requirements
- Python 3.8+
- PyTorch 1.8+
- NumPy 1.19+
- Matplotlib 3.3+
- SciPy 1.6+
- xlrd 2.0+

## Input Data Format
The model expects an Excel file with the following columns:
1. Electronegativity
2. d-orbital of metal
3. Group number
4. Radius (pm)
5. First ionization energy
6. HER activity (target variable)

## Usage
1. Clone the repository:
```bash
git clone https://github.com/NabKh/HER_SACs_JEC2024.git
cd HER_SACs_JEC2024
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Prepare your data:
   - Place your Excel file in `data/`
   - Ensure it follows the format described above

4. Run the training:
```bash
python src/main.py
```

5. Results will be saved in `results/run_YYYYMMDD_HHMMSS/`:
   - `best_model.pkl`: Trained model state
   - `training_losses.txt`: Training and test losses
   - `training_errors.txt`: Training and test errors
   - `training_history.png`: Learning curves plot
   - `train_predictions.txt`: Training set predictions
   - `test_predictions.txt`: Test set predictions
   - `new_predictions.txt`: Predictions for new elements

## Contributing
We welcome contributions and the use of the script in more large and different datasets! Please feel free to submit a Pull Request.

## Contact
- Dr. Nabil Khossossi
- Email: n.khossossi@tudelft.nl
