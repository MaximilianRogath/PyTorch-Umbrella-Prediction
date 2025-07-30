# PyTorch Deep Learning Tutorial for Beginners

A simple, beginner-friendly introduction to deep learning using PyTorch. This tutorial demonstrates binary classification by predicting whether an umbrella is needed based on weather conditions.

## What You'll Learn

- How to create and structure training data
- Building a simple neural network with PyTorch
- Training a model with backpropagation
- Making predictions with a trained model
- Evaluating model performance

## Requirements

- Python 3.8+
- PyTorch 2.0+

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/MaximilianRogath/PyTorch-Umbrella-Prediction.git
   cd pytorch-umbrella-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the tutorial**
   ```bash
   python umbrella_predictor.py
   ```

## Expected Output

```
Epoch   0/300 - Loss: 6.5131
Epoch  50/300 - Loss: 0.0149
Epoch 100/300 - Loss: 0.0053
Epoch 150/300 - Loss: 0.0039
Epoch 200/300 - Loss: 0.0029
Epoch 250/300 - Loss: 0.0022
Prediction for [20°C, 85% humidity, 10 km/h]: 1.000
Umbrella needed (probability: 100.0%)
Training accuracy: 10/10 = 100.0%
```

## How It Works

The neural network learns to classify weather conditions into two categories:
- **0**: No umbrella needed
- **1**: Umbrella needed

### Model Architecture
- **Input**: 3 features (temperature, humidity, wind speed)
- **Hidden Layer**: 5 neurons with ReLU activation
- **Output**: 1 neuron with Sigmoid activation (probability)

### Training Data
10 weather examples with manually labeled umbrella requirements:
- High humidity + wind → Umbrella needed
- Low humidity + calm weather → No umbrella needed

## Customization

You can modify the following parameters in the code:

- **Training epochs**: Change `range(300)` to train for more/fewer iterations
- **Learning rate**: Modify `lr=0.01` in the optimizer
- **Network size**: Adjust `nn.Linear(3, 5)` to change hidden layer size
- **Training data**: Add more weather examples to the dataset

## Key Concepts Explained

- **Binary Classification**: Predicting one of two possible outcomes
- **Neural Network**: A model inspired by biological neural networks
- **Backpropagation**: How the network learns from its mistakes
- **Loss Function**: Measures how wrong the predictions are
- **Optimizer**: The algorithm that improves the network
