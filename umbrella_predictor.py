# PyTorch Deep Learning Tutorial for Beginners
# ============================================
# Goal: Train a neural network to predict if an umbrella is needed
# Based on: Temperature, humidity, and wind speed

import torch
import torch.nn as nn
import torch.optim as optim

# Set random seed for reproducible results
torch.manual_seed(42)

# Step 1: Create training data
# ============================
# We need examples for the computer to learn from
# Each example has 3 features (inputs) and 1 label (correct answer)

# Features: [temperature_celsius, humidity_percent, wind_speed_kmh]
X = torch.tensor([
    [25.0, 30.0, 5.0],   # Warm, dry, light wind
    [18.0, 80.0, 10.0],  # Cool, humid, windy
    [22.0, 60.0, 7.0],   # Mild, medium humidity
    [28.0, 40.0, 3.0],   # Very warm, dry, calm
    [15.0, 90.0, 12.0],  # Cold, very humid, very windy
    [30.0, 20.0, 4.0],   # Hot, very dry, light wind
    [19.0, 85.0, 11.0],  # Cool, very humid, windy
    [21.0, 70.0, 9.0],   # Mild, humid, windy
    [17.0, 95.0, 15.0],  # Cold, extremely humid, stormy
    [27.0, 35.0, 2.0],   # Warm, dry, calm
], dtype=torch.float32)

# Labels: 1 = umbrella needed, 0 = no umbrella needed
y = torch.tensor([
    [0],  # 25°C, 30% humidity, 5kmh wind → NO umbrella
    [1],  # 18°C, 80% humidity, 10kmh wind → umbrella NEEDED
    [1],  # 22°C, 60% humidity, 7kmh wind → umbrella NEEDED
    [0],  # 28°C, 40% humidity, 3kmh wind → NO umbrella
    [1],  # 15°C, 90% humidity, 12kmh wind → umbrella NEEDED
    [0],  # 30°C, 20% humidity, 4kmh wind → NO umbrella
    [1],  # 19°C, 85% humidity, 11kmh wind → umbrella NEEDED
    [1],  # 21°C, 70% humidity, 9kmh wind → umbrella NEEDED
    [1],  # 17°C, 95% humidity, 15kmh wind → umbrella NEEDED
    [0]   # 27°C, 35% humidity, 2kmh wind → NO umbrella
], dtype=torch.float32)

# Step 2: Build the neural network
# ================================
# A neural network consists of layers of artificial neurons
# Each neuron takes numbers, processes them, and outputs a number

model = nn.Sequential(
    # First layer: Takes our 3 features and creates 5 internal values
    nn.Linear(3, 5),   # Linear layer connects every input to every output
                       # 3 inputs (temp, humidity, wind) → 5 hidden neurons
    
    # Activation function: Makes the network "smarter"
    nn.ReLU(),         # ReLU = Rectified Linear Unit
                       # Rule: If number is negative → make it 0
                       # If number is positive → keep it unchanged
                       # This helps the network learn complex patterns
    
    # Second layer: From 5 hidden neurons to 1 output
    nn.Linear(5, 1),   # 5 hidden values → 1 final prediction
    
    # Output activation: Converts any number to a probability (0-1)
    nn.Sigmoid()       # Sigmoid converts any number to a value between 0 and 1
                       # 0 = "definitely no umbrella", 1 = "definitely umbrella needed"
                       # 0.7 = "70% probability umbrella needed"
)

# Step 3: Configure learning method
# =================================
# We need two things:
# 1. A way to measure how wrong our predictions are
# 2. A way to improve the network

# Loss function: Measures how wrong our predictions are
criterion = nn.BCELoss()  # BCE = Binary Cross Entropy
                          # Specifically for yes/no decisions (binary classification)
                          # The more wrong the prediction, the higher the "loss"

# Optimizer: The method HOW the network learns
optimizer = optim.Adam(model.parameters(), lr=0.01)
# Adam = a smart learning algorithm (better than simple methods)
# model.parameters() = all weights/parameters of the network to be adjusted
# lr = learning rate = how big steps to take when learning
# 0.01 = small, careful steps (too big = network doesn't learn, too small = takes forever)

# Step 4: Training (the computer learns!)
# =======================================
# Training = showing the network the same examples many times
# Each time it gets a little bit better

# One "epoch" = going through all training data once
for epoch in range(300):
    # Forward pass: Network makes predictions for all examples
    y_pred = model(X)  # The network "thinks" and gives predictions
    
    # Calculate loss: How wrong are the predictions?
    loss = criterion(y_pred, y)  # Compare predictions with correct answers
    
    # Backward pass: The network learns from its mistakes
    optimizer.zero_grad()  # Clear old learning information
    loss.backward()        # Calculate: Which parameters need to change and how?
    optimizer.step()       # Apply the improvements
    
    # Show progress every 50 epochs
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d}/300 - Loss: {loss.item():.4f}")

# Step 5: Test the trained network
# ================================
# Now let's see if the network really learned

# New example: 20°C, 85% humidity, 10 km/h wind
new_input = torch.tensor([[20.0, 85.0, 10.0]])

# The network makes a prediction
with torch.no_grad():  # Tell PyTorch: "We're not training now, just predicting"
    prediction = model(new_input).item()  # .item() converts tensor to regular number

print(f"Prediction for [20°C, 85% humidity, 10 km/h]: {prediction:.3f}")
if prediction > 0.5:
    print(f"Umbrella needed (probability: {prediction*100:.1f}%)")
else:
    print(f"No umbrella needed (probability: {(1-prediction)*100:.1f}%)")

# Evaluate performance on training data
with torch.no_grad():
    train_predictions = model(X)
    correct = 0
    for i in range(len(X)):
        pred_class = 1 if train_predictions[i].item() > 0.5 else 0
        actual_class = int(y[i].item())
        if pred_class == actual_class:
            correct += 1
    
    accuracy = correct / len(X)
    print(f"Training accuracy: {correct}/{len(X)} = {accuracy*100:.1f}%")