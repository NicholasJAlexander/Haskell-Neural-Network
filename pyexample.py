import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # Input to hidden layer
        self.fc2 = nn.Linear(2, 2)  # Hidden to output layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Initialize the neural network
net = SimpleNN()

# Manually set the weights as in the Haskell example
net.fc1.weight.data = torch.tensor([[0.15, 0.2], [0.25, 0.3]])
net.fc1.bias.data.fill_(0)  # Biases set to 0 for simplicity
net.fc2.weight.data = torch.tensor([[0.4, 0.45], [0.5, 0.55]])
net.fc2.bias.data.fill_(0)

# Define the input vector
input_vector = torch.tensor([0.05, 0.1]).float().unsqueeze(0)  # Add batch dimension

# Define the target output
target_output = torch.tensor([0.01, 0.99]).float().unsqueeze(0)

# Define the optimizer and the loss function
optimizer = optim.SGD(net.parameters(), lr=0.1)  # Learning rate as in Haskell example
criterion = nn.MSELoss()

# Forward propagation
output = net(input_vector)

# Compute loss
loss = criterion(output, target_output)

# Backward propagation
optimizer.zero_grad()  # Clear gradients from the previous step
loss.backward()

# Update weights
optimizer.step()

# Print the updated weights
print("Updated weights for first layer:", net.fc1.weight.data)
print("Updated weights for second layer:", net.fc2.weight.data)
