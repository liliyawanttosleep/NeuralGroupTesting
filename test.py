import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
import os

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # Adjusted the input size
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.dropout(x, 0.25)
        x = x.view(-1, 64 * 12 * 12)  # Adjusted the shape
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, 0.5)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

# Load dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(2):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}')

# Save the trained model to current directory
save_path = 'your_model.pth'
torch.save(model.state_dict(), save_path)
print(f'Model has been trained and saved to {save_path}')

# Generate test images of handwritten digits from 1 to 9
for digit in range(1, 10):
    img = Image.new('L', (28, 28), 255)
    d = ImageDraw.Draw(img)
    try:
        fnt = ImageFont.truetype('arial.ttf', 15)
    except IOError:
        fnt = ImageFont.load_default()
    d.text((7, 7), str(digit), font=fnt, fill=0)
    img.save(f'handwritten_digit_{digit}.png')

# Test the model
model.load_state_dict(torch.load('your_model.pth'))
model.eval()
transform = transforms.Compose([transforms.ToTensor()])

for digit in range(1, 10):
    img_path = f'handwritten_digit_{digit}.png'
    img = Image.open(img_path).convert('L')
    img = transform(img)
    img = img.unsqueeze(0)
    output = model(img)
    _, predicted = torch.max(output.data, 1)
    print(f'The model predicts that the image {img_path} is a {predicted.item()}')
