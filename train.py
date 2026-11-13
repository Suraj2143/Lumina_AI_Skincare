import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# --- CHANGE THIS LINE ---
# Point this to the folder that contains the "dry" and "oily" folders
data_dir = r"C:\Users\suraj\Desktop\valid" # <-- Put your exact path here!

# 1. Prepare data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Check if folders are found correctly
if not os.path.exists(data_dir):
    print(f"ERROR: The path {data_dir} does not exist. Please check the folder path.")
    exit()

dataset = datasets.ImageFolder(data_dir, transform=transform)
print(f"Found {len(dataset)} images belonging to classes: {dataset.classes}")

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 2. Load a "pre-trained" model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

# Change the last layer to output 2 categories (because you have dry and oily)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(dataset.classes)) 

# 3. Train
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
model.train()

print("Starting training...")
for epoch in range(5): 
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Round {epoch+1} complete. Loss: {loss.item():.4f}")

# 4. Save the model
torch.save(model.state_dict(), "skin_model.pth")
print("SUCCESS: Model saved as skin_model.pth in your project folder!")