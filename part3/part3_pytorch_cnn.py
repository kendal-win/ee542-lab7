# write lab 7 part 3 here!
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

#Parts 1-3 start
#use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#Resize images to 64x64 and store in tensor
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

#Load images; each subfolder name becomes a class (cat=0, dog=1)
full_data = datasets.ImageFolder("data/train", transform=transform)
print("Classes:", full_data.class_to_idx)
print("Total images:", len(full_data))

#Split 80% train / 20% test
#use constant seed so we can rerun
train_size = int(0.8 * len(full_data))
test_size = len(full_data) - train_size
gen = torch.Generator().manual_seed(0)
train_data, test_data = random_split(full_data, [train_size, test_size], generator=gen)

#DataLoaders batch the images; shuffle only the training set
#use multiple threads (Anchal, you may need to change if in cloud. Blake computer has 6 core)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=6, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=6, pin_memory=True)

images, labels = next(iter(train_loader))
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("First 10 labels:", labels[:10])
#Parts 1-3 end

#part 4 class for model
class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)    #3x64x64  to 16x62x62
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)   #16x31x31 to 32x29x29
        self.fc1 = nn.Linear(32 * 14 * 14, 128)         #6272 to 128
        self.fc2 = nn.Linear(128, 1)                    #128 to 1 output
        nn.init.constant_(self.fc2.bias, 0.5)           #added to fix constant .5 loss after first run -  start outputs above 0 so the final ReLU isn't dead      

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)      #16x31x31
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)      #32x14x14
        x = torch.flatten(x, 1)                         #6272 per image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

#part 5 create model and output, send to GPU
model = CatDogCNN().to(device)
#part 6
#loss func and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#training
epochs = 10
torch.cuda.synchronize()            #make sure the GPU is idle before starting timer
train_start = time.time()

for epoch in range(10):
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad() #zero gradients
        loss = criterion(model(images), labels)
        loss.backward() #calculate gradients
        optimizer.step() #update the weights using the gradients
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

torch.cuda.synchronize()            #wait for the GPU to finish before stopping the timer
train_time = time.time() - train_start
print(f"Training time: {train_time:.2f} s ({train_time/epochs:.2f} s/epoch)")

#part 8 evaluation on test images
model.eval()
correct = 0
torch.cuda.synchronize()
test_start = time.time()

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        outputs = model(images)
        preds = (outputs > 0.5).float()    #above 0.5 = dog, else cat
        correct += (preds == labels).sum().item()

torch.cuda.synchronize()
test_time = time.time() - test_start
accuracy = correct / len(test_data)

print(f"Test accuracy: {accuracy*100:.2f}% ({correct}/{len(test_data)})")
print(f"Test time: {test_time:.2f} s ({len(test_data)/test_time:.0f} images/s)")