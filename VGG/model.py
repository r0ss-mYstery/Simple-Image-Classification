import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Define the data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create the dataset and dataloader
data_dir = "../data/images"
dataset = datasets.ImageFolder(data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Create the data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the VGG model architecture
device = 'cuda'
model = models.vgg16(pretrained=True)
model = model.to(device)

for param in model.parameters():
    param.requires_grad = False
num_features = model.classifier[6].in_features
features = list(model.classifier.children())[:-1]  # Remove last layer
features.extend([nn.Linear(num_features, len(dataset.classes))])
model.classifier = nn.Sequential(*features)
model.classifier = model.classifier.to(device)

# Initialize the model, loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)

# Train the model
print('training started!')
num_epochs = 10
train_loss_history = []
val_loss_history = []
for epoch in range(num_epochs):
    running_train_loss = 0.0
    running_val_loss = 0.0
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    # Evaluate the model on validation set
    model.eval()
    with torch.no_grad():
        for j, (val_inputs, val_labels) in enumerate(val_loader):
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_labels)
            running_val_loss += val_loss.item()

    # Store the training and validation loss
    train_loss_history.append(running_train_loss / len(train_loader))
    val_loss_history.append(running_val_loss / len(val_loader))

    print(f"Epoch {epoch + 1}: Train Loss = {train_loss_history[-1]}, Val Loss = {val_loss_history[-1]}")

print('Finished Training!')

# Evaluate the model on test set
correct = 0
total = 0
predictions = []
true_labels = []
model.eval()
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Compute the metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, average='macro')
recall = recall_score(true_labels, predictions, average='macro')
f1 = f1_score(true_labels, predictions, average='macro')
confusion = confusion_matrix(true_labels, predictions)

print('Accuracy: {:.2f}%'.format(100 * accuracy))
print('Precision: {:.2f}%'.format(100 * precision))
print('Recall: {:.2f}%'.format(100 * recall))
print('F1: {:.2f}%'.format(100 * f1))
print('Confusion Matrix:\n', confusion)
print('training succeeded!')


