import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
# The reason for the normalization is to make the data more stable and easier to train.
# The mean and standard deviation are calculated from the training data.
# The reason for the ToTensor() is to convert the data to a tensor.
# I used only one channel because the MNIST dataset is grayscale.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5))])
batch_size = 4
# --training set
trainset = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=0)
# --test set
testset = torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=0)

class NeuralNet(nn.Module):
    def __init__(self,input_size=784,hidden_size=128,num_classes=10):
        super().__init__()
        # --input layer, 784 neurons (28x28 flattened MNIST images)
        self.fc1 = nn.Linear(input_size,hidden_size)
        # --hidden layer, 128 neurons (you can experiment with this)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        # --output layer, 10 neurons (one for each digit 0-9)
        self.fc3 = nn.Linear(hidden_size,num_classes)
        # --dropout for regularization (prevents overfitting)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self,x):
        # --flatten the input image (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.view(-1,784)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

net = NeuralNet()
print(f"Network architecture:\n{net}")
print(f"\nTotal parameters: {sum(p.numel() for p in net.parameters())}")

# --loss function
criterion = nn.CrossEntropyLoss()
# --optimizer
optimizer = torch.optim.Adam(net.parameters(),lr=0.001)

# --train the network
for epoch in range(10):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, data in enumerate(trainloader):
        # --forward pass
        images,labels = data
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # --calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # --print progress every 100 batches
        if i % 100 == 99:
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/100:.3f}')
            running_loss = 0.0
    
    # --print epoch results
    epoch_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1} completed - Accuracy: {epoch_accuracy:.2f}%")

print("Training completed!")

# --function to display digits with predictions
def show_predictions(net, testloader, num_images=8):
    """Display MNIST digits with their predictions"""
    net.eval()  # Set to evaluation mode
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    # Get predictions
    with torch.no_grad():
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        probabilities = F.softmax(outputs, dim=1)
    
    # Create subplot
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle('MNIST Digit Predictions', fontsize=16)
    
    for i in range(min(num_images, len(images))):
        row = i // 4
        col = i % 4
        
        # Denormalize the image for display
        img = images[i].squeeze().numpy()
        img = (img * 0.5) + 0.5  # Reverse normalization
        
        # Display image
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].axis('off')
        
        # Add title with prediction info
        true_label = labels[i].item()
        pred_label = predicted[i].item()
        confidence = probabilities[i][pred_label].item()
        
        color = 'green' if true_label == pred_label else 'red'
        axes[row, col].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}', 
                                color=color, fontsize=10)
    
    plt.tight_layout()
    plt.show()

# --function to calculate detailed accuracy
def evaluate_accuracy(net, testloader):
    """Calculate detailed accuracy metrics"""
    net.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Calculate per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # Print overall accuracy
    print(f'\nOverall Accuracy: {100 * correct / total:.2f}%')
    print(f'Correct: {correct}/{total}')
    
    # Print per-class accuracy
    print('\nPer-class Accuracy:')
    for i in range(10):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f'Digit {i}: {accuracy:.1f}% ({int(class_correct[i])}/{int(class_total[i])})')
    
    return correct / total

# --display some predictions
print("\n" + "="*50)
print("VISUALIZING PREDICTIONS")
print("="*50)
show_predictions(net, testloader)

# --calculate detailed accuracy
print("\n" + "="*50)
print("DETAILED ACCURACY ANALYSIS")
print("="*50)
overall_accuracy = evaluate_accuracy(net, testloader)

# --show confusion matrix for first few batches
def show_confusion_matrix(net, testloader, num_batches=5):
    """Show confusion matrix for predictions"""
    net.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for i, data in enumerate(testloader):
            if i >= num_batches:
                break
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Create confusion matrix
    confusion = np.zeros((10, 10), dtype=int)
    for pred, true in zip(all_predictions, all_labels):
        confusion[true][pred] += 1
    
    # Display confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add text annotations
    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(confusion[i, j]), ha='center', va='center')
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.show()

print("\n" + "="*50)
print("CONFUSION MATRIX")
print("="*50)
show_confusion_matrix(net, testloader)







        




