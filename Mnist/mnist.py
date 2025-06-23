import torch
import torchvision
import torchvision.transforms as transforms
# The reason for the normalization is to make the data more stable and easier to train.
# The mean and standard deviation are calculated from the training data.
# The reason for the ToTensor() is to convert the data to a tensor.
# I used only one channel because the MNIST dataset is grayscale.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5))])
batch_size = 4
trainset = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=2)
testset = torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=2)
print(plt.imshow(trainset.train_data[0].numpy(),cmap='gray'))
