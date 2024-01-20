import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import random


## Load train data
train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

## Load test data
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

## Visualise some data
#print(len(train_data)) # 60000
#print(len(test_data)) # 10000
image, label = train_data[0] # get the first sample and split it into the numerical values and the label
#print(image.shape) # torch.Size([1, 28, 28])

class_names = train_data.classes
#print(class_names) # ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

fig = plt.figure(figsize=(6,6)) # set the size of the plot
rows, columns = 4, 4
for i in range(1, rows * columns + 1): # iterate rows*columns times to get 16 items
    random_idx = torch.randint(0, len(train_data), size=[1]).item() # randomly get one index from the training set
    img, lbl = train_data[random_idx] # get the image and label from a random index
    fig.add_subplot(rows, columns, i) # add subplots with given number of rows and columns and with index i
    plt.imshow(img.squeeze(), cmap="gray") # squeeze() the image to get rid of the "colour" dimension
    plt.title(class_names[lbl])
    plt.axis(False)

## Create dataloaders
BATCH_SIZE = 32 # divide the data into batches to facilitate learning
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

## Create a CNN model
class Numbers_Model(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*49, # we get this number from flattening the data after it passes through the convolutional blocks
                      out_features=output_shape)
        )
    
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
    
model = Numbers_Model(input_shape=1, # number of colours in the input data
                      hidden_units=10,
                      output_shape=10) # equal to len(class_names)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

# to quickly get the number of correct in_features in the nn.Linear layer we create a sample and pass it into the model 
# and then copy the value from the RuntimeError
#train_features_batch, train_labels_batch = next(iter(train_dataloader))
#model(train_features_batch[0])

## Train and test the model
def train(model: torch.nn.Module,
          data_loader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer):
    train_loss = 0
    for batch, (X,y) in enumerate(train_dataloader):
        model.train()
        y_predictions = model(X)
        loss = loss_fn(y_predictions, y)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")

    train_loss /= len(train_dataloader)
    print(f"\nTrain loss: {train_loss:.4f}")

def test(model: torch.nn.Module,
          data_loader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer):
    test_loss = 0
    model.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            test_predictions = model(X)
            test_loss += loss_fn(test_predictions, y)
        test_loss /= len(test_dataloader)
        print(f"\nTest loss: {test_loss:.4f}")

epochs = 4
def work():
    for epoch in range(epochs):
        print(f"Epoch: {epoch}.")
        train(model=model,
              data_loader=train_dataloader,
              loss_fn=loss_fn,
              optimizer=optimizer)
        test(model=model,
             data_loader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer)
        
work()

## Visualise some predictions
def make_predictions(model: torch.nn.Module,
                     data: list):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0)

            # forward
            pred_logit = model(sample)

            # probability
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            pred_probs.append(pred_prob)

    return torch.stack(pred_probs)

random.seed(42)
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=16): # randomly choose 16 samples from test_data
    test_samples.append(sample)
    test_labels.append(label)

pred_probs = make_predictions(model=model, data=test_samples)
pred_classes = pred_probs.argmax(dim=1) #probabilities to labels

plt.figure(figsize=(9, 9))
nrows = 4
ncols = 4
for i, sample in enumerate(test_samples):
    plt.subplot(nrows, ncols, i+1)
    plt.imshow(sample.squeeze(), cmap="gray")
    pred_label = class_names[pred_classes[i]]
    truth_label = class_names[test_labels[i]]
    title = f"Pred: {pred_label} | Truth: {truth_label}"

    if pred_label == truth_label:
        plt.title(title, fontsize=10, c="g")
    else:
        plt.title(title, fontsize=10, c="r")

    plt.axis(False)


## Create a confusion matrix
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

y_preds = []
model.eval()
with torch.inference_mode():
  for X, y in test_dataloader:
    y_logit = model(X)
    y_pred = torch.softmax(y_logit.squeeze(), dim=0).argmax(dim=1)
    y_preds.append(y_pred)
y_pred_tensor = torch.cat(y_preds)

confmat = ConfusionMatrix(task="multiclass", 
                          num_classes=len(class_names))
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets) #etykiety celów (1,2,3 itd.)

fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(),
                                class_names=class_names,
                                figsize=(10,7))