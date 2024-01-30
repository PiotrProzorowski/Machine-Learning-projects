
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("housing.csv")
data.dropna(inplace=True)
data_shuffled = data.sample(n=len(data), random_state=1)
data_shuffled_final = pd.concat([data_shuffled.drop("ocean_proximity", axis=1),
                                pd.get_dummies(data_shuffled["ocean_proximity"])], axis=1)

# Push the median_house_value column to the last place
data_final = data_shuffled_final[['longitude',	'latitude',
                                  'housing_median_age',	'total_rooms',	'total_bedrooms',
                                  'population',	'households',	'median_income',	'<1H OCEAN',	'INLAND',
                                  'ISLAND',	'NEAR BAY',	'NEAR OCEAN', 'median_house_value']]

X = data_final.to_numpy()[:,:-1] # All the columns except the last one
y = data_final.to_numpy()[:, -1] # Just the last column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler

# We manipulate the data to make all values be similar in scale
scaler = StandardScaler().fit(X_train[:, :8])

def preprocessor(x):
  A = np.copy(x)
  A[:, :8] = scaler.transform(A[:, :8])
  return A

X_train = preprocessor(X_train)
X_test = preprocessor(X_test)
# print(pd.DataFrame(X_train_preprocessed).hist())
# print(pd.DataFrame(X_test_preprocessed).hist())

X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

X_train = X_train.type(torch.float)
X_test = X_test.type(torch.float)
y_train = y_train.type(torch.float)
y_test = y_test.type(torch.float)

class HousingModel_1(nn.Module):
  def __init__(self):
    super().__init__()
    self.stack = nn.Sequential(
        nn.Linear(13, 20),
        nn.ReLU(),
        nn.Linear(20, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )

  def forward(self, x):
    x = self.stack(x)
    return x

model_1 = HousingModel_1()

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)
epochs = 3000
for epoch in range(epochs):
  model_1.train()
  y_pred = model_1(X_train).squeeze()
  loss = loss_fn(y_pred, y_train)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()


  model_1.eval()
  with torch.inference_mode():
    test_pred = model_1(X_test).squeeze()
    test_loss = loss_fn(test_pred, y_test)


  if epoch % 200 == 0:
    print(f"Epoch: {epoch}, Loss: {loss}, Test loss: {test_loss}")

