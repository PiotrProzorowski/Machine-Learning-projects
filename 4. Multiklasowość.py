### Multiklasowość

import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np


## Tworzenie danych
# Ustalanie hiperparametrów (często WIELIMI LITERAMI)
NUMBER_OF_CLASSES = 4
NUMBER_OF_FEATURES = 2
RANDOM_SEED = 42

# Tworzenie danych
X_blob, y_blob = make_blobs(n_samples=1000, n_features=NUMBER_OF_FEATURES, centers=NUMBER_OF_CLASSES, cluster_std=1.5, random_state=RANDOM_SEED)

# Zmiana danych na tensory
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

# Rodzielenie danych na treningowe i testowe
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob, y_blob, test_size=0.2, random_state=RANDOM_SEED)

# Wizualizacja danych
plt.figure(figsize=(10,7))
plt.scatter(X_blob[:,0], X_blob[:,1], c=y_blob, cmap=plt.cm.RdYlBu)


## Budowanie modelu
class BlobModel_1(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=16):
        super().__init__()
        self.linear_layer_stack = nn.Sequential( # Przechodzi po kolei przez wszystkie warstwy
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features) 
        )
    def forward(self, x):
        return self.linear_layer_stack(x)
 

model_4 = BlobModel_1(input_features=2,
                      output_features=4,
                      hidden_units=8)


# Tworzenie funkcji loss i optimizera
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_4.parameters(), lr=0.1)

## Należy zamienić wyniki obliczeń modelu (logity) na prawdopodobieństwo
## i następnie na wartości y (0,1,2 lub 3)

## Każda wartość z przewidywanych y_pred_probs wskazuje na prawdopodobieństwo,
## że dany punkt należy kolejno do 0 klasy, 1 klasy, 2 klasy lub 3 klasy)
y_logits = model_4(X_blob_test)
y_pred_probs = torch.softmax(y_logits, dim=1)
# print(y_pred_probs[0])

# Zmieniamy prawdopodobieństwo w nazwy (labels)
y_preds = torch.argmax(y_pred_probs, dim=1)
# print(y_preds)

## Tworzymy pętlę szkoleniową i testową
torch.manual_seed(42)
epochs = 500

for epoch in range(epochs):
    model_4.train()
    y_logits = model_4(X_blob_train) # wyniki modelu przypisujemy do zmiennej y_logits
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1) # przepuszczamy logity przez funkcję softmax, która zmienia je na prawdopodobieństwo
                                                        # i następnie znajdujemy najwyższą wartość, żeby model decydował, do którego zbioru danych przynależy dana wartość
    loss = loss_fn(y_logits, y_blob_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits, y_blob_test)

    if epoch % 50 == 0:
        print(f"Loss: {loss:.4f}, Test loss: {test_loss:.4f}")

## Tworzymy przewidywania modelu

model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)

# Zamieniamy logity na prawdopodobieństwo
y_pred_probs = torch.softmax(y_logits, dim=1)

# Zamieniamy prawdopodobieństwo na wynik (opis)
y_preds = torch.argmax(y_pred_probs, dim=1)


def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


## Wyświetlamy wyniki
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_4, X_blob_train, y_blob_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_4, X_blob_test, y_blob_test)


## Ocenianie wyników modelu
# Dokładność (accuracy) - ocenia prawidłowość przyporządkowania,
#   nie sprawdza się przy nierównych zbiorach danych
# Precyzja (precision) - stosunek klas prawdziwie pozytywnych do wszystkich próbek
# Ponowne wywołanie (recall) - stosunek klas prawdziwie pozytywnych
#   do prawdziwie pozytywnych i prawdziwie negatywnych
# F1-score - połączenie precyzji i ponownego wywołania
# Tablica pomyłek (confusion matrix)

from torchmetrics import Accuracy

torchmetric_accuracy = Accuracy(task='multiclass', num_classes=4)
torchmetric_accuracy(y_preds, y_blob_test)



