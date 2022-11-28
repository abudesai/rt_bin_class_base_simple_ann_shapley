import numpy as np, pandas as pd
import os
import sys
import joblib
import json
import time
import torch as T
from torch.utils.data import Dataset as Dataset2, DataLoader
from torchvision import datasets, transforms

device = "cuda:0" if T.cuda.is_available() else "cpu"
print("Using device: ", device)


model_params_fname = "model_params.save"
model_wts_fname = "model_wts.save"
history_fname = "history.json"

MODEL_NAME = "bin_class_base_simple_ann_pytorch_shapley"


def get_activation(activation):
    if activation == "tanh":
        activation = T.tanh
    elif activation == "relu":
        activation = T.relu
    elif activation == "none":
        activation == T.nn.Identity
    else:
        raise Exception(f"Error: Unrecognized activation type: {activation}")
    return activation


class Net(T.nn.Module):
    def __init__(self, D, K, activation):
        super(Net, self).__init__()
        M1 = max(2, int(D * 1.5))
        M2 = 4
        self.activation = get_activation(activation)
        self.hid1 = T.nn.Linear(D, M1)
        self.hid2 = T.nn.Linear(M1, M2)
        self.oupt = T.nn.Linear(M2, K)

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        x = self.activation(self.hid1(x))
        x = self.activation(self.hid2(x))
        x = self.oupt(x)  # no softmax: CrossEntropyLoss()
        return x

    def get_num_parameters(self):
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp


class Dataset(Dataset2):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        # Get one item from the dataset
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def get_loss(model, device, data_loader, loss_function):
    model.eval()
    loss_total = 0
    with T.no_grad():
        for data in data_loader:
            # inputs,  labels = T.FloatTensor(data[0]).to(device), T.LongTensor(data[1]).to(device)
            inputs = data[0].to(device).float()
            labels = data[1].type(T.LongTensor).to(device)
            # labels = labels.type(T.LongTensor)
            output = model(inputs.view(inputs.shape[0], -1))
            loss = loss_function(output, labels)
            loss_total += loss.item()
    return loss_total / len(data_loader)


def test(device, model, test_loader):
    model.eval()
    total = 0
    correct = 0
    with T.no_grad():
        for data in test_loader:
            input = data[0].to(device)
            label = data[1].to(device)

            output = model(input.view(input.shape[0], -1))
            _, predicted = T.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    print("Accuracy:", correct / total)


class Classifier:
    def __init__(self, D, K, lr=1e-3, activation="relu", **kwargs) -> None:
        self.D = D
        self.K = K
        self.activation = activation
        self.lr = lr

        self.net = Net(D=self.D, K=self.K, activation=self.activation).to(device)
        self.criterion = T.nn.CrossEntropyLoss()
        self.optimizer = T.optim.SGD(self.net.parameters(), lr=self.lr)
        self.print_period = 10

    def fit(
        self,
        train_X,
        train_y,
        valid_X=None,
        valid_y=None,
        batch_size=64,
        epochs=100,
        verbose=0,
    ):

        train_dataset = Dataset(train_X, train_y)
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )

        if valid_X is not None and valid_y is not None:
            valid_dataset = Dataset(valid_X, valid_y)
            valid_loader = DataLoader(
                dataset=valid_dataset, batch_size=batch_size, shuffle=True
            )
        else:
            valid_loader = None

        losses = self._run_training(
            train_loader,
            valid_loader,
            epochs,
            use_early_stopping=True,
            patience=20,
            verbose=verbose,
        )

        return losses

    def _run_training(
        self,
        train_loader,
        valid_loader,
        epochs,
        use_early_stopping=True,
        patience=3,
        verbose=1,
    ):
        best_loss = 1e7
        losses = []
        min_epochs = 1
        for epoch in range(epochs):
            for times, data in enumerate(train_loader):
                inputs, labels = data[0].to(device).float(), data[1].type(
                    T.LongTensor
                ).to(device)
                output = self.net(inputs)
                loss = self.criterion(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            current_loss = loss.item()

            if use_early_stopping:
                # Early stopping
                if valid_loader is not None:
                    current_loss = get_loss(
                        self.net, device, valid_loader, self.criterion
                    )
                losses.append({"epoch": epoch, "loss": current_loss})
                if current_loss < best_loss:
                    trigger_times = 0
                    best_loss = current_loss
                else:
                    trigger_times += 1
                    if trigger_times >= patience and epoch >= min_epochs:
                        if verbose == 1:
                            print("Early stopping!")
                        return losses
            else:
                losses.append({"epoch": epoch, "loss": current_loss})

            # Show progress
            if verbose == 1:
                if epoch % self.print_period == 0 or epoch == epochs - 1:
                    print(
                        f"Epoch: {epoch+1}/{epochs}, loss: {np.round(loss.item(), 5)}"
                    )

        return losses

    def predict(self, X):
        X = T.FloatTensor(X).to(device)
        preds = T.softmax(self.net(X), dim=-1).detach().cpu().numpy()
        return preds

    def summary(self):
        self.net.summary()

    def evaluate(self, x_test, y_test):
        """Evaluate the model and return the loss and metrics"""
        if self.net is not None:
            dataset = Dataset(x_test, y_test)
            data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)
            current_loss = get_loss(self.net, device, data_loader, self.criterion)
            return current_loss

    def save(self, model_path):
        model_params = {
            "D": self.D,
            "K": self.K,
            "lr": self.lr,
            "activation": self.activation,
        }
        joblib.dump(model_params, os.path.join(model_path, model_params_fname))
        T.save(self.net.state_dict(), os.path.join(model_path, model_wts_fname))

    @classmethod
    def load(cls, model_path):
        model_params = joblib.load(os.path.join(model_path, model_params_fname))
        classifier = cls(**model_params)
        classifier.net.load_state_dict(
            T.load(os.path.join(model_path, model_wts_fname))
        )
        return classifier


def save_model(model, model_path):
    model.save(model_path)


def load_model(model_path):
    try:
        model = Classifier.load(model_path)
    except:
        raise Exception(
            f"""Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?"""
        )
    return model


def save_training_history(history, f_path):
    with open(os.path.join(f_path, history_fname), mode="w") as f:
        f.write(json.dumps(history, indent=2))


def get_data_based_model_params(X, y):
    """
    Set any model parameters that are data dependent.
    For example, number of layers or neurons in a neural network as a function of data shape.
    """
    return {"D": X.shape[1], "K": len(set(y))}
