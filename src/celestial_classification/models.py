import torch

import torch.nn as nn
import torch.optim as optim

import numpy as np

from torch.utils.data import DataLoader, TensorDataset, Subset

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_predict, KFold

from xgboost import XGBClassifier

from celestial_classification import config


class Model():
    
    def __init__(
        self,
        model_name: str,
        x,
        y
    ):

        model_dict = {
            "rf": RandomForestClassifier,
            "logr": LogisticRegression,
            "gnb": GaussianNB,
            "dt": DecisionTreeClassifier,
            "knn": KNeighborsClassifier,
            "xgb": XGBClassifier,
            "mlp": TorchMLP
    }

        self.model_name = model_name

        self.kfold = KFold(**config.training.kfold_parameters)

        self.x = x
        self.y = y

        model = model_dict[model_name]

        if model_name == "mlp":
            self.model = model(
                input_size = self.x.shape[1],
                output_size = np.unique(self.y).shape[0],
                kf = self.kfold,
                **config.training.hyperparameters[model_name]
            )
        else:
            self.model = model(**config.training.hyperparameters[model_name])
        

    def getTrainPreds(self):
        
        if self.model_name == "mlp":
            predictions = self.model.cross_val_predict(
                torch.tensor(self.x, dtype=torch.float32), 
                torch.tensor(self.y, dtype=torch.long)
            ).cpu().tolist()

        else:
            predictions = cross_val_predict(self.model, self.x, self.y, cv=self.kfold, method="predict_proba").tolist()

        return predictions
    

class TorchMLP(nn.Module):

    def __init__(
            self, 
            input_size, 
            hidden_size, 
            output_size, 
            kf, 
            n_epochs,
            lr, 
            batch_size=16, 
            shuffle_dataset = False
    ):
        
        super(TorchMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

        self.n_epochs = n_epochs
        self.kf = kf
        self.batch_size = batch_size
        self.shuffle_dataset = shuffle_dataset
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr) # Can make this variable if necessary
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
    
    def setDataset(self, X, y):
        self.dataset = TensorDataset(X, y)
    
    def getLoader(self, data):
        return DataLoader(data, batch_size=self.batch_size, shuffle=self.shuffle_dataset)
    
    def fit(self, loader):
        for epoch in range(self.n_epochs):
            for inputs, targets in loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Should replace prints with log.info - tqdm as well? 
            print(f"Epoch [{epoch+1}/{self.n_epochs}], loss: {loss.item():.4f}")

    def cross_val_predict(self, X, y):
        self.setDataset(X, y)

        n_samples = len(self.dataset)
        n_classes = self.model[-1].out_features

        # Preallocate tensor for probabilities
        all_preds = torch.zeros((n_samples, n_classes))

        for fold, (train_idx, val_idx) in enumerate(self.kf.split(self.dataset)):
            train_loader = self.getLoader(Subset(self.dataset, train_idx))
            val_loader = self.getLoader(Subset(self.dataset, val_idx))

            # Train on training set
            self.model.train()
            self.fit(train_loader)

            self.model.eval()
            probs = []

            with torch.no_grad():
                for xb, yb in val_loader:
                    outputs = self.model(xb)
                    prob = torch.softmax(outputs, dim=1)
                    probs.append(prob)

            probs = torch.cat(probs, dim=0)
            val_idx = torch.tensor(val_idx, dtype=torch.long)

            all_preds[val_idx] = probs  # Fill in the proper rows with softmaxed probabilities

        return all_preds