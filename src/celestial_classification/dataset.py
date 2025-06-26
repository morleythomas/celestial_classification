import math
import ast

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from celestial_classification import models
from celestial_classification import config


class Dataset:
    
    def __init__(
        self,
        cols: list = config.dataset.extract_cols,
        location: str = config.dataset.dir,
        test_size: float = config.dataset.test_size,
        random_seed: int = config.random_seed
    ):

        self.dataset = pd.read_csv(location)
        self.dataset = self.dataset[cols]

        self.setTrainTestSet(random_seed, test_size)

    @staticmethod
    def loadFromValPreds(
        preds_location: str
    ):
        
        dataset = Dataset()
        dataset.training_dataset_preds = pd.read_csv(preds_location)

        if dataset.x_train.shape[0] != dataset.training_dataset_preds.shape[0]:
            raise Exception("training data and predictions data do not match")
        
        return dataset
    
    def setTrainTestSet(
        self,
        random_seed,
        test_size
    ):

        x_train, x_test, y_train, y_test = train_test_split(
            self.dataset.drop(columns=['class']), 
            self.dataset[['class']], 
            test_size=test_size,
            random_state=random_seed
        )

        training_dataset = x_train.copy()
        training_dataset['class'] = y_train['class']
        training_dataset = training_dataset.reset_index(drop=True)

        testing_dataset = x_test.copy()
        testing_dataset['class'] = y_test['class']
        testing_dataset = testing_dataset.reset_index(drop=True)

        x_train = x_train.to_numpy()
        x_test = x_test.to_numpy()

        y_train = y_train['class'].apply(
            lambda x: 0 if x == "GALAXY" else 1 if x == "STAR" else 2
        ).to_numpy()

        y_test = y_test['class'].apply(
            lambda x: 0 if x == "GALAXY" else 1 if x == "STAR" else 2
        ).to_numpy()

        scaler = StandardScaler().fit(x_train)

        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        self.training_dataset = training_dataset
        self.testing_dataset = testing_dataset
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def getValidationPreds(
        self,
        models_list: list = config.training.model_list,
        k_folds: int = config.training.kfold_parameters["n_splits"]
    ):

        training_dataset_preds = self.training_dataset.copy()
        
        classes = training_dataset_preds['class'].value_counts().index
        class_mapping = {cls: i for i, cls in enumerate(classes)}
        training_dataset_preds['class_binary'] = training_dataset_preds['class'].apply(
            lambda x: [1 if i == class_mapping[x] else 0 for i in range(len(class_mapping.keys()))]
            )

        for model_name in models_list:
            model = models.Model(model_name, self.x_train, self.y_train, self.x_test, self.y_test, k_folds)
            predictions = model.getTrainPreds()

            training_dataset_preds[f"{model_name}_preds"] = pd.Series(predictions)

        self.training_dataset_preds = training_dataset_preds

    def printClassificationReport(
        self,
        model: str
    ):
        y_true = self.training_dataset_preds['class']
    
        # Extract predicted probabilities and convert to class labels
        y_score = self.training_dataset_preds[f"{model}_preds"]
        if isinstance(y_score.iloc[0], str):
            y_score = y_score.apply(ast.literal_eval)
        y_score = np.vstack(y_score.values).astype(float)
        y_pred = y_score.argmax(axis=1)

        # Map prediction indices back to class labels
        class_order = self.training_dataset_preds['class'].value_counts().index.tolist()
        y_pred_labels = [class_order[i] for i in y_pred]

        # Print classification report
        print(f"\nClassification Report for {model}:\n")
        print(classification_report(y_true, y_pred_labels, target_names=class_order))

    def showValROCCurves(
        self,
        grid_cols: int = config.plots.rocauc_grid_cols
    ):
        class_order = self.training_dataset_preds['class'].value_counts().index

        n_classes = len(class_order)

        y_true = self.training_dataset_preds['class_binary']
        if isinstance(y_true.iloc[0], str):  # If stored as string
            y_true = y_true.apply(ast.literal_eval)
        y_true = np.vstack(y_true.values).astype(int)
        y_true = np.array(y_true)

        # Identify classifier names by extracting unique prefixes before '_preds'
        pred_cols = [col for col in self.training_dataset_preds.columns if col.endswith("_preds")]
        classifiers = list(set(col.replace("_preds", '') for col in pred_cols))

        n_classifiers = len(classifiers)
        n_cols = grid_cols
        n_rows = math.ceil(n_classifiers / n_cols)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        axs = axs.flatten()

        for i, clf in enumerate(classifiers):
            ax = axs[i]
            
            y_score_raw = self.training_dataset_preds[f"{clf}_preds"]
            if isinstance(y_score_raw.iloc[0], str):
                y_score_parsed = y_score_raw.apply(ast.literal_eval)
            else:
                y_score_parsed = y_score_raw
            y_score = np.vstack(y_score_parsed.values).astype(float)

            for j in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true[:, j], y_score[:, j])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'{class_order[j]} (AUC = {roc_auc:.2f})')

            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_title(f"ROC Curves: {clf}")
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc='lower right')
            ax.grid(True)

        # Remove any extra subplots
        for j in range(i+1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()


