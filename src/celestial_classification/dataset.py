import math
import ast

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
    )

from imblearn.over_sampling import SMOTE
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from celestial_classification import models
from celestial_classification import config


class Dataset:

    def __init__(
        self,
        cols: list = config.dataset.extract_cols,
        location: str = config.dataset.dir,
        test_size: float = config.dataset.test_size,
        random_seed: int = config.random_seed,
        augmentation: bool = False
    ):

        self.dataset = pd.read_csv(location)
        self.dataset = self.dataset[cols]

        self.setTrainTestSet(random_seed, test_size, augmentation)

    @staticmethod
    def loadFromPredsFile(
        preds_location: str,
        test: bool = False
    ):
        
        dataset = Dataset()

        if test:
            dataset.test_dataset_preds = pd.read_csv(preds_location)
            if dataset.x_test.shape[0] != dataset.test_dataset_preds.shape[0]:
                raise Exception("test data and predictions data do not match")
        else:
            dataset.training_dataset_preds = pd.read_csv(preds_location)
            if dataset.x_train.shape[0] != dataset.training_dataset_preds.shape[0]:
                raise Exception("training data and predictions data do not match")
        
        return dataset
    
    def setTrainTestSet(
        self,
        random_seed,
        test_size,
        augmentation
    ):
        cols = self.dataset.columns
        class_order = self.dataset["class"].value_counts().index.tolist()

        x_train, x_test, y_train, y_test = train_test_split(
            self.dataset.drop(columns=['class']),
            self.dataset[['class']], 
            test_size=test_size,
            random_state=random_seed
        )

        y_train = y_train['class'].apply(lambda x: class_order.index(x)).to_numpy()
        y_test = y_test['class'].apply(lambda x: class_order.index(x)).to_numpy()

        x_train = x_train.to_numpy()
        x_test = x_test.to_numpy()

        if augmentation:
            dataset = pd.DataFrame(np.column_stack((x_train, y_train)), columns = cols)

            if True:
                # TODO: add option to skip/include outlier filter
                clf = LocalOutlierFactor()
                _ = clf.fit_predict(dataset)

                x_score = clf.negative_outlier_factor_
                outlier_score = pd.DataFrame()
                outlier_score["score"] = x_score

                threshold = -1.5                                            
                filter = outlier_score["score"] < threshold
                outlier_index = outlier_score[filter].index.tolist()

                dataset.drop(outlier_index, inplace=True)

                x_train = dataset.drop(['class'], axis = 1)
                y_train = dataset.loc[:,'class'].values

            sm = SMOTE(random_state=random_seed)
            print('Original dataset shape %s' % Counter(y_train))
            x_train, y_train = sm.fit_resample(x_train, y_train)
            x_train, y_train = shuffle(x_train, y_train, random_state=random_seed)
            print('Resampled dataset shape %s' % Counter(y_train))

        scaler = StandardScaler().fit(x_train)

        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        training_dataset = pd.DataFrame(np.column_stack((x_train, y_train)), columns = cols)
        test_dataset = pd.DataFrame(np.column_stack((x_test, y_test)), columns = cols)

        self.class_order = class_order
        self.training_dataset = training_dataset
        self.test_dataset = test_dataset
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def getPreds(
        self,
        models_list: list = config.training.model_list,
        test: bool = False
    ):
        if test:
            df = self.test_dataset.copy()
        else:
            df = self.training_dataset.copy()
        
        classes = range(0, len(self.class_order))

        class_mapping = {cls: i for i, cls in enumerate(classes)}
        df['class_binary'] = df['class'].apply(
            lambda x: [1 if i == class_mapping[x] else 0 for i in range(len(class_mapping.keys()))]
            )

        for model_name in models_list:
            if test:
                model = models.Model(model_name, self.x_test, self.y_test)
                # TODO: get test preds
            else:
                model = models.Model(model_name, self.x_train, self.y_train)
                predictions = model.getTrainPreds()

            df[f"{model_name}_preds"] = predictions

        if test:
            self.test_dataset_preds = df
        else:
            self.training_dataset_preds = df

    def printClassificationReport(
        self,
        model: str,
        test: bool = False
    ):
        if test:
            df = self.test_dataset_preds.copy()
        else:
            df = self.training_dataset_preds.copy()

        class_order = self.class_order

        y_true = df['class_binary']
        if isinstance(y_true.iloc[0], str):
            y_true = y_true.apply(ast.literal_eval)
        y_true = np.vstack(y_true.values).astype(int)
        y_true = y_true.argmax(axis=1)
    
        y_score = df[f"{model}_preds"]
        if isinstance(y_score.iloc[0], str):
            y_score = y_score.apply(ast.literal_eval)
        y_score = np.vstack(y_score.values).astype(float)
        y_pred = y_score.argmax(axis=1)

        y_pred_labels = [class_order[i] for i in y_pred]


        print(f"\nClassification Report for {model}:\n")
        print(classification_report(y_true, y_pred, target_names=class_order))

    def showROCCurves(
        self,
        grid_cols: int = config.plots.rocauc_grid_cols,
        test: bool = False
    ): 

        if test:
            df = self.test_dataset_preds.copy()
        else:
            df = self.training_dataset_preds.copy()

        class_order = self.class_order

        n_classes = len(class_order)

        y_true = df['class_binary']
        if isinstance(y_true.iloc[0], str):
            y_true = y_true.apply(ast.literal_eval)
        y_true = np.vstack(y_true.values).astype(int)
        y_true = np.array(y_true)

        pred_cols = [col for col in df.columns if col.endswith("_preds")]
        classifiers = list(set(col.replace("_preds", '') for col in pred_cols))

        n_classifiers = len(classifiers)
        n_cols = grid_cols
        n_rows = math.ceil(n_classifiers / n_cols)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        axs = axs.flatten()

        for i, clf in enumerate(classifiers):
            ax = axs[i]
            
            y_score_raw = df[f"{clf}_preds"]
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

        for j in range(i+1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()


    def plotConfusionMatrix(
            self,
            model_name: str,
            test: bool = False
    ):
        
        if test:
            df = self.test_dataset_preds.copy()
        else:
            df = self.training_dataset_preds.copy()
        
        classes = self.class_order
        
        y_true = df['class_binary']
        if isinstance(y_true.iloc[0], str):
            y_true = y_true.apply(ast.literal_eval)

        y_true = y_true.apply(lambda x: np.argmax(x)).apply(lambda x: classes[x]).values

        if f"{model_name}_preds" not in df.columns:
            raise ValueError(f"Model '{model_name}' not found")

        y_pred = df[f"{model_name}_preds"]
        if isinstance(y_pred.iloc[0], str):
            y_pred = y_pred.apply(ast.literal_eval)
        y_pred = y_pred.apply(lambda x: np.argmax(x)).apply(lambda x: classes[x]).values

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot()