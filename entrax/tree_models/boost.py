import numpy as np
import numpy.typing as npt
from entrax.tree_models.cart import CartBoost
import pandas as pd
from entrax.utils.log import setup_logger, write_debug_message, write_info
from typing import List,Any
from time import perf_counter
from entrax.utils.general import format_elapsed_time
from entrax.tree_models.model import Model
from entrax.utils.IO import DefaultPaths


class BinaryBoost(Model):
    def __init__(self, df: pd.DataFrame, class_column_name:str,
                 number_of_trees:int = 100, max_depht: int | None = None,
                 gamma: float = 0.0, info:int=0):
        start = perf_counter()
        super().__init__(additional_message=f"number_of_trees: {number_of_trees}\n", info=info)
        #initialize
        y = df[class_column_name].to_numpy()
        f = np.zeros(len(df))

        self.trees:List[CartBoost] = []
        for i in range(number_of_trees):
            tree_start = perf_counter()
            predictions = self.sigmoid(f)
            tree = CartBoost(df, class_column_name, predictions, max_depht, gamma, info=0)
            self.trees.append(tree)
            f += tree.predict(df)
            write_debug_message(owner=self, debug_message=f"round {i+1}  loss = {self.loss(self.sigmoid(f), y):.6f}  {format_elapsed_time(perf_counter() - tree_start)}")    
        write_debug_message(owner=self, debug_message=f"{format_elapsed_time(perf_counter() - start)}")

    def sigmoid(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # avoid overflow
        z = np.clip(z, -500, 500)
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float64)
    
    def loss(self, p: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    def predict(self, X:pd.DataFrame)->npt.NDArray[Any]:

        f = np.zeros(len(X), dtype=np.float64)

        for t in self.trees:
            f += t.predict(X)
        return (self.sigmoid(f) >= 0.5).astype(int)

    def visualize(self, title:str, view:bool=False)->None:
        for i in range(len(self.trees)):
            self.trees[i].visualize(f"{title}_{i}", view)


class MultiBoost(Model):
    def __init__(self, df: pd.DataFrame, class_column_name:str,
                 number_of_trees:int = 100, max_depht: int | None = None,
                 gamma: float = 0.0, info:int=0):
        start = perf_counter()
        super().__init__(additional_message=f"number_of_trees: {number_of_trees}\n", info=info)
        one_hot = pd.get_dummies(df[class_column_name], prefix=class_column_name).astype(int)
        self.class_names = [col.replace(f"{class_column_name}_", "") for col in one_hot.columns]
        k = one_hot.shape[1]
        dataframes=[]
        for column in one_hot.columns:
            df_k = pd.concat([df.drop(columns=[class_column_name]), one_hot[[column]]], axis=1)
            dataframes.append((column,df_k))
        y = one_hot.to_numpy()
        f = np.zeros((len(df), k))
        self.trees:List[List[CartBoost]] = [[] for _ in range(k)]
        for i in range(number_of_trees):
            round_start = perf_counter()
            predictions = self.softmax(f)
            for j in range(k):
                tree = CartBoost(df=dataframes[j][1], class_column_name=dataframes[j][0], predictions=predictions[:,j], max_depht=max_depht, gamma=gamma, info=0)
                self.trees[j].append(tree)
                f[:,j] += tree.predict(dataframes[j][1])
            write_debug_message(owner=self, debug_message=f"round {i+1}  loss = {self.loss(self.softmax(f), y):.6f}  {format_elapsed_time(perf_counter() - round_start)}")

        write_debug_message(owner=self, debug_message=f"{format_elapsed_time(perf_counter() - start)}")
    
    def softmax(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # avoid overflow
        z = z - z.max(axis=1, keepdims=True)          # shift
        exp_z = np.exp(z)
        result = exp_z / exp_z.sum(axis=1, keepdims=True) 
        return np.array(result, dtype=np.float64)

    def loss(self, p: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
        eps = 1e-15
        p = np.clip(p, eps, 1-eps)
        return float(-(y * np.log(p)).sum(axis=1).mean())


    def predict(self, X:pd.DataFrame)->npt.NDArray[np.float64]:
        f = np.zeros((len(X),len(self.trees)))
        for j in range(len(self.trees)):
            for t in self.trees[j]:
                f[:,j] += t.predict(X)
            
        predictions = self.softmax(f)
        predictions_indices = np.argmax(predictions, axis=1)
        return np.array([self.class_names[i] for i in predictions_indices], dtype=np.str_)
    
    def visualize(self, title:str, view:bool=False)->None:
        for i in range(len(self.trees)):
            for j in range(len(self.trees[i])):
                self.trees[i][j].visualize(f"{title}_{i}", view)
