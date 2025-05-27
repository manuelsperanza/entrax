import pandas as pd
import numpy as np
from graphviz import Digraph # type: ignore
from typing import Any, cast
from abc import abstractmethod
from time import perf_counter
import numpy.typing as npt
from entrax.utils.log import setup_logger, write_debug_message
from datetime import datetime 
from entrax.tree_models.model import Model
from entrax.utils.general import format_elapsed_time

class CartTemplate(Model):
    _counter = 0
    def __init__(self, class_column_name: str,
                 max_depht: int | None = None,  # allow None for unlimited
                 gamma: float = 0,
                 info: int = 0):
        self.id = f"node_{self.__class__._counter}"
        self.__class__._counter+=1
        self.isleaf = False  # Flag to indicate if the node is a leaf
        self.node_name = ""
        self.predict_value:Any = None # None 
        self.left_node: CartTemplate | None = None
        self.right_node: CartTemplate | None = None
        #for not leaf
        self.node_gain = -np.inf
        self.node_split = 0.0
        self.class_column_name = class_column_name
        self.max_depht = max_depht
        self.gamma = gamma

        super().__init__(info=info)

    def _train(self, df: pd.DataFrame) -> None:
        start = perf_counter()
        num_unique = df[self.class_column_name].nunique()
        # stop conditions: only check depth if max_depht is set
        if num_unique == 1 \
           or (self.max_depht is not None and self.max_depht == 0) \
           or df.shape[1] == 1:
            self._make_leaf(df)
        else:
            # Iterate through columns to find the best split
            for column in df.columns:
                if column == self.class_column_name:  # Skip the class column
                    continue
                t, gain = self.find_best_split(df, column)  # Find the best threshold and Gini index
                if gain > self.node_gain:  # Update the best split if a lower Gini index is found
                    self.node_gain = gain
                    self.node_name = column
                    self.node_split = t

            if self.node_gain <= self.gamma:
                self._make_leaf(df)
            
            else:
                self._make_next_nodes(df)
        write_debug_message(owner=self, debug_message=f"{format_elapsed_time(perf_counter() - start)}")

    @abstractmethod
    def _make_next_nodes(self, df:pd.DataFrame, )->None: ...
    
    @abstractmethod
    def _make_leaf(self, df:pd.DataFrame)->None: ...

    def __str__(self) -> str:
        return f"{self.node_name}"
    
    def _visualize_recursively(self, dot:Digraph, parent:str='', arrow_label:str='')->None:
        #dot.node('X', 'Fancy Root', shape='diamond', style='filled', fillcolor='lightcoral', fontcolor='white')
        dot.node(self.id, str(self))

        if parent:
            dot.edge(parent, self.id, label=arrow_label)

        if not self.isleaf:
            if self.left_node != None:
                self.left_node._visualize_recursively(dot, self.id, f"≤ {round(self.node_split,2)}")
            if self.right_node != None:
                self.right_node._visualize_recursively(dot, self.id, f"> {round(self.node_split,2)}")
    
    def visualize(self, title:str, view:bool=False)->None:
        dot = Digraph()
        self._visualize_recursively(dot)
        dot.render(title, view=view, format='png')

    @abstractmethod
    def compute_split_gain(self, df: pd.DataFrame, t: float, column: str)->float: ...
    
    def find_best_split(self, df: pd.DataFrame, column: str) -> tuple[float, float]:
        # Sort the DataFrame by the column to find potential split thresholds
        thresholds = df[column].sort_values().to_numpy()
        thresholds = (thresholds[:-1] + thresholds[1:]) / 2  # Calculate midpoints between consecutive values

        # Initialize variables to track the best split
        best_gain = -np.inf
        best_split = thresholds[0]

        # Iterate through thresholds to find the best split
        for t in thresholds:
            gain = self.compute_split_gain(df, t, column)
            if gain > best_gain:  # Update the best split if a lower Gini index is found
                best_gain = gain
                best_split = t

        # Return the best threshold and its Gini index
        return best_split, best_gain          

    def predict(self, X: pd.DataFrame) -> npt.NDArray[Any]:
        # mypy sees .to_numpy() as Any, so we cast it to the declared ndarray type
        any_results = X.apply(self._predict_recursive, axis=1).to_numpy()
        return cast(npt.NDArray[np.float64], any_results)

    def _predict_recursive(self, row: pd.DataFrame) -> Any:
        # Base case: If the node is a leaf return the prediction
        if self.isleaf:
            return self.predict_value

        # ensure children exist before recursing
        assert self.left_node is not None and self.right_node is not None

        if row[self.node_name] <= self.node_split:
            return self.left_node._predict_recursive(row)
        else:
            return self.right_node._predict_recursive(row)

class Cart(CartTemplate):
    def __init__(self, df: pd.DataFrame, class_column_name: str,
                 max_depht: int | None = None,  # default None
                 gamma: float = 0.0,
                 info: int = 0):
        super().__init__(class_column_name, max_depht,
                         gamma=gamma, info=info)
        super()._train(df)

    def _make_next_nodes(self, df: pd.DataFrame) -> None:
        split = df[self.node_name] <= self.node_split
        left_df = df[split]
        right_df = df[~split]
        if len(left_df) == 0 or len(right_df) == 0:
            self._make_leaf(df)
        else:
            # compute child depth
            next_depth = None if self.max_depht is None else self.max_depht - 1
            self.left_node = self.__class__(left_df, self.class_column_name, max_depht=next_depth)
            self.right_node = self.__class__(right_df, self.class_column_name, max_depht=next_depth)

    def _make_leaf(self, df: pd.DataFrame) -> None:
        self.isleaf = True
        proportions = df[self.class_column_name].value_counts(normalize=True)
        # idxmax() might be int or str, cast node_name → str
        self.node_name = str(proportions.idxmax())
        self.predict_value = proportions.idxmax()
 
    def gini(self, df: pd.DataFrame, class_column: str) -> np.float64:
        # Calculate the Gini impurity for the given DataFrame and class column
        proportions = df[class_column].value_counts(normalize=True).to_numpy()
        proportions = np.square(proportions)
        return np.float64((1 - proportions.sum()))
    
    def compute_split_gain(self, df: pd.DataFrame, t: float, column: str) -> float:
        # Split the DataFrame into left and right subsets based on the threshold
        ldf = df.loc[df[column] <= t]
        rdf = df.loc[df[column] > t]

        # Calculate the sizes of the subsets
        Nl = len(ldf)
        Nr = len(rdf)
        N = Nl + Nr

        # Calculate the Gini impurity for each subset
        gl = self.gini(ldf, self.class_column_name)
        gr = self.gini(rdf, self.class_column_name)

        # Return the weighted Gini impurity of the split
        gini_split = (Nl / N) * gl + (Nr / N) * gr
        
        parent_impurity = self.gini(df, self.class_column_name)
        return float(parent_impurity - gini_split)       
     
class CartBoost(CartTemplate):
    def __init__(self, df: pd.DataFrame, class_column_name: str,
                 predictions: npt.NDArray[np.float64],
                 max_depht: int | None = None,  # default None
                 gamma: float = 0.0,
                 learning_rate: float = 0.3,
                 info: int = 0):
        super().__init__(class_column_name, max_depht,
                         gamma=gamma, info=info)
# store for children
        self.learning_rate = learning_rate
        y = df[self.class_column_name].to_numpy()
        self.predictions = predictions
        self.gradients = self.predictions - y
        self.hessians = self.predictions * (1-self.predictions)              

        # Calculate the sizes of the subsets
        self.G = self.gradients.sum()
        self.H = self.hessians.sum()

        self.l2_reg = 1
        self.part3 = (self.G**2)/(self.H+self.l2_reg)
        super()._train(df)
    
    def _make_next_nodes(self, df:pd.DataFrame)->None:
        split = df[self.node_name] <= self.node_split
        left_df = df[split]
        right_df = df[~split]
        left_predictions = self.predictions[split]
        right_predictions = self.predictions[~split]
        if len(left_df) == 0 or len(right_df) == 0:
            self._make_leaf(df)
        else:
            next_depth = None if self.max_depht is None else self.max_depht - 1
            self.left_node = CartBoost(left_df, self.class_column_name,
                                       left_predictions, max_depht=next_depth,
                                       gamma=self.gamma,
                                       learning_rate=self.learning_rate)
            self.right_node = CartBoost(right_df, self.class_column_name,
                                        right_predictions, max_depht=next_depth,
                                        gamma=self.gamma,
                                        learning_rate=self.learning_rate)

    def _make_leaf(self, df:pd.DataFrame)->None:
        self.isleaf = True
        w_opt = - self.G / (self.H + self.l2_reg)   # optimal weight
        self.predict_value = self.learning_rate * w_opt
        self.node_name = f"{self.predict_value}"
    
    def compute_split_gain(self, df: pd.DataFrame, t: float, column: str)->float:
        # Split the DataFrame into left and right subsets based on the threshold
        #ldf = df.loc[df[column] <= t]
        #rdf = df.loc[df[column] > t]
        mask = df[column].to_numpy() <= t

        Gl = self.gradients[mask].sum()
        Hl = self.hessians[mask].sum()
        Gr = self.G - Gl
        Hr = self.H - Hl
        
        part1 = (Gl**2)/(Hl+self.l2_reg)
        part2 = (Gr**2)/(Hr+self.l2_reg)
        gain = (1/2) * (part1 + part2 - self.part3)
        # cast numpy scalar to Python float
        return float(gain)