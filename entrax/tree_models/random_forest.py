import numpy as np
import numpy.typing as npt
import pandas as pd
from collections import Counter
from entrax.tree_models.cart import Cart
from entrax.utils.data import select_random_columns
from entrax.utils.log import write_debug_message
from time import perf_counter
from entrax.tree_models.model import Model
from entrax.utils.general import format_elapsed_time
from typing import Any

class RandomForest(Model):
    def __init__(self, df: pd.DataFrame, class_column_name:str, number_of_trees:int=100, max_depht: int|None = None, gamma: float = 0.0, info:int=0):
        start = perf_counter()
        super().__init__(additional_message=f"number_of_trees: {number_of_trees}\n", info=info)

        self.number_of_selected_columns = int(np.ceil(np.sqrt(len(df.columns))))
        self.trees:list[Cart] = []
        for t in range(number_of_trees):
            write_debug_message(owner=self, debug_message=f"training tree number {t+1}")
            random_sample = df.sample(n=len(df), replace=True, random_state=42).reset_index(drop=True)
            random_sample = select_random_columns(df, mandatory_col=class_column_name, n_random=self.number_of_selected_columns)
            self.trees.append(Cart(random_sample, class_column_name=class_column_name, max_depht=max_depht, gamma=gamma, info=0))
        write_debug_message(owner=self, debug_message=f"{format_elapsed_time(perf_counter() - start)}")
        
    def predict(self, X:pd.DataFrame)->npt.NDArray[Any]:
        
        # Get all predictions: one list of predictions per tree
        all_predictions = [tree.predict(X) for tree in self.trees]

        # Transpose the list so we group predictions by sample index
        grouped_predictions = list(zip(*all_predictions))

        # For each sample index, compute the majority vote
        majority_votes = []
        for preds in grouped_predictions:
            counts = Counter(preds)
            max_freq = max(counts.values())
            majority = [val for val, freq in counts.items() if freq == max_freq]
            majority_votes.append(majority[0])  # Pick the first in case of tie

        return np.array(majority_votes)
    
    def visualize(self, title:str, view:bool=False)->None:
        for i in range(len(self.trees)):
            self.trees[i].visualize(f"{title}_{i}", view)