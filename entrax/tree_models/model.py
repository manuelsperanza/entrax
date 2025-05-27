import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report  # type: ignore
from typing import Sequence
import numpy.typing as npt
from entrax.utils.log import write_info
from pathlib import Path
from entrax.utils.IO import DefaultPaths, get_available_filepath
from abc import ABC, abstractmethod
import numpy.typing as npt
from typing import Any
from datetime import datetime 
from entrax.utils.log import setup_logger, write_info

class Model(ABC):
    def __init__(self, additional_message: str = "", info:int=0) -> None:
        now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        log_filename = f"log_{self.__class__.__name__}_{now}.txt"
        setup_logger(owner=self, log_filename=log_filename, level=info)
        message = f"{self.__class__.__name__} started:\n"
        message += additional_message
        write_info(owner=self, info=message)

    @abstractmethod
    def predict(self, X:pd.DataFrame) -> npt.NDArray[Any]: ...

    def test_predictions(
        self,
        X: pd.DataFrame,
        y: pd.Series,  # type: ignore
        matrix_path: Path = DefaultPaths.IMAGES.value / "confusionMatrix.png",      # now a Path
    ) -> None:
        pred = self.predict(X)
        acc = accuracy_score(y, pred)
        cm = confusion_matrix(y, pred)
        report = classification_report(y, pred)

        # Build the full evaluation string using the info level
        report_str = "--- Model Evaluation ---\n"
        report_str += f"Accuracy: {acc:.4f}\n\n"
        report_str += "Confusion Matrix:\n"
        report_str += f"{cm}\n\n"
        report_str += "Classification Report:\n"
        report_str += report
        write_info(owner=self, info=report_str)

        # Prepare labels for heatmap
        labels: Sequence[str] = list(map(str, sorted(y.unique().tolist())))

        # ensure output folder exists
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        # avoid overwriting existing images
        save_path = get_available_filepath(matrix_path)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()