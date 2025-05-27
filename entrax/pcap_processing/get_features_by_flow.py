from pathlib import Path
import pandas as pd
from typing import Optional
from datetime import datetime

from entrax.pcap_processing.flow import Flow
from entrax.utils.constants import Classes, Features
from entrax.utils.log import setup_logger, write_set_info
from entrax.utils.data import dataframe_summary

class GetFeaturesByFlow:
    def __init__(self, flows:list[Flow], label:Classes=Classes.NOCLASS, save_features_path:Optional[Path]=None, info:int=0):
        
        self.__flows = flows
        self.__label = label
        
        rows = []

        self.__save_features_path = save_features_path

        self.__columns = [col.value for col in Features] + [Classes.CLASS.value]
        now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        self.__log_filename = f"log_{self.__label}_{now}.txt"
        setup_logger(owner=self, log_filename=self.__log_filename, level=info)
        for flow in self.__flows:
            new_row = {**flow.calculate_features(), Classes.CLASS.value:self.__label.value}
            rows.append(new_row)

        if label != Classes.NOCLASS:
                        
            TrainingSet = pd.DataFrame(rows, columns=self.__columns)
            write_set_info(owner=self, info=dataframe_summary(TrainingSet), head=str(TrainingSet.head()))
        
            if self.__save_features_path is not None:
                if not self.__save_features_path.parent.exists():
                    self.__save_features_path.parent.mkdir(parents=True)

                file_exists = self.__save_features_path.is_file()
                # Append to CSV, write header only if the file does not exist
                TrainingSet.to_csv(self.__save_features_path, mode='a', header=not file_exists, index=False)