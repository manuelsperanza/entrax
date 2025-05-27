from typing import Any, Sequence
import pandas as pd
from pathlib import Path
import seaborn as sb
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from entrax.utils.log import setup_logger, write_info, write_debug_message
from entrax.utils.numeric import linespace_extra
from entrax.utils.constants import Classes
from entrax.utils.IO import DefaultPaths, get_available_filepath
from entrax.utils.data import load_dataframe

class DatasetAnalyser():
    def __init__(self, dataset_path:Path, save_output:bool=True, show_dataframe_info:bool=False, show_correlation_matrix:bool=False, show_box_plot:bool=False, show_pca:bool=False, outliers_info:bool=True, show_histograms:bool=False):
        setup_logger(owner=self, log_filename=__name__, level=2)
        if not dataset_path.is_file() or dataset_path.suffix != ".csv":
            raise ValueError("Invalid CSV path provided.")

        self.__df = load_dataframe(dataset_path)
        self.__save_output=save_output 
        if show_dataframe_info:
            self.show_dataframe_info()
        
        if show_correlation_matrix:
            self.show_correlation_matrix()
        
        if show_box_plot:
            self.show_box_plot()
        
        if show_pca:
            self.show_pca()

        if outliers_info:
            self.outliers_info()
        
        if show_histograms:
            self.show_histograms()

    def show_dataframe_info(self)->None:
        setup_logger(owner=self, log_filename=__name__, level=2)
        write_info(owner=self, info=f"\n{self.__df.head()}")
        write_info(owner=self, info=f"Loaded dataset shape: {self.__df.shape}")
        write_info(owner=self, info=f"Class distribution in dataset:\n self.__df.value_counts()")

    def show_correlation_matrix(self)->None:
        df = self.__df.drop(Classes.CLASS.value, axis=1) 
        corr_matrix = df.corr().abs()
        plt.figure(figsize=(14, 12))
        sb.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, square=True, cbar=True, annot_kws={'size': 10})
        plt.title('Correlation Matrix', fontsize=16)
        plt.tight_layout()
        if self.__save_output:
            path = get_available_filepath(DefaultPaths.IMAGES.value / "CorrelationMatrix.png")
            if not path.parent.exists():
                path.parent.mkdir(parents=True)
            plt.savefig(path, dpi=500)
        plt.show()

        from entrax.utils.data import drop_highly_correlated
        to_drop = drop_highly_correlated(df)
        df_cleaned = df.drop(columns=to_drop)
        write_info(owner=self, info=f"features to drop: {to_drop}")
        write_info(owner=self, info=f"features to keep: {df_cleaned.columns}")
        corr_matrix = df_cleaned.corr().abs()
        plt.figure(figsize=(14, 12))
        sb.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, square=True, cbar=True, annot_kws={'size': 10})
        plt.title('Correlation Matrix Cleaned', fontsize=16)
        plt.tight_layout()
        if self.__save_output:
            path = get_available_filepath(DefaultPaths.IMAGES.value / "CorrelationMatrixCleaned.png")
            plt.savefig(path, dpi=500)
        plt.show()

    def show_box_plot(self)->None:
        df = self.__df.drop(Classes.CLASS.value, axis=1)
        i = 1
        n = len(df.columns)//5 + 1
        for col in df.columns:
            plt.subplot(n,6,i)
            i+=1
            df.boxplot(column=col)
        
        if self.__save_output:
            path = DefaultPaths.IMAGES.value / "BoxPlot.png"
            plt.savefig(path, dpi=500)
        plt.show()
    
    def show_pca(self)->None:
        from sklearn.decomposition import PCA # type: ignore
        from sklearn.preprocessing import StandardScaler # type: ignore
        from sklearn.tree import DecisionTreeClassifier # type: ignore
        from sklearn.model_selection import train_test_split # type: ignore
        from sklearn.metrics import classification_report, accuracy_score # type: ignore
        from copy import deepcopy
        import matplotlib.pyplot as plt

        scaler = StandardScaler()
        df_features = deepcopy(self.__df.drop(Classes.CLASS.value, axis=1))
        df_target = self.__df[Classes.CLASS.value]

        X_train, X_test, y_train, y_test = train_test_split(
            df_features, df_target, 
            test_size=0.2, 
            random_state=42,
            stratify=df_target
        )

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pca = PCA()
        pca.fit(X_train_scaled)

        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()

        n = 33

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
                marker='o', linestyle='--')
        plt.axvline(x=n, color='r', linestyle='--', label='Selected Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA: Cumulative Explained Variance by Components')
        plt.legend()
        plt.grid()
        plt.show()

        pca_keeped = PCA(n_components=n)
        X_train_pca = pca_keeped.fit_transform(X_train_scaled)
        X_test_pca = pca_keeped.transform(X_test_scaled)

        dt_classifier = DecisionTreeClassifier(random_state=42)
        dt_classifier.fit(X_train_pca, y_train)

        test_predictions = dt_classifier.predict(X_test_pca)

        print("\nTest Results:")
        print(classification_report(y_test, test_predictions))
        print(f"Test Accuracy: {accuracy_score(y_test, test_predictions):.4f}")

        print("\nFeature Importances from PCA Components:")
        for i, importance in enumerate(dt_classifier.feature_importances_):
            print(f"PC{i+1}: {importance:.4f}")
    
    def outliers_info(self)->None:
        classes_counts =self.__df[Classes.CLASS.value].value_counts()
        for col in self.__df:
            if col == Classes.CLASS.value:
                continue
            feature = self.__df[col]
            q1 = feature.quantile(0.25)
            q3 = feature.quantile(0.75)
            iqr = q3 - q1
            below = q1 - 1.5 * iqr
            above = q3 + 1.5 * iqr
                
            outlier_mask = (feature < below) | (feature > above)
            outliers = feature[outlier_mask]
            outlier_indices = self.__df.index[outlier_mask]
            n_outliers = len(outliers)

            outlier_classes = self.__df.loc[outlier_indices, Classes.CLASS.value]
            if outlier_classes.empty:
                classes_outliers_counts = pd.Series(0, index=classes_counts.index)
            else:
                classes_outliers_counts = outlier_classes.value_counts()
                classes_outliers_counts = classes_outliers_counts.reindex(classes_counts.index, fill_value=0)
            
            classes_counts_list = list(classes_counts.items())
            classes_outliers_list = list(classes_outliers_counts.items())

            message = f"\nFor {col}:"
            message += f"  - Q1 (25th percentile): {q1:.2f}\n"
            message += f"  - Q3 (75th percentile): {q3:.2f}\n"
            message += f"  - IQR bounds: [{below:.2f}, {above:.2f}]\n"
            message += f"  - Number of outliers: {n_outliers}\n"
            message += f"  - Outliers distribution by class:\n"
            
            header = ["Class", "Count", "N Outliers", "Ratio on outliers", "Ratio on class"]
            col_widths = [20, 7, 12, 20, 15]
            
            rows = []
            for (class_name, class_count), (_, outlier_count) in zip(classes_counts_list, classes_outliers_list):
                outlier_ratio = (outlier_count/n_outliers)*100 if n_outliers > 0 else 0
                class_ratio = (outlier_count/class_count)*100 if class_count > 0 else 0
                rows.append([
                    class_name,
                    str(class_count),
                    str(outlier_count),
                    f"{outlier_ratio:.2f}%",
                    f"{class_ratio:.2f}%"
                ])
            
            message += "".join(f"{h:<{w}}" for h, w in zip(header, col_widths))
            message += "\n"
            message += "-" * sum(col_widths)
            message += "\n"
            for row in rows:
                message += "".join(f"{cell:<{w}}" for cell, w in zip(row, col_widths))
                message += "\n"
            
            message += ("\n" + "="*80 + "\n")

            write_info(owner=self, info=message)

    def show_histograms(self)->None:

        df = self.__df.drop(Classes.CLASS.value, axis=1)
        result = self.__df[Classes.CLASS.value].value_counts()
        n = 2
        for i in range(0, len(df.columns), n):
            melted_df = pd.melt(df.iloc[:, i:i+n])
            g = sb.FacetGrid(melted_df, col='variable', col_wrap=2, sharex=False, sharey=False, height=4, aspect=1.2)
            
            def plot_with_outliers(data: pd.DataFrame, **kwargs: dict[str, Any]) -> None:
                col = data['variable'].iloc[0]
                feature = data['value']
                
                q1 = feature.quantile(0.25)
                q3 = feature.quantile(0.75)
                iqr = q3 - q1
                below = q1 - 1.5 * iqr
                above = q3 + 1.5 * iqr
                
                outlier_mask = (feature < below) | (feature > above)
                outliers = feature[outlier_mask]
                not_outliers = feature[~outlier_mask]
                n_outliers = len(outliers)
                
                outlier_indices = data.index[outlier_mask]

                p = np.array([feature.min(), feature.max(), below, above])
                p.sort() 
                
                bins = linespace_extra(np.round(p, 1), 1)
                plt.hist(not_outliers, bins=bins, color='skyblue', edgecolor='skyblue', alpha=0.7, label='Data')
                plt.hist(outliers, bins=bins, color='red', edgecolor='red', alpha=0.7, label='Outliers')
                plt.ylim(top=1000)

                plt.axvline(below, color='orange', linestyle='--', linewidth=2, label=f'Lower Bound: {below:.2f}')
                plt.axvline(above, color='green', linestyle='--', linewidth=2, label=f'Upper Bound: {above:.2f}')
                
                plt.title(f'Histogram of {col} (Outliers: {n_outliers})')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.legend()
                            
            g.map_dataframe(plot_with_outliers)
            plt.tight_layout()
            if self.__save_output:
                path = DefaultPaths.IMAGES.value / f"Hist_{i//2}.png"
                plt.savefig(path, dpi=600)
            else:
                plt.show()
            plt.close(g.figure)