import pytest
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # avoid GUI during testing
import matplotlib.pyplot as plt

from entrax.pcap_processing.dataset_analyser import DatasetAnalyser

def test_invalid_csv_path(tmp_path: Path):
    invalid_csv = tmp_path / "invalid.txt"
    invalid_csv.write_text("dummy content")
    with pytest.raises(ValueError):
        DatasetAnalyser(invalid_csv)

def test_valid_csv(tmp_path: Path):
    # Create a minimal CSV file with required columns.
    df = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
        "class": ["A", "B", "A"]
    })
    csv_file = tmp_path / "test.csv"
    df.to_csv(csv_file, index=False)
    
    # Initialize DatasetAnalyser with plotting disabled.
    analyser = DatasetAnalyser(
        csv_file, 
        save_output=False,
        show_dataframe_info=True, 
        show_correlation_matrix=False, 
        show_box_plot=False, 
        show_pca=False, 
        outliers_info=False, 
        show_histograms=False
    )
    analyser.show_dataframe_info()
    plt.close('all')
