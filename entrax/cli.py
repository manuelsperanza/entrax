import argcomplete
from argcomplete.completers import FilesCompleter
from argparse import ArgumentParser, Namespace  
from pathlib import Path
import os
import subprocess
from typing import Any

from entrax.utils.IO import DefaultPaths

class CustomFilesCompleter(FilesCompleter):
    def __call__(self, prefix: Any, **kwargs: Any) -> Any:  # Explicit return type
        # Call parent __call__, ignoring type check if necessary
        completions = super().__call__(prefix, **kwargs)  # type: ignore

        # Ensure completions is a list
        if not isinstance(completions, list):
            return []  # Default to an empty list if it's not iterable

        # Strip the directory prefix if completing inside a directory
        if "/" in prefix:
            dirname = os.path.dirname(prefix)
            completions = [c.replace(dirname + "/", "", 1) for c in completions]

        return completions

def get_flows_by_pcap(args:Namespace)->None:
    from entrax.pcap_processing.get_flows_by_pcap import GetFlowsByPCAP
    GetFlowsByPCAP(pcap_path=args.pcap_path, save_flows_folder_path=args.save_flows, resume=args.resume, info=args.info)

def get_features_by_flows(args:Namespace)->None:
    from entrax.pcap_processing.get_features_by_flow import GetFeaturesByFlow
    from entrax.utils.IO import get_files_recursively
    from entrax.utils.general import get_class
    from entrax.pcap_processing.flow import read_from_csv
    from entrax.utils.constants import Classes
    from collections import defaultdict
    from entrax.utils.IO import get_available_filepath

    flows_paths = get_files_recursively(args.flows_path, ['.csv'])
    flows_dict = defaultdict(list)
    for flow_path in flows_paths:
        label = get_class(flow_path.parent)
        if label == Classes.AUDIO_STREAMING or label == Classes.VPN_AUDIO_STREAMING:
            continue  # Skip audio streaming flows too few samples
        flow = read_from_csv(flow_path)
        if flow.filter():
            flows_dict[label].append(flow)
    if args.save_features is not None:
        save_features_path = get_available_filepath(args.save_features)
    else:
        save_features_path = None
    for label, flows in flows_dict.items():
        GetFeaturesByFlow(flows, label=label, save_features_path=save_features_path, info=args.info)

def get_features_by_pcap(args:Namespace)->None:
    from entrax.pcap_processing.get_flows_by_pcap import GetFlowsByPCAP
    from entrax.pcap_processing.get_features_by_flow import GetFeaturesByFlow
    from entrax.utils.IO import get_available_filepath
    flows_getter = GetFlowsByPCAP(args.pcap_path, save_flows_folder_path=args.save_flows, get_classes=True, resume=args.resume, info=args.info)
    flows_dict = flows_getter.get_flows()
    save_features_path = get_available_filepath(args.save_features)
    for label, flows in flows_dict.items():
        GetFeaturesByFlow(flows=flows, label=label, save_features_path=save_features_path, info=args.info)

def analyse_dataset(args:Namespace)->None:
    from entrax.pcap_processing.dataset_analyser import DatasetAnalyser
    DatasetAnalyser(dataset_path=args.dataset_path, save_output=args.save_output, show_dataframe_info=args.info_dataset, show_correlation_matrix=args.correlation_matrix, show_box_plot=args.box_plot, show_pca=args.pca, outliers_info=args.outliers_info, show_histograms=args.histograms)

def analyse_flows(args:Namespace)->None:
    from entrax.pcap_processing.flows_analyser import FlowsAnalyser
    FlowsAnalyser(flows_path=args.flows_path)

def trees_models(args:Namespace)->None:
    from entrax.utils.data import load_dataframe, clean_and_split, drop_highly_correlated
    from entrax.utils.constants import Classes
    import pandas as pd

    class_column_name = Classes.CLASS.value
    df = load_dataframe(args.dataset_path)
    if args.dataset_test is None:
        train, test = clean_and_split(df, class_column_name=class_column_name, n=args.number_of_samples, select_option=args.select_option, test_size=args.test_size, random_state=args.random_state)
    else:
        train = df
        test = load_dataframe(args.dataset_test)
        # Clean training data and remove same features from test
        y_train = train[class_column_name]
        X_train = train.drop(columns=[class_column_name])
        to_drop = drop_highly_correlated(X_train)
        X_train_clean = X_train.drop(columns=to_drop)
        train = pd.concat([X_train_clean, y_train], axis=1)
        test = test.drop(columns=to_drop)

    if args.cart:
        from entrax.tree_models.cart import Cart
        cart = Cart(train, class_column_name, max_depht=args.max_depht, info=args.info)
        cart.test_predictions(test, test[class_column_name])
        if args.visualize:
            cart.visualize("CART", True)
    if args.random_forest:
        from entrax.tree_models.random_forest import RandomForest
        rf = RandomForest(df=train, class_column_name=class_column_name, number_of_trees=args.tree_number, max_depht=args.max_depht, info=args.info)
        rf.test_predictions(test, test[class_column_name])
        if args.visualize:
            rf.visualize("CART", True)
    if args.boost:
        from entrax.tree_models.boost import BinaryBoost
        b = BinaryBoost(df=train, class_column_name=class_column_name, number_of_trees=args.tree_number, max_depht=args.max_depht, info=args.info)
        b.test_predictions(test, test[class_column_name])
        if args.visualize:
            b.visualize("CART", True)

    if args.multi_boost:
        from entrax.tree_models.boost import MultiBoost
        mb = MultiBoost(df=train, class_column_name=class_column_name, number_of_trees=args.tree_number, max_depht=args.max_depht, info=args.info)
        mb.test_predictions(test, test[class_column_name])
        if args.visualize:
            mb.visualize("CART", True)
    
def tools(args:Namespace)->None:
    if args.add_autocomplete is not None:
        register_autocomplete = "\neval \"$(register-python-argcomplete entrax)\"\n"

        with args.add_autocomplete.open('a') as file:
            file.write(register_autocomplete)
        print(f"Changes saved to {args.add_autocomplete} Run: \n'source {args.add_autocomplete}' to apply.")

    if args.mypy:
        import tempfile

        tempdir = tempfile.gettempdir()
        target = Path(__file__).parent
        subprocess.run(["mypy", target, "--cache-dir", tempdir, "--pretty", "--strict"])
    
    if args.merge is not None:

        import pandas as pd

        from entrax.utils.IO import get_files_recursively
        from entrax.utils.IO import get_available_filepath

        csv_files = get_files_recursively(args.merge, ['.csv'])
        if len(csv_files) == 0:
            print(f"No CSV files found in {args.merge}")
            return
        merged_df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
        
        # save the merged DataFrame to a new CSV file
        path = args.merge
        if not path.is_dir():
            path = path.parent
        filename = get_available_filepath(path / 'merged.csv')
        merged_df.to_csv(filename, index=False)
        print(f"Merged {len(csv_files)} CSV files into {filename}")

def main()->None:
    parser = ArgumentParser(
        description="CLI tools for processing PCAP files and analyzing datasets. For adding bash autocomplete, run 'entrax tools -a [path to .bashrc|activate (for virtualenv)]'."
    )
    
    PCAP_PATH = DefaultPaths.PCAP.value
    FLOWS_PATH = DefaultPaths.FLOWS.value
    FEATURES_PATH = DefaultPaths.FEATURES.value
    
    subparsers = parser.add_subparsers(dest="command", required=True)
        
    get_flows_by_pcap_command = subparsers.add_parser("getFlowsByPcap", help="Generate flows from a PCAP file")
    get_flows_by_pcap_command.add_argument("pcap_path", nargs='?', default=PCAP_PATH, type=Path, help="Path to the PCAP file").completer = CustomFilesCompleter()  # type: ignore
    get_flows_by_pcap_command.add_argument('-sfw', '--save-flows', nargs='?', const=FLOWS_PATH, type=Path, help='Destination folder to save generated flows').completer = CustomFilesCompleter()  # type: ignore
    get_flows_by_pcap_command.add_argument('-r', '--resume', action='store_true', help="Resume from previous analysis if available")
    get_flows_by_pcap_command.add_argument('-i', '--info', nargs='?', const=1, default=0, type=int, help="Level of additional execution information: 0, 1, or 2")
    get_flows_by_pcap_command.set_defaults(func=get_flows_by_pcap)
    
    get_features_by_pcap_command = subparsers.add_parser("getFeaturesByPcap", help="Extract features from PCAP flows")
    get_features_by_pcap_command.add_argument("pcap_path", nargs='?', default=PCAP_PATH, type=Path, help="Path to the PCAP file").completer = CustomFilesCompleter()  # type: ignore
    get_features_by_pcap_command.add_argument('-sfw', '--save-flows', nargs='?', const=FLOWS_PATH, type=Path, help='Folder where flows are saved').completer = CustomFilesCompleter()  # type: ignore
    get_features_by_pcap_command.add_argument('-sft', '--save-features', nargs='?', const=FEATURES_PATH, type=Path, help='Folder to save extracted features').completer = CustomFilesCompleter()  # type: ignore
    get_features_by_pcap_command.add_argument('-r', '--resume', action='store_true', help="Resume extraction if previous results exist")
    get_features_by_pcap_command.add_argument('-i', '--info', nargs='?', const=1, default=0, type=int, help="Set verbosity level for feature extraction")
    get_features_by_pcap_command.set_defaults(func=get_features_by_pcap)
    
    get_features_by_flows_command = subparsers.add_parser("getFeaturesByFlows", help="Extract features from pre-generated flows")
    get_features_by_flows_command.add_argument("flows_path", nargs='?', default=FLOWS_PATH, type=Path, help="Path to folder containing flows CSV files").completer = CustomFilesCompleter()  # type: ignore
    get_features_by_flows_command.add_argument('-sft', '--save-features', nargs='?', const=FEATURES_PATH, type=Path, help='Folder to save extracted features').completer = CustomFilesCompleter()  # type: ignore
    get_features_by_flows_command.add_argument('-i', '--info', nargs='?', const=1, default=0, type=int, help="Set verbosity level for feature extraction")
    get_features_by_flows_command.set_defaults(func=get_features_by_flows)
    
    analyse_dataset_command = subparsers.add_parser("analyseDataset", help="Analyze dataset and generate analysis reports")
    analyse_dataset_command.add_argument("dataset_path", nargs='?', default=FEATURES_PATH, type=Path, help="File path to the dataset for analysis").completer = CustomFilesCompleter()  # type: ignore
    analyse_dataset_command.add_argument('-I', '--info-dataset', action='store_true', help="Print detailed dataset information")
    analyse_dataset_command.add_argument('-c', '--correlation-matrix', action='store_true', help="Show correlation matrix")
    analyse_dataset_command.add_argument('-b', '--box-plot', action='store_true', help="Display a box plot for data visualization")
    analyse_dataset_command.add_argument('-p', '--pca', action='store_true', help="Perform PCA on the dataset")
    analyse_dataset_command.add_argument('-o', '--outliers-info', action='store_true', help="Show outlier information")
    analyse_dataset_command.add_argument('-H', '--histograms', action='store_true', help="Display histograms for dataset features")
    analyse_dataset_command.add_argument('-s', '--save-output', action='store_true', help="Save the resulting analysis output to a file")
    analyse_dataset_command.set_defaults(func=analyse_dataset)
    
    analyse_flows_command = subparsers.add_parser("analyseFlows", help="Analyze flows data for insights")
    analyse_flows_command.add_argument("flows_path", nargs='?', default=FLOWS_PATH, type=Path, help="Path to the flows folder").completer = CustomFilesCompleter()  # type: ignore
    analyse_flows_command.set_defaults(func=analyse_flows)
    
    tree_models_command = subparsers.add_parser("treesModels", help="Run tree-based models (CART, Random Forest, Boost) on dataset")
    tree_models_command.add_argument("dataset_path", nargs='?', default=FEATURES_PATH, type=Path, help="Path to the dataset file for model training").completer = CustomFilesCompleter()  # type: ignore
    tree_models_command.add_argument('-c', '--cart', action='store_true', help="Run CART model")
    tree_models_command.add_argument('-rf', '--random-forest', action='store_true', help="Run Random Forest model")
    tree_models_command.add_argument('-b', '--boost', action='store_true', help="Run Boost model")
    tree_models_command.add_argument('-mb', '--multi-boost', action='store_true', help="Run Multy Boost model")
    tree_models_command.add_argument('-tn', '--tree_number', type=int, default=100, help="Number of trees for Random Forest")
    tree_models_command.add_argument('-n', '--number_of_samples', type=int, default=0, help="Max samples per class (≤0 means all)")
    tree_models_command.add_argument('-so', '--select_option', type=int, default=0, help="Selection option: 0 (all), 1 (subset), 2 (VPN vs non-VPN)")
    tree_models_command.add_argument('-ts', '--test_size', type=float, default=0.2, help="Test set proportion")
    tree_models_command.add_argument('-r', '--random_state', type=int, default=42, help="Random state for splitting")
    tree_models_command.add_argument('-md', '--max-depht', type=int, default=None, help="Maximum depth for tree models")
    tree_models_command.add_argument('-v', '--visualize', action='store_true', help="Visualize the tree model results")
    tree_models_command.add_argument('-i', '--info', nargs='?', const=1, default=0, type=int, help="Set verbosity level for feature extraction")
    tree_models_command.add_argument('-dt', '--dataset-test', type=Path, default=None, help="Path to test dataset file (if provided, use for testing)")
    tree_models_command.set_defaults(func=trees_models)
    
    tools_command = subparsers.add_parser("tools", help="Setup development and utility tools")
    tools_command.add_argument("-a", "--add-autocomplete", type=Path, required=False, help="File/directory to add autocomplete support").completer = CustomFilesCompleter()  # type: ignore
    tools_command.add_argument("-mp", "--mypy", action='store_true', help="Run mypy static analysis")
    tools_command.add_argument("-m", "--merge", nargs='?', const=FEATURES_PATH.parent, type=Path, help="Merge all csv files in the directory in a unique csv file").completer = CustomFilesCompleter()  # type: ignore
    tools_command.set_defaults(func=tools)
    
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()