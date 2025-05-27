from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

import matplotlib
matplotlib.use('TkAgg')

from entrax.utils.log import setup_logger, write_info, write_debug_message
from entrax.utils.IO import get_files_recursively
from entrax.utils.general import get_class
from entrax.pcap_processing.flow import read_from_csv
from entrax.utils.constants import TlsInfo
from typing import Any, Dict, List  # <-- new import for type annotations

class FlowsAnalyser():
    def __init__(self, flows_path:Path):
        setup_logger(owner=self, log_filename=__name__, level=2)

        flows_paths = get_files_recursively(flows_path, ['.csv'])
        flows_dict = defaultdict(list)

        data: Dict[Any, Dict[Any, int]] = {}

        i = 0
        for flow_path in flows_paths:
            print(f"Progress: {i}", end='\r')
            i+=1
            label = get_class(flow_path.parent)
            flows_dict[label].append(read_from_csv(flow_path))

        print()
        self.analyseTLSMetadata(flows_dict=flows_dict, data=data)

    def analyseTLSMetadata(self, flows_dict: Dict[Any, List[Any]], data: Dict[Any, Dict[Any, int]]) -> None:  # Updated type annotations and return type
        i = 0
        for label, flows in flows_dict.items():
            print(f"Progress: {i}", end='\r')
            i+=1
            data[label] = {TlsInfo.NOTLS : 0, TlsInfo.TLS:0, TlsInfo.TLS1:0 , TlsInfo.TLS2:0, TlsInfo.TLS3:0}
            for flow in flows:
                res = flow.get_tls_info()
                data[label] = {key: data[label][key] + res[key] for key in res}

        main_labels = [l.value for l in data.keys()]
        main_totals = [sum(sub.values()) for sub in data.values()]
        grand_total = sum(main_totals)

        # Generate colors
        main_cmap = plt.get_cmap("Set3")
        main_colors = [main_cmap(i) for i in range(len(main_labels))]

        # Sub colors: generate distinct ones
        all_sub_labels = set(k for d in data.values() for k in d.keys())
        sub_cmap = plt.get_cmap("tab20")  # good for many categories
        sub_color_map = {label: sub_cmap(i % 20) for i, label in enumerate(all_sub_labels)}


        # Prepare outer slices
        outer_labels = []
        outer_sizes = []
        outer_colors = []
        outer_pct_texts = []

        for i, main in enumerate(data.keys()):
            main_total = sum(data[main].values())
            for sub in data[main]:
                size = data[main][sub]
                outer_sizes.append(size)
                outer_labels.append(f"{sub.value} ({main.value})")
                outer_colors.append(sub_color_map[sub])
                pct = size / main_total * 100
                outer_pct_texts.append(f"{pct:.1f}%")

        

        # Plot
        fig, ax = plt.subplots(figsize=(10, 10))

        # Outer pie with safe unpacking
        pie_result = ax.pie(
            outer_sizes,
            labels=None,  # Hide text labels
            colors=outer_colors,
            radius=1.2,
            wedgeprops=dict(width=0.3, edgecolor='white'),
            autopct=lambda pct: "",  # Only show if > 5%
            pctdistance=0.95
        )
        if len(pie_result) == 3:
            wedges_outer, texts_outer, autotexts_outer = pie_result
        else:
            wedges_outer, texts_outer = pie_result
            autotexts_outer = []

        # 👇 Assign the correct percentage to each sub wedge
        # Replace default percentage texts with your own (relative to main)
        for i, txt in enumerate(autotexts_outer):
            pct_value = float(outer_pct_texts[i].strip('%'))  # Convert "x.y%" → float
            if pct_value > 5:
                txt.set_text(outer_pct_texts[i])
                txt.set_fontsize(10)
            else:
                txt.set_text("") 

        # Add manual legend to prevent overlapping
        # Create a legend with unique sub labels only
        handles = []
        labels_seen = set()

        for label in all_sub_labels:
            if label not in labels_seen:
                patch = Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=sub_color_map[label],
                                markersize=10, label=label.value)
                handles.append(patch)
                labels_seen.add(label)

        ax.legend(handles=handles, title="Subcategories", loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11)

        # Inner pie
        ax.pie(
            main_totals,
            labels=main_labels,
            colors=main_colors,
            radius=0.9,
            wedgeprops=dict(width=0.3, edgecolor='white'),
            autopct=lambda pct: f"{pct:.1f}%" if pct > 0.5 else "",
            pctdistance=0.55,
            labeldistance=0.8
        )

        # Title and layout
        plt.title("TLS distribution in packets (%)", fontsize=18)
        ax.set(aspect="equal")
        plt.tight_layout()
        plt.show()