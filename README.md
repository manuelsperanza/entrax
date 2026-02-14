# 📦 EnTrAX (Encrypted Traffic Analyser X)

EnTrAX is a Python toolkit and CLI for processing PCAP traffic captures, generating flow/feature datasets, and running analysis workflows. It includes utilities for dataset inspection and classic tree-based models (CART, Random Forest, Boosting) used in traffic classification experiments.

The project is packaged with `pyproject.toml` and exposes a command-line interface via `entrax`.

## 📂 Project Structure

A high-level overview of the repository:

```
entrax/
├── entrax/               # Main Python package
│   ├── cli.py            # CLI entry point
│   ├── pcap_processing/  # PCAP → flows/features + analysis helpers
│   ├── tree_models/      # CART, Random Forest, Boosting implementations
│   └── utils/            # IO, logging, constants, and helper utilities
├── tests/                # Unit and module tests
├── pyproject.toml        # Project metadata & build configuration
└── .gitignore            # Git ignore rules
```

## 📋 Prerequisites

Ensure you have **Python 3.8+** and **pip** installed.

Optionally, create a virtual environment:

```bash
# On Linux/macOS
python3 -m venv venvName
source venv/bin/activate

# On Windows
python -m venv venvName
.\venv\Scripts\activate
```

## 🚀 Installation

1. Clone the repository:

```bash
git clone https://gitlab.unige.ch/Manuel.Speranza/entrax.git
cd entrax
```

2. Install the package:

```bash
pip install .
```

### 3. (Optional) Enable Autocompletion  

To enable autocompletion for your CLI tool, add the following command to your shell configuration:  

#### **For system-wide autocompletion (Bash users)**  
```bash
echo 'eval "$(register-python-argcomplete entrax)"' >> ~/.bashrc
source ~/.bashrc
```  

#### **For virtual environments (venv users)**  
Replace `venvName` with your actual virtual environment name:  
```bash
echo 'eval "$(register-python-argcomplete entrax)"' >> venvName/bin/activate
source venvName/bin/activate
```  

For **Zsh users**, replace `~/.bashrc` with `~/.zshrc`.  

This installs dependencies and registers your CLI tool.

## 🛠️ Usage

### Run the CLI

After installation, you can use the custom CLI command:

```bash
entrax --help
```

**Usage Examples:**

- Generate flows from a PCAP file:

```bash
entrax getFlowsByPcap path/to/input.pcap -sfw data/flows -i 1
```

- Extract features from PCAP flows:

```bash
entrax getFeaturesByPcap path/to/input.pcap -sfw data/flows -sft data/features -i 1
```

- Extract features from pre-generated flows:

```bash
entrax getFeaturesByFlows data/flows -sft data/features -i 1
```

- Analyze a dataset with correlation and box plot:

```bash
entrax analyseDataset data/features/TrainingSet.csv -I -c -b
```

## Workflow for Development

1. **Always work on `dev`**  
    When adding features or fixing bugs, switch to the `dev` branch:

```bash
git checkout dev
```

2. **Commit your changes**

```bash
git add .
git commit -m "Added feature X"
```

3. **Test your changes**
Run your tests to ensure everything is working correctly before merging.

## Merge to main When Stable

1. **Once your changes in dev are tested and stable:**

```bash
git checkout main
git merge dev
git commit -m "Merge dev into main"
```

2. **Then tag the stable version:**

```bash
git tag v1.0  # Increment for future versions (e.g., v1.1, v2.0)
```

This ensures that main always contains a reliable version of your project.

## Rollback if Needed

If something goes wrong and you need to revert:

1. **Check your commit history:**

```bash
git log --oneline
```

2. **Roll back to a previous commit:**

```bash
git reset --hard <commit-hash>
```

3. **If you only want to undo the last commit but keep your changes:**

```bash
git reset --soft HEAD~1
```

This allows you to recover from mistakes and maintain a clean project history.





