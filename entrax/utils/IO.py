from pathlib import Path
from enum import Enum

class DefaultPaths(Enum):
    PCAP = Path(__file__).parent.parent.parent.parent / "data/dataset/pcap/"
    FLOWS = Path(__file__).parent.parent.parent.parent / "data/dataset/flows/"
    FEATURES = Path(__file__).parent.parent.parent.parent /  "data/dataset/csv/Set.csv/"
    LOG = Path(__file__).parent.parent.parent.parent / "data/logs/"
    IMAGES = Path(__file__).parent.parent.parent.parent / "data/images/"

def get_files_recursively(path: Path, suffix_list: list[str])-> list[Path]:
    path_list = []
    
    if path.is_file() and (path.suffix in suffix_list):
        return [path]  # Return a list containing the file
    
    elif path.is_dir():
        for item in path.iterdir():
            path_list.extend(get_files_recursively(item, suffix_list))  # Recursively gather files
    
    return path_list  # Always return a list, even if empty

def get_available_filepath(path:Path) -> Path:
    """
    Returns an available file path by adding numbered suffixes if needed.
    
    Args:
        default_path (str or Path): The original file path to check
        
    Returns:
        Path: The first available path (either original or with _n suffix)
    """
    if not path.exists():
        return path
    
    # Split the path into components
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    
    # Find all existing files with similar names
    existing_files = []
    for f in parent.iterdir():
        if f.is_file() and f.stem.startswith(stem) and f.suffix == suffix:
            existing_files.append(f.name)
    
    # Find the next number to use
    next_num = 0
    list_nums = []
    for name in existing_files:
        if name == path.name:  # The original file exists
            next_num = 1
        else:
            # Try to extract the number from names like stem_123.suffix
            remaining = Path(name).stem[len(stem):]
            if remaining.startswith('_'):
                try:
                    list_nums.append(int(remaining[1:]))
                except ValueError:
                    pass
        
    list_nums.sort()
    for i in range(len(list_nums)):
        if next_num == list_nums[i]:
            next_num +=1
            i+=1
        else:
            break
    
    # Return the first available path
    return parent / f"{stem}_{next_num}{suffix}"
