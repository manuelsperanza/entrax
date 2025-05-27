import tempfile
import pytest
from pathlib import Path
from entrax.utils.IO import get_files_recursively, get_available_filepath


# Fixtures
@pytest.fixture
def tmp_test_dir(tmp_path:Path)->Path:
    """Creates a temporary directory structure for testing."""
    # Create a test directory
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()

    # Create files with different extensions
    file1 = test_dir / "file1.txt"
    file1.write_text("Test file 1")

    file2 = test_dir / "file2.md"
    file2.write_text("Test file 2")

    file3 = test_dir / "file3.py"
    file3.write_text("print('hello')")

    # Create a subdirectory
    sub_dir = test_dir / "subdir"
    sub_dir.mkdir()

    # Create a file inside subdirectory
    file4 = sub_dir / "file4.txt"
    file4.write_text("Test file 4")

    return test_dir

@pytest.fixture
def temp_dir():
    """Fixture that creates and cleans up a temporary directory for filepath tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# Existing tests for get_files_recursively
def test_get_files_recursively_txt(tmp_test_dir:Path)->None:
    """Test retrieving `.txt` files recursively."""
    expected_files = {tmp_test_dir / "file1.txt", tmp_test_dir / "subdir" / "file4.txt"}
    result_files = set(get_files_recursively(tmp_test_dir, [".txt"]))
    assert result_files == expected_files


def test_get_files_recursively_md(tmp_test_dir:Path)->None:
    """Test retrieving `.md` files recursively."""
    expected_files = {tmp_test_dir / "file2.md"}
    result_files = set(get_files_recursively(tmp_test_dir, [".md"]))
    assert result_files == expected_files

    def test_file_exists_no_suffixes(self, temp_dir):
        """Test when only the base file exists - should return _1 suffix"""
        file_path = temp_dir / "test.txt"
        file_path.touch()
        result = get_available_filepath(file_path)
        assert result == temp_dir / "test_1.txt"

def test_get_files_recursively_py(tmp_test_dir:Path)->None:
    """Test retrieving `.py` files recursively."""
    expected_files = {tmp_test_dir / "file3.py"}
    result_files = set(get_files_recursively(tmp_test_dir, [".py"]))
    assert result_files == expected_files

    def test_gap_in_numbering(self, temp_dir):
        """Test with non-sequential numbering (1 and 3 exist) - should fill gap with 2"""
        base_path = temp_dir / "test.txt"
        base_path.touch()
        (temp_dir / "test_1.txt").touch()
        (temp_dir / "test_3.txt").touch()
        result = get_available_filepath(base_path)
        assert result == temp_dir / "test_2.txt"

def test_get_files_recursively_multiple_suffixes(tmp_test_dir:Path)->None:
    """Test retrieving `.txt` and `.md` files together."""
    expected_files = {
        tmp_test_dir / "file1.txt",
        tmp_test_dir / "file2.md",
        tmp_test_dir / "subdir" / "file4.txt",
    }
    result_files = set(get_files_recursively(tmp_test_dir, [".txt", ".md"]))
    assert result_files == expected_files

    def test_with_similar_prefix(self, temp_dir):
        """Test that files with similar but not matching prefixes are ignored"""
        base_path = temp_dir / "test.txt"
        base_path.touch()
        (temp_dir / "testfile_1.txt").touch()  # Similar but different prefix
        result = get_available_filepath(base_path)
        assert result == temp_dir / "test_1.txt"

def test_get_files_recursively_no_matches(tmp_test_dir:Path)->None:
    """Test when no files match the given suffix."""
    result_files = get_files_recursively(tmp_test_dir, [".csv"])  # No CSV files in the dir
    assert result_files == []  # Expect an empty list

    def test_with_existing_suffix_only(self, temp_dir):
        """Test when only suffixed versions exist (no base file)"""
        (temp_dir / "test_1.txt").touch()
        (temp_dir / "test_2.txt").touch()
        base_path = temp_dir / "test.txt"
        result = get_available_filepath(base_path)
        assert result == base_path  # Should return base path since it doesn't exist

# New tests for get_available_filepath
def test_get_available_filepath_no_existing_file(tmp_test_dir:Path)->None:
    """Test when no files exist - should return original"""
    test_file = tmp_test_dir / "new_file.txt"
    result = get_available_filepath(test_file)
    assert result == test_file


def test_get_available_filepath_single_existing(tmp_test_dir:Path)->None:
    """Test when only base file exists - should return _1 version"""
    test_file = tmp_test_dir / "document.pdf"
    test_file.touch()
    result = get_available_filepath(test_file)
    assert result == tmp_test_dir / "document_1.pdf"


def test_get_available_filepath_multiple_existing(tmp_test_dir:Path)->None:
    """Test when multiple numbered versions exist"""
    base = tmp_test_dir / "image.jpg"
    base.touch()
    (tmp_test_dir / "image_1.jpg").touch()
    (tmp_test_dir / "image_2.jpg").touch()
    result = get_available_filepath(base)
    assert result == tmp_test_dir / "image_3.jpg"


def test_get_available_filepath_gaps_in_numbering(tmp_test_dir:Path)->None:
    """Test when there are gaps in the numbering sequence"""
    base = tmp_test_dir / "data.json"
    base.touch()
    (tmp_test_dir / "data_1.json").touch()
    (tmp_test_dir / "data_3.json").touch()  # Gap in sequence
    result = get_available_filepath(base)
    assert result == tmp_test_dir / "data_4.json"  # Should use next after highest


def test_get_available_filepath_no_extension(tmp_test_dir:Path)->None:
    """Test with filenames that have no extension"""
    base = tmp_test_dir / "README"
    base.touch()
    (tmp_test_dir / "README_1").touch()
    result = get_available_filepath(base)
    assert result == tmp_test_dir / "README_2"


def test_get_available_filepath_with_subdirectory(tmp_test_dir:Path)->None:
    """Test that it maintains directory structure"""
    subdir = tmp_test_dir / "config"
    subdir.mkdir()
    base = subdir / "settings.ini"
    base.touch()
    result = get_available_filepath(base)
    assert result == subdir / "settings_1.ini"


def test_get_available_filepath_special_chars(tmp_test_dir:Path)->None:
    """Test with filenames containing special characters"""
    base = tmp_test_dir / "my-file_v1.dat"
    base.touch()
    (tmp_test_dir / "my-file_v1_1.dat").touch()
    result = get_available_filepath(base)
    assert result == tmp_test_dir / "my-file_v1_2.dat"
