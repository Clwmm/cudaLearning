import os
import subprocess

def format_cpp_files(directory, clang_format_path="clang-format", file_extensions=None, ignore_directories=None):
    """
    Formats all C++ and CUDA files in the given directory using clang-format.

    Args:
        directory (str): The path to the project directory.
        clang_format_path (str): Path to the clang-format executable.
        file_extensions (list): List of file extensions to format (default: C++ and CUDA extensions).
        ignore_directories (list): List of directory names to ignore (default: None).
    """
    if file_extensions is None:
        file_extensions = [".cpp", ".hpp", ".h", ".c", ".cc", ".hh", ".cu", ".cuh"]

    if ignore_directories is None:
        ignore_directories = []

    for root, dirs, files in os.walk(directory):
        # Remove ignored directories from the traversal
        dirs[:] = [d for d in dirs if not any(d.startswith(prefix) for prefix in ignore_directories)]

        for file in files:
            if any(file.endswith(ext) for ext in file_extensions):
                file_path = os.path.join(root, file)
                try:
                    subprocess.run([clang_format_path, "-i", file_path], check=True)
                    print(f"Formatted: {file_path}")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to format {file_path}: {e}")
                except FileNotFoundError:
                    print(f"clang-format not found at {clang_format_path}")
                    return

if __name__ == "__main__":
    # Set the root directory to one level above the directory where this Python script is located
    script_directory = os.path.dirname(os.path.abspath(__file__))
    root_directory = os.path.dirname(script_directory)

    # Directories to ignore
    ignore_dirs = ["cmake-"]

    print(f"Formatting files in the root directory: {root_directory}, ignoring directories starting with {ignore_dirs}")
    format_cpp_files(root_directory, ignore_directories=ignore_dirs)