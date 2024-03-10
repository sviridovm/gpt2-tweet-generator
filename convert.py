import nbformat
import os


def py_to_ipynb(py_file, ipynb_file):
    with open(py_file, 'r') as py:
        py_code = py.read()

    notebook = nbformat.v4.new_notebook()
    code_cell = nbformat.v4.new_code_cell(source=py_code)
    notebook.cells.append(code_cell)

    with open(ipynb_file, 'w') as ipynb:
        nbformat.write(notebook, ipynb)


def convert_all_py_to_ipynb(directory="/"):
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    for filename in os.listdir(directory):
        if filename.endswith(".py"):

            if filename == "convert.py":
                continue

            py_file_path = os.path.join(directory, filename)
            ipynb_file_path = os.path.splitext(py_file_path)[0] + ".ipynb"
            py_to_ipynb(py_file_path, ipynb_file_path)
            print(f"Converted {filename} to {
                  os.path.basename(ipynb_file_path)}")


if __name__ == "__main__":
    # source_directory = "/path/to/your/py/files"
    source_directory = os.getcwd()
    convert_all_py_to_ipynb(source_directory)
