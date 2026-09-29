"""Execute every notebook from repo root and save its verified outputs."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def main():
    for path in sorted(ROOT.glob("Chapter_*/notebook.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(
            notebook, timeout=120, kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
        nbformat.write(notebook, path)
        errors = [output for cell in notebook.cells if cell.cell_type == "code"
                  for output in cell.get("outputs", []) if output.output_type == "error"]
        if errors:
            raise RuntimeError(f"{path}: {len(errors)} code errors")
        print(f"PASS {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
