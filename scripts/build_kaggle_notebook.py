from __future__ import annotations

from pathlib import Path

import nbformat


def main() -> None:
    source_path = Path("notebooks/rogii_flat_tail_baseline.py")
    output_path = Path("notebooks/rogii_submission.ipynb")
    source = source_path.read_text(encoding="utf-8")

    notebook = nbformat.v4.new_notebook(
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        }
    )
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# ROGII conservative saturated-ramp submission\n\n"
            "Offline, inference-only notebook. It discovers the hidden rerun wells, "
            "uses only `MD` and `TVT_input`, and writes `/kaggle/working/submission.csv`."
        ),
        nbformat.v4.new_code_cell(source),
    ]
    nbformat.write(notebook, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
