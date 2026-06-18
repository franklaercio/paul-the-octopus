"""Orquestra os notebooks do pipeline em ordem e valida as entradas.

Ordem executável: 01_features -> 02_train -> 03_predict (00_eda fica de fora).

A lógica científica vive nos notebooks; este script valida os contratos dos CSVs
de entrada, executa os notebooks em ordem e salva os notebooks executados em
artifacts/. Conforme cada etapa for implementada, acrescente aqui a validação
dos artefatos de saída (features, modelo, submission).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from scripts.validate_data import ROOT, ValidationError, validate_repository

NOTEBOOKS_DIR = ROOT / "notebooks"
ARTIFACTS_DIR = ROOT / "artifacts"

# Notebooks que compõem o pipeline executável, na ordem. 00_eda fica de fora.
PIPELINE_NOTEBOOKS = (
    "01_features.ipynb",
    "02_train.ipynb",
    "03_predict.ipynb",
)


def execute_notebook(path: Path, output_dir: Path, timeout: int) -> Path:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute(cwd=str(ROOT))
    output_dir.mkdir(parents=True, exist_ok=True)
    executed = output_dir / f"{path.stem}.executed.ipynb"
    nbformat.write(notebook, executed)
    return executed


def run_pipeline(output_dir: Path, timeout: int) -> None:
    validate_repository()
    for name in PIPELINE_NOTEBOOKS:
        path = NOTEBOOKS_DIR / name
        if not path.is_file():
            raise ValidationError(f"Notebook ausente: {path}")
        executed = execute_notebook(path, output_dir, timeout)
        print(f"OK: {name} executado -> {executed.name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Executa os notebooks do pipeline em ordem e valida as entradas."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Pasta onde gravar os notebooks executados.",
    )
    parser.add_argument("--timeout", type=int, default=900, help="Timeout por celula, em segundos.")
    args = parser.parse_args()

    try:
        run_pipeline(args.output_dir.resolve(), args.timeout)
    except (ValidationError, Exception) as exc:
        print(f"ERRO: {exc}")
        return 1

    print("OK: pipeline executado")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
