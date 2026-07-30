import csv
from pathlib import Path
from typing import Any, TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RNA_FILE = PROJECT_ROOT / "test_rna.tsv"

SUPPORTED_EXPRESSION_COLUMNS = {
    "unstranded",
    "stranded_first",
    "stranded_second",
    "tpm_unstranded",
    "fpkm_unstranded",
    "fpkm_uq_unstranded",
}


def _convert_number(value: str | None) -> float | int | None:
    """
    Convert a TSV value into an integer, float, or None.
    """

    if value is None:
        return None

    cleaned_value = value.strip()

    if cleaned_value == "":
        return None

    try:
        if "." not in cleaned_value:
            return int(cleaned_value)

        return float(cleaned_value)

    except ValueError:
        return None


def _find_header_and_reader(
    file_handle: TextIO,
) -> csv.DictReader:
    """
    Skip comment lines and create a TSV dictionary reader.
    """

    while True:
        position = file_handle.tell()
        line = file_handle.readline()

        if line == "":
            raise ValueError(
                "The RNA-seq file does not contain a valid header."
            )

        if line.startswith("#"):
            continue

        file_handle.seek(position)
        break

    return csv.DictReader(
        file_handle,
        delimiter="\t",
    )


def get_gene_expression(
    genes: list[str],
    expression_column: str = "tpm_unstranded",
    file_path: Path | None = None,
) -> dict[str, Any]:
    """
    Retrieve expression values for selected genes.
    """

    selected_file = file_path or DEFAULT_RNA_FILE

    if not selected_file.exists():
        raise FileNotFoundError(
            f"RNA-seq file was not found at: {selected_file}"
        )

    normalized_column = expression_column.strip().lower()

    if normalized_column not in SUPPORTED_EXPRESSION_COLUMNS:
        supported = ", ".join(
            sorted(SUPPORTED_EXPRESSION_COLUMNS)
        )

        raise ValueError(
            f"Unsupported expression column '{expression_column}'. "
            f"Supported columns are: {supported}."
        )

    requested_genes = {
        gene.strip().upper()
        for gene in genes
        if gene.strip()
    }

    if not requested_genes:
        raise ValueError(
            "At least one valid gene symbol must be provided."
        )

    expression_results: dict[str, dict[str, Any]] = {}

    with selected_file.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as file_handle:
        reader = _find_header_and_reader(file_handle)

        required_columns = {
            "gene_id",
            "gene_name",
            "gene_type",
            normalized_column,
        }

        available_columns = set(reader.fieldnames or [])
        missing_columns = required_columns - available_columns

        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))

            raise ValueError(
                f"The RNA-seq file is missing required columns: "
                f"{missing_text}."
            )

        for row in reader:
            gene_name = (
                row.get("gene_name") or ""
            ).strip().upper()

            if gene_name not in requested_genes:
                continue

            expression_results[gene_name] = {
                "gene_id": row.get("gene_id"),
                "gene_name": row.get("gene_name"),
                "gene_type": row.get("gene_type"),
                "measurement": normalized_column,
                "value": _convert_number(
                    row.get(normalized_column)
                ),
            }

            if len(expression_results) == len(requested_genes):
                break

    missing_genes = sorted(
        requested_genes - set(expression_results)
    )

    ordered_results = {
        gene: expression_results[gene]
        for gene in sorted(expression_results)
    }

    return {
        "source_file": selected_file.name,
        "source_path": str(selected_file),
        "expression_measurement": normalized_column,
        "requested_gene_count": len(requested_genes),
        "found_gene_count": len(ordered_results),
        "missing_gene_count": len(missing_genes),
        "genes": ordered_results,
        "missing_genes": missing_genes,
    }


def preview_rna_file(
    row_count: int = 10,
    file_path: Path | None = None,
) -> dict[str, Any]:
    """
    Return a structured preview of the RNA-seq file.
    """

    selected_file = file_path or DEFAULT_RNA_FILE

    if not selected_file.exists():
        raise FileNotFoundError(
            f"RNA-seq file was not found at: {selected_file}"
        )

    rows: list[dict[str, Any]] = []

    with selected_file.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as file_handle:
        reader = _find_header_and_reader(file_handle)

        for row in reader:
            rows.append(dict(row))

            if len(rows) >= row_count:
                break

        columns = reader.fieldnames or []

    return {
        "source_file": selected_file.name,
        "source_path": str(selected_file),
        "row_count": len(rows),
        "columns": columns,
        "rows": rows,
    }