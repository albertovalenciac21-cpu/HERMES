from __future__ import annotations

import hashlib
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


GDC_DATA_URL = "https://api.gdc.cancer.gov/data"

CURL_CONNECT_TIMEOUT_SECONDS = 30
CURL_MAX_TIME_SECONDS = 1800
CURL_RETRY_COUNT = 8
CURL_RETRY_DELAY_SECONDS = 3


class DownloadManagerError(RuntimeError):
    """Raised when a GDC file cannot be downloaded or validated."""


def _find_curl() -> str:
    """
    Locate curl on Windows, macOS, or Linux.
    """

    curl_path = shutil.which("curl.exe") or shutil.which("curl")

    if not curl_path:
        raise DownloadManagerError(
            "curl was not found on this computer. On Windows, "
            "confirm that curl.exe is available by running "
            "'curl.exe --version' in PowerShell."
        )

    return curl_path


def _calculate_md5(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Calculate the MD5 checksum of a local file.
    """

    md5_hash = hashlib.md5()

    with path.open("rb") as input_file:
        while True:
            chunk = input_file.read(chunk_size)

            if not chunk:
                break

            md5_hash.update(chunk)

    return md5_hash.hexdigest()


def _normalize_md5(
    value: str | None,
) -> str | None:
    """
    Normalize an expected MD5 value.
    """

    if value is None:
        return None

    cleaned = value.strip().lower()

    return cleaned or None


def _validate_file(
    path: Path,
    expected_size: int | None = None,
    expected_md5: str | None = None,
) -> dict[str, Any]:
    """
    Validate file existence, size, and optional MD5 checksum.
    """

    if not path.exists():
        return {
            "valid": False,
            "reason": "file_missing",
            "actual_size": None,
            "actual_md5": None,
        }

    actual_size = path.stat().st_size

    if actual_size <= 0:
        return {
            "valid": False,
            "reason": "file_empty",
            "actual_size": actual_size,
            "actual_md5": None,
        }

    if (
        expected_size is not None
        and actual_size != expected_size
    ):
        return {
            "valid": False,
            "reason": (
                f"size_mismatch: expected {expected_size}, "
                f"received {actual_size}"
            ),
            "actual_size": actual_size,
            "actual_md5": None,
        }

    normalized_expected_md5 = _normalize_md5(expected_md5)
    actual_md5 = None

    if normalized_expected_md5:
        actual_md5 = _calculate_md5(path)

        if actual_md5.lower() != normalized_expected_md5:
            return {
                "valid": False,
                "reason": (
                    f"md5_mismatch: expected "
                    f"{normalized_expected_md5}, "
                    f"received {actual_md5}"
                ),
                "actual_size": actual_size,
                "actual_md5": actual_md5,
            }

    return {
        "valid": True,
        "reason": None,
        "actual_size": actual_size,
        "actual_md5": actual_md5,
    }


def _remove_file_if_present(
    path: Path,
) -> None:
    """
    Remove a file when it exists.
    """

    try:
        if path.exists():
            path.unlink()
    except OSError as exc:
        raise DownloadManagerError(
            f"Unable to remove incomplete file '{path}': {exc}"
        ) from exc


def _build_curl_command(
    curl_path: str,
    file_id: str,
    partial_path: Path,
) -> list[str]:
    """
    Build the curl command used for a resumable GDC download.
    """

    return [
        curl_path,
        "--http1.1",
        "--location",
        "--fail",
        "--show-error",
        "--silent",
        "--retry",
        str(CURL_RETRY_COUNT),
        "--retry-all-errors",
        "--retry-delay",
        str(CURL_RETRY_DELAY_SECONDS),
        "--connect-timeout",
        str(CURL_CONNECT_TIMEOUT_SECONDS),
        "--max-time",
        str(CURL_MAX_TIME_SECONDS),
        "--continue-at",
        "-",
        "--output",
        str(partial_path),
        f"{GDC_DATA_URL}/{file_id}",
    ]


def download_gdc_file(
    file_id: str,
    destination: Path,
    expected_size: int | None = None,
    expected_md5: str | None = None,
    outer_attempts: int = 4,
) -> dict[str, Any]:
    """
    Download one GDC file using curl.

    Features:
    - local caching
    - HTTP/1.1
    - automatic curl retries
    - outer Python retry loop
    - partial-download resumption
    - file-size validation
    - MD5 checksum validation
    """

    if not file_id or not file_id.strip():
        raise ValueError("A valid GDC file ID is required.")

    if outer_attempts < 1:
        raise ValueError(
            "outer_attempts must be at least 1."
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected_md5 = _normalize_md5(expected_md5)

    existing_validation = _validate_file(
        path=destination,
        expected_size=expected_size,
        expected_md5=expected_md5,
    )

    if existing_validation["valid"]:
        return {
            "path": str(destination),
            "downloaded_now": False,
            "resumed_download": False,
            "file_size": existing_validation["actual_size"],
            "expected_size": expected_size,
            "md5_verified": bool(expected_md5),
            "md5": (
                existing_validation["actual_md5"]
                or expected_md5
            ),
            "download_method": "local_cache",
        }

    if destination.exists():
        _remove_file_if_present(destination)

    partial_path = Path(f"{destination}.part")
    curl_path = _find_curl()

    last_error = "Unknown download failure."
    resumed_download = (
        partial_path.exists()
        and partial_path.stat().st_size > 0
    )

    for attempt in range(1, outer_attempts + 1):
        command = _build_curl_command(
            curl_path=curl_path,
            file_id=file_id,
            partial_path=partial_path,
        )

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=CURL_MAX_TIME_SECONDS + 60,
            )

        except subprocess.TimeoutExpired:
            last_error = (
                "curl exceeded the maximum permitted download time."
            )

            if attempt < outer_attempts:
                time.sleep(attempt * 3)
                continue

            break

        except OSError as exc:
            last_error = f"Unable to start curl: {exc}"

            if attempt < outer_attempts:
                time.sleep(attempt * 3)
                continue

            break

        if result.returncode != 0:
            error_output = (
                result.stderr.strip()
                or result.stdout.strip()
                or "No curl error message was returned."
            )

            last_error = (
                f"curl exited with code {result.returncode}: "
                f"{error_output}"
            )

            if partial_path.exists():
                resumed_download = (
                    partial_path.stat().st_size > 0
                )

            if attempt < outer_attempts:
                time.sleep(attempt * 3)
                continue

            break

        partial_validation = _validate_file(
            path=partial_path,
            expected_size=expected_size,
            expected_md5=expected_md5,
        )

        if not partial_validation["valid"]:
            last_error = (
                "The downloaded file failed validation: "
                f"{partial_validation['reason']}"
            )

            # A complete but corrupt file should not be resumed.
            # Remove it and begin cleanly on the next attempt.
            if partial_validation["reason"].startswith(
                "md5_mismatch"
            ):
                _remove_file_if_present(partial_path)

            if attempt < outer_attempts:
                time.sleep(attempt * 3)
                continue

            break

        partial_path.replace(destination)

        return {
            "path": str(destination),
            "downloaded_now": True,
            "resumed_download": resumed_download,
            "file_size": partial_validation["actual_size"],
            "expected_size": expected_size,
            "md5_verified": bool(expected_md5),
            "md5": (
                partial_validation["actual_md5"]
                or expected_md5
            ),
            "download_method": "curl",
            "curl_attempt": attempt,
        }

    partial_size = (
        partial_path.stat().st_size
        if partial_path.exists()
        else 0
    )

    raise DownloadManagerError(
        f"Unable to download GDC file {file_id} after "
        f"{outer_attempts} outer attempts. "
        f"Partial bytes retained: {partial_size}. "
        f"Last error: {last_error}"
    )