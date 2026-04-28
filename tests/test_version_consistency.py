"""Tests for version consistency across all files."""

import re
from pathlib import Path


def test_version_consistency() -> None:
    """All version declarations should be consistent."""
    project_root = Path(__file__).parent.parent

    # Expected version from SPEC.md
    spec_file = project_root / "SPEC.md"
    spec_content = spec_file.read_text()
    spec_match = re.search(r"\*\*Version:\*\*\s*(\d+\.\d+\.\d+)", spec_content)
    assert spec_match, "SPEC.md should contain Version field"
    expected_version = spec_match.group(1)

    # Version in __init__.py
    init_file = project_root / "blend" / "__init__.py"
    init_content = init_file.read_text()
    init_match = re.search(r'__version__\s*=\s*["\'](\d+\.\d+\.\d+)["\']', init_content)
    assert init_match, "__init__.py should contain __version__"
    assert init_match.group(1) == expected_version, (
        f"__init__.py version {init_match.group(1)} != SPEC.md version {expected_version}"
    )

    # Version in api.py (FastAPI app)
    api_file = project_root / "blend" / "api.py"
    api_content = api_file.read_text()
    api_match = re.search(r'version=["\'](\d+\.\d+\.\d+)["\']', api_content)
    assert api_match, "api.py should contain FastAPI version"
    assert api_match.group(1) == expected_version, (
        f"api.py version {api_match.group(1)} != SPEC.md version {expected_version}"
    )

    # Version in cli.py
    cli_file = project_root / "blend" / "cli.py"
    if cli_file.exists():
        cli_content = cli_file.read_text()
        cli_match = re.search(r'version=["\'](\d+\.\d+\.\d+)["\']', cli_content)
        if cli_match:
            assert cli_match.group(1) == expected_version, (
                f"cli.py version {cli_match.group(1)} != SPEC.md version {expected_version}"
            )
