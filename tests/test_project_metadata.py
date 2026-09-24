from pathlib import Path


def test_required_files_exist():
    root = Path(__file__).parents[1]
    for name in ["README.md", "Dockerfile", "environment.yml", "requirements.txt", "pyproject.toml"]:
        assert (root / name).exists()
