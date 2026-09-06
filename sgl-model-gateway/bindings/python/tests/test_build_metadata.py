import tomllib
from pathlib import Path


def test_source_build_provides_protoc():
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))

    assert "protoc-wheel-0" in pyproject["build-system"]["requires"]
