import importlib.metadata
import unittest
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

try:
    import tomllib
except ImportError:
    import tomli as tomllib


register_cpu_ci(est_time=1, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[4]


class TestGlm53DependencyCompatibility(CustomTestCase):
    def test_kernels_range_matches_pinned_transformers(self):
        with (REPO_ROOT / "python" / "pyproject.toml").open("rb") as file:
            project = tomllib.load(file)["project"]

        requirements = {
            canonicalize_name(requirement.name): requirement
            for value in project["dependencies"]
            for requirement in [Requirement(value)]
        }
        sglang_kernels = requirements["kernels"]
        sglang_transformers = requirements["transformers"]

        installed_transformers = Version(
            importlib.metadata.version("transformers")
        )
        self.assertIn(installed_transformers, sglang_transformers.specifier)

        transformers_kernels = {
            str(requirement.specifier)
            for value in importlib.metadata.requires("transformers") or ()
            for requirement in [Requirement(value)]
            if canonicalize_name(requirement.name) == "kernels"
        }
        self.assertEqual(transformers_kernels, {"<0.17,>=0.16.0"})
        self.assertEqual(str(sglang_kernels.specifier), "<0.17,>=0.16.0")


if __name__ == "__main__":
    unittest.main()
