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
    @staticmethod
    def _project():
        with (REPO_ROOT / "python" / "pyproject.toml").open("rb") as file:
            return tomllib.load(file)["project"]

    def test_kernels_range_matches_pinned_transformers(self):
        project = self._project()

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

    def test_training_stack_dependency_contract(self):
        project = self._project()
        requirements = {
            canonicalize_name(requirement.name): requirement
            for value in project["dependencies"]
            for requirement in [Requirement(value)]
        }

        self.assertEqual(str(requirements["openai"].specifier), "<4,>=2.6.1")
        self.assertIn(Version("3.6.0"), requirements["openai"].specifier)
        self.assertNotIn("flash-attn-4", requirements)

        fa4 = Requirement(project["optional-dependencies"]["fa4"][0])
        self.assertEqual(canonicalize_name(fa4.name), "flash-attn-4")
        self.assertEqual(str(fa4.specifier), ">=4.0.0b18")

    def test_jit_toolkit_matches_torch_runtime(self):
        project = self._project()
        requirements = {
            canonicalize_name(requirement.name): requirement
            for value in project["dependencies"]
            for requirement in [Requirement(value)]
        }
        sglang_toolkit = requirements["cuda-toolkit"]
        torch_toolkit = next(
            requirement
            for value in importlib.metadata.requires("torch") or ()
            for requirement in [Requirement(value)]
            if canonicalize_name(requirement.name) == "cuda-toolkit"
        )

        self.assertEqual(sglang_toolkit.extras, {"cccl", "nvcc"})
        self.assertEqual(sglang_toolkit.specifier, torch_toolkit.specifier)


if __name__ == "__main__":
    unittest.main()
