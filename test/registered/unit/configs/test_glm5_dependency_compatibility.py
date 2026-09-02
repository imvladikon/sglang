import importlib.metadata
import unittest
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version
from transformers import AutoConfig

try:
    import tomllib
except ImportError:
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[4]


class TestGlm5DependencyCompatibility(unittest.TestCase):
    @staticmethod
    def _project():
        with (REPO_ROOT / "python" / "pyproject.toml").open("rb") as file:
            return tomllib.load(file)["project"]

    @classmethod
    def _requirements(cls):
        return {
            canonicalize_name(requirement.name): requirement
            for value in cls._project()["dependencies"]
            for requirement in [Requirement(value)]
        }

    def test_transformers_contains_glm5_next(self):
        config = AutoConfig.for_model("glm5_next")

        self.assertEqual(config.model_type, "glm5_next")

    def test_kernels_range_matches_transformers(self):
        requirements = self._requirements()
        installed_transformers = Version(importlib.metadata.version("transformers"))
        self.assertIn(installed_transformers, requirements["transformers"].specifier)

        transformers_kernels = {
            str(requirement.specifier)
            for value in importlib.metadata.requires("transformers") or ()
            for requirement in [Requirement(value)]
            if canonicalize_name(requirement.name) == "kernels"
        }
        self.assertEqual(transformers_kernels, {"<0.17,>=0.16.0"})
        self.assertEqual(str(requirements["kernels"].specifier), "<0.17,>=0.16.0")

    def test_training_stack_dependency_contract(self):
        project = self._project()
        requirements = self._requirements()

        self.assertEqual(str(requirements["openai"].specifier), "<4,>=2.6.1")
        self.assertNotIn("flash-attn-4", requirements)
        fa4 = Requirement(project["optional-dependencies"]["fa4"][0])
        self.assertEqual(canonicalize_name(fa4.name), "flash-attn-4")
        self.assertEqual(str(fa4.specifier), ">=4.0.0b18")

    def test_jit_toolkit_matches_torch_runtime(self):
        requirements = self._requirements()
        torch_toolkit = next(
            requirement
            for value in importlib.metadata.requires("torch") or ()
            for requirement in [Requirement(value)]
            if canonicalize_name(requirement.name) == "cuda-toolkit"
        )

        self.assertEqual(requirements["cuda-toolkit"].extras, {"cccl", "nvcc"})
        self.assertEqual(requirements["cuda-toolkit"].specifier, torch_toolkit.specifier)


if __name__ == "__main__":
    unittest.main()
