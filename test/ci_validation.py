#!/usr/bin/env python
"""
CI/CD Pipeline Validation Script for HRM-CodeGen

This script validates the core functionality of the CI/CD pipeline,
including environment setup, dependencies, Pass@k evaluation components,
and Docker integration. It serves as a smoke test to ensure the research
validation infrastructure is properly configured.

Usage:
    python test/ci_validation.py [--docker] [--full] [--verbose]

Options:
    --docker    Run Docker-specific tests
    --full      Run all tests including resource-intensive ones
    --verbose   Show detailed test output
"""

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Test configuration
REQUIRED_PACKAGES = [
    "torch",
    "numpy",
    "pandas",
    "matplotlib",
    "transformers",
    "wandb",
    "pytest",
    "docker",
]

REQUIRED_ENV_VARS = ["WANDB_API_KEY", "DOCKER_USERNAME", "DOCKER_PASSWORD"]

REQUIRED_DIRS = ["scripts", "configs", "data", "tests"]

REQUIRED_FILES = [
    "scripts/security/safe_code_executor.py",
    "scripts/benchmark_inference.py",
    "scripts/benchmark_training.py",
    "Dockerfile",
    ".github/workflows/ci.yml",
    ".github/workflows/benchmark.yml",
]

# Sample code for Pass@k testing
SAMPLE_CODE = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

SAMPLE_TEST_CASES = [
    {"input": "fibonacci(5)", "expected_output": "5"},
    {"input": "fibonacci(10)", "expected_output": "55"},
]


class CIPipelineValidator:
    """Validates the CI/CD pipeline components."""

    def __init__(
        self,
        docker_tests: bool = False,
        full_tests: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize the validator.

        Args:
            docker_tests: Whether to run Docker-specific tests
            full_tests: Whether to run resource-intensive tests
            verbose: Whether to show detailed output
        """
        self.docker_tests = docker_tests
        self.full_tests = full_tests
        self.verbose = verbose
        self.results = {}
        self.temp_dir = None

        # Create a temporary directory for test artifacts
        self.temp_dir = tempfile.mkdtemp(prefix="hrm_ci_validation_")

        # Set up logging
        self._log("Initializing CI/CD pipeline validator")
        self._log(f"Python version: {platform.python_version()}")
        self._log(f"Platform: {platform.platform()}")
        self._log(f"Working directory: {os.getcwd()}")
        self._log(f"Temporary directory: {self.temp_dir}")

    def __del__(self):
        """Clean up resources."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _log(self, message: str):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(f"[CI Validator] {message}")

    def run_all_tests(self) -> Dict[str, Dict[str, Union[bool, str]]]:
        """
        Run all validation tests.

        Returns:
            Dictionary with test results
        """
        self._log("Starting validation tests")

        # Basic Python functionality
        self.results["python"] = self.test_python_functionality()

        # Environment setup
        self.results["environment"] = self.test_environment_setup()

        # Dependencies
        self.results["dependencies"] = self.test_dependencies()

        # Repository structure
        self.results["repository"] = self.test_repository_structure()

        # Pass@k evaluation components
        self.results["pass_at_k"] = self.test_pass_at_k_components()

        # Docker integration
        if self.docker_tests:
            self.results["docker"] = self.test_docker_integration()

        # GitHub Actions workflows
        self.results["github_actions"] = self.test_github_actions()

        # Print summary
        self._print_summary()

        return self.results

    def test_python_functionality(self) -> Dict[str, Union[bool, str]]:
        """
        Test basic Python functionality.

        Returns:
            Test result dictionary
        """
        self._log("Testing Python functionality")
        result = {"passed": True, "message": "Python functionality tests passed"}

        try:
            # Test basic Python operations
            assert 1 + 1 == 2, "Basic arithmetic failed"
            assert "test".upper() == "TEST", "String operations failed"
            assert [1, 2, 3][-1] == 3, "List operations failed"

            # Test exception handling
            try:
                1 / 0
                assert False, "Exception not raised"
            except ZeroDivisionError:
                pass

            # Test file I/O
            test_file = os.path.join(self.temp_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test")
            with open(test_file, "r") as f:
                assert f.read() == "test", "File I/O failed"

            # Test subprocess
            output = subprocess.check_output(
                ["python", "-c", "print('subprocess test')"]
            )
            assert b"subprocess test" in output, "Subprocess failed"

        except Exception as e:
            result["passed"] = False
            result["message"] = f"Python functionality tests failed: {str(e)}"

        self._log(
            f"Python functionality tests: {'PASSED' if result['passed'] else 'FAILED'}"
        )
        return result

    def test_environment_setup(self) -> Dict[str, Union[bool, str]]:
        """
        Test environment setup.

        Returns:
            Test result dictionary
        """
        self._log("Testing environment setup")
        result = {"passed": True, "message": "Environment setup tests passed"}
        missing_vars = []

        try:
            # Check Python version
            assert sys.version_info >= (3, 8), "Python version must be at least 3.8"

            # Check environment variables
            for var in REQUIRED_ENV_VARS:
                if not os.environ.get(var):
                    missing_vars.append(var)

            if missing_vars:
                result["passed"] = False
                result["message"] = (
                    f"Missing environment variables: {', '.join(missing_vars)}"
                )
                result["missing_vars"] = missing_vars

            # Check for GPU if full tests are enabled
            if self.full_tests:
                try:
                    import torch

                    result["gpu_available"] = torch.cuda.is_available()
                    if torch.cuda.is_available():
                        result["gpu_name"] = torch.cuda.get_device_name(0)
                        result["gpu_count"] = torch.cuda.device_count()
                    else:
                        self._log("GPU not available, some tests may be skipped")
                except ImportError:
                    result["gpu_available"] = False
                    self._log("Torch not installed, GPU tests skipped")

        except Exception as e:
            result["passed"] = False
            result["message"] = f"Environment setup tests failed: {str(e)}"

        self._log(
            f"Environment setup tests: {'PASSED' if result['passed'] else 'FAILED'}"
        )
        return result

    def test_dependencies(self) -> Dict[str, Union[bool, str]]:
        """
        Test required dependencies.

        Returns:
            Test result dictionary
        """
        self._log("Testing dependencies")
        result = {"passed": True, "message": "Dependency tests passed"}
        missing_packages = []

        try:
            # Check required packages
            for package in REQUIRED_PACKAGES:
                try:
                    importlib.import_module(package)
                except ImportError:
                    missing_packages.append(package)

            if missing_packages:
                result["passed"] = False
                result["message"] = f"Missing packages: {', '.join(missing_packages)}"
                result["missing_packages"] = missing_packages

            # Check specific package functionality if full tests are enabled
            if self.full_tests and result["passed"]:
                # Test torch
                import torch

                x = torch.rand(5, 3)

                # Test numpy
                import numpy as np

                arr = np.array([1, 2, 3])
                assert np.mean(arr) == 2, "NumPy functionality failed"

                # Test transformers
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                tokens = tokenizer("Hello world", return_tensors="pt")
                assert "input_ids" in tokens, "Transformers functionality failed"

        except Exception as e:
            result["passed"] = False
            result["message"] = f"Dependency tests failed: {str(e)}"

        self._log(f"Dependency tests: {'PASSED' if result['passed'] else 'FAILED'}")
        return result

    def test_repository_structure(self) -> Dict[str, Union[bool, str]]:
        """
        Test repository structure.

        Returns:
            Test result dictionary
        """
        self._log("Testing repository structure")
        result = {"passed": True, "message": "Repository structure tests passed"}
        missing_dirs = []
        missing_files = []

        try:
            # Check required directories
            for directory in REQUIRED_DIRS:
                if not os.path.isdir(directory):
                    missing_dirs.append(directory)

            # Check required files
            for file_path in REQUIRED_FILES:
                if not os.path.isfile(file_path):
                    missing_files.append(file_path)

            if missing_dirs:
                result["passed"] = False
                result["message"] = f"Missing directories: {', '.join(missing_dirs)}"
                result["missing_dirs"] = missing_dirs

            if missing_files:
                result["passed"] = False
                result["message"] = f"Missing files: {', '.join(missing_files)}"
                result["missing_files"] = missing_files

        except Exception as e:
            result["passed"] = False
            result["message"] = f"Repository structure tests failed: {str(e)}"

        self._log(
            f"Repository structure tests: {'PASSED' if result['passed'] else 'FAILED'}"
        )
        return result

    def test_pass_at_k_components(self) -> Dict[str, Union[bool, str]]:
        """
        Test Pass@k evaluation components.

        Returns:
            Test result dictionary
        """
        self._log("Testing Pass@k evaluation components")
        result = {"passed": True, "message": "Pass@k component tests passed"}

        try:
            # Check if safe_code_executor.py exists and can be imported
            if not os.path.isfile("scripts/security/safe_code_executor.py"):
                result["passed"] = False
                result["message"] = "safe_code_executor.py not found"
                return result

            # Add scripts directory to path for importing
            sys.path.insert(0, os.path.abspath("."))

            # Try to import the safe code executor
            try:
                from scripts.security.safe_code_executor import (
                    SafeCodeExecutor,
                    PassKEvaluator,
                )

                # Test basic functionality if full tests are enabled
                if self.full_tests:
                    # Create a temporary executor for testing
                    executor = SafeCodeExecutor(
                        timeout=5,
                        memory_limit="256m",
                        cpu_limit=0.1,
                        execution_dir=self.temp_dir,
                    )

                    # Test code execution
                    execution_result = executor.execute_code(
                        code=SAMPLE_CODE, test_cases=SAMPLE_TEST_CASES
                    )

                    assert execution_result.success, "Code execution failed"
                    assert execution_result.passed, "Test cases failed"

                    # Test Pass@k evaluation
                    evaluator = PassKEvaluator(executor=executor)
                    evaluation_result = evaluator.evaluate(
                        problem_id="test_problem",
                        generated_codes=[SAMPLE_CODE],
                        test_cases=SAMPLE_TEST_CASES,
                    )

                    assert (
                        evaluation_result["num_passed"] > 0
                    ), "Pass@k evaluation failed"
                    assert (
                        "pass@1" in evaluation_result["metrics"]
                    ), "Pass@k metrics missing"

                    # Clean up
                    executor.cleanup()
                else:
                    self._log("Skipping full Pass@k tests (use --full to enable)")

            except ImportError as e:
                result["passed"] = False
                result["message"] = f"Failed to import safe_code_executor: {str(e)}"

        except Exception as e:
            result["passed"] = False
            result["message"] = f"Pass@k component tests failed: {str(e)}"

        self._log(
            f"Pass@k component tests: {'PASSED' if result['passed'] else 'FAILED'}"
        )
        return result

    def test_docker_integration(self) -> Dict[str, Union[bool, str]]:
        """
        Test Docker integration.

        Returns:
            Test result dictionary
        """
        self._log("Testing Docker integration")
        result = {"passed": True, "message": "Docker integration tests passed"}

        try:
            # Check if Docker is installed
            try:
                docker_version = subprocess.check_output(
                    ["docker", "--version"]
                ).decode("utf-8")
                result["docker_version"] = docker_version.strip()
                self._log(f"Docker version: {docker_version.strip()}")
            except (subprocess.SubprocessError, FileNotFoundError):
                result["passed"] = False
                result["message"] = "Docker not installed or not in PATH"
                return result

            # Check if Dockerfile exists
            if not os.path.isfile("Dockerfile"):
                result["passed"] = False
                result["message"] = "Dockerfile not found"
                return result

            # Check Docker image build if full tests are enabled
            if self.full_tests:
                # Build a minimal test image
                test_tag = f"hrm-codegen-test:{int(time.time())}"

                try:
                    build_cmd = [
                        "docker",
                        "build",
                        "-t",
                        test_tag,
                        "--target",
                        "base",
                        ".",
                    ]
                    self._log(f"Building test Docker image: {' '.join(build_cmd)}")

                    build_process = subprocess.run(
                        build_cmd,
                        stdout=subprocess.PIPE if not self.verbose else None,
                        stderr=subprocess.PIPE if not self.verbose else None,
                        check=True,
                    )

                    # Check if image exists
                    images = subprocess.check_output(
                        ["docker", "images", "-q", test_tag]
                    ).decode("utf-8")
                    if not images.strip():
                        result["passed"] = False
                        result["message"] = "Failed to build Docker image"
                    else:
                        self._log(f"Successfully built Docker image: {test_tag}")

                        # Clean up
                        subprocess.run(
                            ["docker", "rmi", test_tag],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )

                except subprocess.CalledProcessError as e:
                    result["passed"] = False
                    result["message"] = (
                        f"Docker build failed: {e.stderr.decode('utf-8') if e.stderr else str(e)}"
                    )
            else:
                self._log("Skipping Docker build test (use --full to enable)")

        except Exception as e:
            result["passed"] = False
            result["message"] = f"Docker integration tests failed: {str(e)}"

        self._log(
            f"Docker integration tests: {'PASSED' if result['passed'] else 'FAILED'}"
        )
        return result

    def test_github_actions(self) -> Dict[str, Union[bool, str]]:
        """
        Test GitHub Actions workflows.

        Returns:
            Test result dictionary
        """
        self._log("Testing GitHub Actions workflows")
        result = {"passed": True, "message": "GitHub Actions workflow tests passed"}
        missing_workflows = []

        try:
            # Check if GitHub Actions workflows exist
            workflows_dir = ".github/workflows"
            required_workflows = [
                "ci.yml",
                "benchmark.yml",
                "monitoring.yml",
                "security.yml",
            ]

            if not os.path.isdir(workflows_dir):
                result["passed"] = False
                result["message"] = "GitHub Actions workflows directory not found"
                return result

            # Check required workflows
            for workflow in required_workflows:
                workflow_path = os.path.join(workflows_dir, workflow)
                if not os.path.isfile(workflow_path):
                    missing_workflows.append(workflow)

            if missing_workflows:
                result["passed"] = False
                result["message"] = (
                    f"Missing GitHub Actions workflows: {', '.join(missing_workflows)}"
                )
                result["missing_workflows"] = missing_workflows
                return result

            # Validate workflow syntax
            for workflow in required_workflows:
                workflow_path = os.path.join(workflows_dir, workflow)

                # Check if workflow file is valid YAML
                try:
                    import yaml

                    with open(workflow_path, "r") as f:
                        workflow_yaml = yaml.safe_load(f)

                    # Check basic workflow structure
                    assert "name" in workflow_yaml, f"Missing 'name' in {workflow}"
                    assert "on" in workflow_yaml, f"Missing 'on' in {workflow}"
                    assert "jobs" in workflow_yaml, f"Missing 'jobs' in {workflow}"

                except (yaml.YAMLError, AssertionError) as e:
                    result["passed"] = False
                    result["message"] = f"Invalid workflow YAML in {workflow}: {str(e)}"
                    return result

        except Exception as e:
            result["passed"] = False
            result["message"] = f"GitHub Actions workflow tests failed: {str(e)}"

        self._log(
            f"GitHub Actions workflow tests: {'PASSED' if result['passed'] else 'FAILED'}"
        )
        return result

    def _print_summary(self):
        """Print a summary of test results."""
        print("\n" + "=" * 60)
        print("CI/CD PIPELINE VALIDATION SUMMARY")
        print("=" * 60)

        all_passed = True
        for test_name, test_result in self.results.items():
            status = "✅ PASSED" if test_result.get("passed", False) else "❌ FAILED"
            print(f"{test_name.upper()}: {status}")
            if not test_result.get("passed", False):
                all_passed = False
                print(f"  - {test_result.get('message', 'Unknown error')}")

        print("\n" + "=" * 60)
        if all_passed:
            print("✅ ALL TESTS PASSED - CI/CD PIPELINE READY FOR HRM VALIDATION!")
        else:
            print("❌ SOME TESTS FAILED - FIX ISSUES BEFORE PROCEEDING")
        print("=" * 60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate CI/CD pipeline for HRM-CodeGen"
    )
    parser.add_argument(
        "--docker", action="store_true", help="Run Docker-specific tests"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all tests including resource-intensive ones",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed test output"
    )
    args = parser.parse_args()

    # Run validation tests
    validator = CIPipelineValidator(
        docker_tests=args.docker, full_tests=args.full, verbose=args.verbose
    )
    results = validator.run_all_tests()

    # Determine exit code based on test results
    all_passed = all(result.get("passed", False) for result in results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
