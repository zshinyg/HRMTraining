#!/usr/bin/env python
"""
Quick CI/CD Pipeline Validation Script for HRM-CodeGen

A simplified validation script that checks for the existence and basic structure
of critical infrastructure components without complex dependencies.

Usage:
    python test/quick_ci_test.py [--verbose]

Options:
    --verbose   Show detailed test output
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path


# Test configuration
REQUIRED_ENV_VARS = [
    "WANDB_API_KEY", "DOCKER_USERNAME", "DOCKER_PASSWORD"
]

REQUIRED_DIRS = [
    "scripts", "configs", ".github/workflows", "scripts/security"
]

REQUIRED_FILES = [
    # CI/CD
    ".github/workflows/ci.yml",
    ".github/workflows/benchmark.yml",
    ".github/workflows/monitoring.yml",
    ".github/workflows/security.yml",
    
    # Docker
    "Dockerfile",
    "scripts/docker_entrypoint.sh",
    
    # Security
    "scripts/security/safe_code_executor.py",
    
    # Benchmarking
    "scripts/benchmark_inference.py",
    "scripts/benchmark_training.py",
    
    # Documentation
    "INFRASTRUCTURE_SETUP.md",
    "SECURITY_FRAMEWORK.md",
    "MONITORING_GUIDE.md"
]

WORKFLOW_REQUIRED_KEYS = ["name", "on", "jobs"]


class QuickCIValidator:
    """Simplified validator for CI/CD pipeline components."""
    
    def __init__(self, verbose=False):
        """Initialize the validator."""
        self.verbose = verbose
        self.results = {}
        self.all_passed = True
        
        # Set up logging
        self._log("Initializing Quick CI/CD pipeline validator")
        self._log(f"Python version: {sys.version.split()[0]}")
        self._log(f"Working directory: {os.getcwd()}")
    
    def _log(self, message):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(f"[CI Validator] {message}")
    
    def run_all_tests(self):
        """Run all validation tests."""
        self._log("Starting validation tests")
        
        # Environment setup
        self.results["environment"] = self.test_environment_setup()
        
        # Repository structure
        self.results["repository"] = self.test_repository_structure()
        
        # GitHub Actions workflows
        self.results["github_actions"] = self.test_github_actions()
        
        # Docker files
        self.results["docker"] = self.test_docker_files()
        
        # Documentation
        self.results["documentation"] = self.test_documentation()
        
        # Print summary
        self._print_summary()
        
        return self.all_passed
    
    def test_environment_setup(self):
        """Test environment setup."""
        self._log("Testing environment setup")
        result = {"passed": True, "message": "Environment setup tests passed"}
        missing_vars = []
        
        # Check environment variables
        for var in REQUIRED_ENV_VARS:
            if not os.environ.get(var):
                missing_vars.append(var)
        
        if missing_vars:
            result["passed"] = False
            result["message"] = f"Missing environment variables: {', '.join(missing_vars)}"
            self.all_passed = False
        
        self._log(f"Environment setup tests: {'PASSED' if result['passed'] else 'FAILED'}")
        return result
    
    def test_repository_structure(self):
        """Test repository structure."""
        self._log("Testing repository structure")
        result = {"passed": True, "message": "Repository structure tests passed"}
        missing_dirs = []
        missing_files = []
        
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
            self.all_passed = False
        
        if missing_files:
            result["passed"] = False
            result["message"] = f"Missing files: {', '.join(missing_files)}"
            self.all_passed = False
        
        self._log(f"Repository structure tests: {'PASSED' if result['passed'] else 'FAILED'}")
        return result
    
    def test_github_actions(self):
        """Test GitHub Actions workflows."""
        self._log("Testing GitHub Actions workflows")
        result = {"passed": True, "message": "GitHub Actions workflow tests passed"}
        invalid_workflows = []
        
        # Check required workflows
        workflows_dir = ".github/workflows"
        required_workflows = ["ci.yml", "benchmark.yml", "monitoring.yml", "security.yml"]
        
        if not os.path.isdir(workflows_dir):
            result["passed"] = False
            result["message"] = "GitHub Actions workflows directory not found"
            self.all_passed = False
            return result
        
        # Validate workflow syntax
        for workflow in required_workflows:
            workflow_path = os.path.join(workflows_dir, workflow)
            
            if not os.path.isfile(workflow_path):
                invalid_workflows.append(f"{workflow} (missing)")
                continue
            
            # Check if workflow file is valid YAML
            try:
                with open(workflow_path, "r") as f:
                    workflow_yaml = yaml.safe_load(f)
                
                # Check basic workflow structure
                missing_keys = [key for key in WORKFLOW_REQUIRED_KEYS if key not in workflow_yaml]
                if missing_keys:
                    invalid_workflows.append(f"{workflow} (missing keys: {', '.join(missing_keys)})")
                    
            except Exception as e:
                invalid_workflows.append(f"{workflow} (error: {str(e)})")
        
        if invalid_workflows:
            result["passed"] = False
            result["message"] = f"Invalid GitHub Actions workflows: {', '.join(invalid_workflows)}"
            self.all_passed = False
        
        self._log(f"GitHub Actions workflow tests: {'PASSED' if result['passed'] else 'FAILED'}")
        return result
    
    def test_docker_files(self):
        """Test Docker files."""
        self._log("Testing Docker files")
        result = {"passed": True, "message": "Docker files tests passed"}
        issues = []
        
        # Check Dockerfile
        dockerfile_path = "Dockerfile"
        if not os.path.isfile(dockerfile_path):
            issues.append("Dockerfile missing")
        else:
            # Check basic Dockerfile content
            with open(dockerfile_path, "r") as f:
                dockerfile_content = f.read()
                if "FROM" not in dockerfile_content:
                    issues.append("Dockerfile missing FROM directive")
                if "ENTRYPOINT" not in dockerfile_content and "CMD" not in dockerfile_content:
                    issues.append("Dockerfile missing ENTRYPOINT or CMD")
        
        # Check docker_entrypoint.sh
        entrypoint_path = "scripts/docker_entrypoint.sh"
        if not os.path.isfile(entrypoint_path):
            issues.append("docker_entrypoint.sh missing")
        else:
            # Check if entrypoint is executable
            if not os.access(entrypoint_path, os.X_OK):
                issues.append("docker_entrypoint.sh not executable")
        
        if issues:
            result["passed"] = False
            result["message"] = f"Docker file issues: {', '.join(issues)}"
            self.all_passed = False
        
        self._log(f"Docker files tests: {'PASSED' if result['passed'] else 'FAILED'}")
        return result
    
    def test_documentation(self):
        """Test documentation files."""
        self._log("Testing documentation")
        result = {"passed": True, "message": "Documentation tests passed"}
        missing_docs = []
        
        # Check required documentation
        required_docs = [
            "INFRASTRUCTURE_SETUP.md",
            "SECURITY_FRAMEWORK.md",
            "MONITORING_GUIDE.md"
        ]
        
        for doc in required_docs:
            if not os.path.isfile(doc):
                missing_docs.append(doc)
        
        if missing_docs:
            result["passed"] = False
            result["message"] = f"Missing documentation: {', '.join(missing_docs)}"
            self.all_passed = False
        
        self._log(f"Documentation tests: {'PASSED' if result['passed'] else 'FAILED'}")
        return result
    
    def _print_summary(self):
        """Print a summary of test results."""
        print("\n" + "=" * 60)
        print("CI/CD PIPELINE QUICK VALIDATION SUMMARY")
        print("=" * 60)
        
        for test_name, test_result in self.results.items():
            status = "✅ PASSED" if test_result.get("passed", False) else "❌ FAILED"
            print(f"{test_name.upper()}: {status}")
            if not test_result.get("passed", False):
                print(f"  - {test_result.get('message', 'Unknown error')}")
        
        print("\n" + "=" * 60)
        if self.all_passed:
            print("✅ ALL TESTS PASSED - CI/CD PIPELINE READY FOR HRM VALIDATION!")
        else:
            print("❌ SOME TESTS FAILED - FIX ISSUES BEFORE PROCEEDING")
        print("=" * 60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Quick validation for HRM-CodeGen CI/CD pipeline")
    parser.add_argument("--verbose", action="store_true", help="Show detailed test output")
    args = parser.parse_args()
    
    # Run validation tests
    validator = QuickCIValidator(verbose=args.verbose)
    success = validator.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
