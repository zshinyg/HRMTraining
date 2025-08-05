#!/usr/bin/env python
"""
Safe Code Execution Sandbox for HRM-CodeGen

This module provides a secure environment for executing generated code during
Pass@k evaluation. It uses Docker containers to isolate code execution,
enforces resource limits, restricts network access, and prevents malicious
code execution.

Features:
- Docker-based isolation for secure code execution
- Resource limits (CPU, memory, execution time)
- Network isolation and filesystem restrictions
- Malicious code detection and prevention
- Test case execution against generated code
- Error handling and timeout mechanisms
- Logging and monitoring of execution attempts
- Integration with evaluation pipeline
- Support for multiple Python environments
- Cleanup and resource management

Usage:
    from scripts.security.safe_code_executor import SafeCodeExecutor

    executor = SafeCodeExecutor()
    result = executor.execute_code(
        code="def add(a, b): return a + b",
        test_cases=[{"input": "add(1, 2)", "expected_output": "3"}],
        timeout=5
    )
"""

import base64
import docker
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("safe_code_executor")

# Constants
DEFAULT_TIMEOUT = 10  # seconds
DEFAULT_MEMORY_LIMIT = "512m"
DEFAULT_CPU_LIMIT = 0.5  # CPU cores
MAX_CODE_SIZE = 100 * 1024  # 100 KB
MAX_OUTPUT_SIZE = 1 * 1024 * 1024  # 1 MB
DOCKER_IMAGE_PREFIX = "hrm-codegen-sandbox"
BANNED_MODULES = [
    "subprocess", "os", "sys", "socket", "requests", "urllib", 
    "http", "ftplib", "telnetlib", "smtplib", "importlib",
    "builtins", "ctypes", "multiprocessing", "threading"
]
BANNED_FUNCTIONS = [
    "eval", "exec", "compile", "globals", "locals", "getattr", "setattr",
    "delattr", "__import__", "open", "file", "input", "raw_input"
]
BANNED_PATTERNS = [
    r"__[\w]+__",  # Dunder methods
    r"import\s+(?:" + "|".join(BANNED_MODULES) + ")",  # Banned imports
    r"from\s+(?:" + "|".join(BANNED_MODULES) + ")\s+import",  # Banned from imports
    r"(?:" + "|".join(BANNED_FUNCTIONS) + r")\s*\(",  # Banned function calls
    r"open\s*\(",  # File operations
    r"\\x[0-9a-fA-F]{2}",  # Hex escape sequences
    r"\\[0-7]{3}",  # Octal escape sequences
]


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    passed: bool
    error: Optional[str] = None
    output: Optional[str] = None
    execution_time: float = 0.0
    memory_usage: float = 0.0
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    code_violations: List[str] = field(default_factory=list)


class SafeCodeExecutor:
    """
    Safe code execution environment using Docker containers.
    
    This class provides a secure sandbox for executing untrusted code
    during Pass@k evaluation. It uses Docker containers to isolate
    the execution environment, enforces resource limits, and prevents
    malicious code from being executed.
    """

    def __init__(
        self,
        docker_image: str = None,
        python_version: str = "3.10",
        timeout: int = DEFAULT_TIMEOUT,
        memory_limit: str = DEFAULT_MEMORY_LIMIT,
        cpu_limit: float = DEFAULT_CPU_LIMIT,
        network_disabled: bool = True,
        read_only_filesystem: bool = True,
        enable_malicious_code_detection: bool = True,
        log_executions: bool = True,
        execution_dir: str = None,
    ):
        """
        Initialize the safe code executor.
        
        Args:
            docker_image: Docker image to use (if None, a default image will be built)
            python_version: Python version to use in the sandbox
            timeout: Default execution timeout in seconds
            memory_limit: Memory limit for the container (e.g., "512m", "1g")
            cpu_limit: CPU limit for the container (in cores)
            network_disabled: Whether to disable network access
            read_only_filesystem: Whether to use a read-only filesystem
            enable_malicious_code_detection: Whether to check for malicious code
            log_executions: Whether to log execution attempts
            execution_dir: Directory to store execution files (if None, a temp dir is used)
        """
        self.python_version = python_version
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.network_disabled = network_disabled
        self.read_only_filesystem = read_only_filesystem
        self.enable_malicious_code_detection = enable_malicious_code_detection
        self.log_executions = log_executions
        
        # Set up execution directory
        if execution_dir:
            self.execution_dir = Path(execution_dir)
            self.execution_dir.mkdir(parents=True, exist_ok=True)
            self.cleanup_execution_dir = False
        else:
            self.execution_dir = Path(tempfile.mkdtemp(prefix="hrm_codegen_sandbox_"))
            self.cleanup_execution_dir = True
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise RuntimeError(f"Docker initialization failed: {e}")
        
        # Set up Docker image
        self.docker_image = docker_image
        if not self.docker_image:
            self.docker_image = self._build_or_pull_docker_image()
        
        # Initialize container registry
        self.containers = {}
        
        logger.info(f"SafeCodeExecutor initialized with Python {python_version}, "
                   f"timeout={timeout}s, memory_limit={memory_limit}, cpu_limit={cpu_limit}")
    
    def __del__(self):
        """Cleanup resources when the executor is destroyed."""
        self.cleanup()
    
    def _build_or_pull_docker_image(self) -> str:
        """
        Build or pull the Docker image for code execution.
        
        Returns:
            Name of the Docker image
        """
        image_name = f"{DOCKER_IMAGE_PREFIX}-py{self.python_version}"
        image_tag = "latest"
        full_image_name = f"{image_name}:{image_tag}"
        
        # Check if image already exists
        try:
            self.docker_client.images.get(full_image_name)
            logger.info(f"Docker image {full_image_name} already exists")
            return full_image_name
        except docker.errors.ImageNotFound:
            logger.info(f"Building Docker image {full_image_name}")
        
        # Create a temporary directory for the Dockerfile
        with tempfile.TemporaryDirectory() as temp_dir:
            dockerfile_path = Path(temp_dir) / "Dockerfile"
            
            # Create Dockerfile
            with open(dockerfile_path, "w") as f:
                f.write(f"""
FROM python:{self.python_version}-slim

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r coderunner && useradd -r -g coderunner coderunner

# Create working directory
WORKDIR /sandbox
RUN chown coderunner:coderunner /sandbox

# Install Python dependencies for testing
RUN pip install --no-cache-dir pytest numpy

# Switch to non-root user
USER coderunner

# Set up entrypoint
COPY sandbox_entrypoint.py /sandbox/
ENTRYPOINT ["python", "/sandbox/sandbox_entrypoint.py"]
""")
            
            # Create entrypoint script
            entrypoint_path = Path(temp_dir) / "sandbox_entrypoint.py"
            with open(entrypoint_path, "w") as f:
                f.write("""
import ast
import base64
import json
import os
import sys
import traceback
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Any, Optional

def run_code_with_test_cases(code: str, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run the provided code and execute test cases against it.
    
    Args:
        code: Python code to execute
        test_cases: List of test cases to run
        
    Returns:
        Dictionary with execution results
    """
    result = {
        "success": False,
        "passed": False,
        "error": None,
        "output": "",
        "test_results": []
    }
    
    # Capture stdout and stderr
    output_buffer = StringIO()
    
    try:
        # Execute the code in a restricted environment
        global_env = {}
        local_env = {}
        
        with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
            # Execute the code
            exec(code, global_env, local_env)
        
        # Run test cases
        all_passed = True
        for i, test_case in enumerate(test_cases):
            test_result = {
                "test_id": i,
                "passed": False,
                "input": test_case.get("input", ""),
                "expected_output": test_case.get("expected_output", ""),
                "actual_output": None,
                "error": None
            }
            
            try:
                with redirect_stdout(StringIO()) as test_stdout, redirect_stderr(StringIO()):
                    # Execute the test input
                    actual_output = eval(test_case["input"], global_env, local_env)
                    test_stdout_content = test_stdout.getvalue().strip()
                    
                    # Handle different types of expected output
                    expected_output = test_case["expected_output"]
                    
                    # Check if the output matches the expected output
                    if isinstance(actual_output, (int, float, bool, str, list, dict, tuple, set)):
                        # For simple types, convert to string and compare
                        actual_str = str(actual_output)
                        expected_str = str(expected_output)
                        test_result["passed"] = actual_str == expected_str
                    else:
                        # For complex types, use string representation
                        test_result["passed"] = str(actual_output) == str(expected_output)
                    
                    # Include stdout if it's not empty
                    if test_stdout_content:
                        test_result["stdout"] = test_stdout_content
                    
                    test_result["actual_output"] = str(actual_output)
            
            except Exception as e:
                test_result["error"] = str(e)
                test_result["traceback"] = traceback.format_exc()
                all_passed = False
            
            result["test_results"].append(test_result)
            all_passed = all_passed and test_result["passed"]
        
        result["success"] = True
        result["passed"] = all_passed
        result["output"] = output_buffer.getvalue()
    
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        result["output"] = output_buffer.getvalue()
    
    return result

def main():
    """Main entrypoint for the sandbox."""
    # Read input from environment variables
    code_b64 = os.environ.get("CODE", "")
    test_cases_b64 = os.environ.get("TEST_CASES", "[]")
    
    try:
        # Decode base64 inputs
        code = base64.b64decode(code_b64).decode("utf-8")
        test_cases = json.loads(base64.b64decode(test_cases_b64).decode("utf-8"))
        
        # Run the code with test cases
        result = run_code_with_test_cases(code, test_cases)
        
        # Output the result as JSON
        print(json.dumps(result))
    except Exception as e:
        # Handle any unexpected errors
        error_result = {
            "success": False,
            "passed": False,
            "error": f"Sandbox error: {str(e)}",
            "traceback": traceback.format_exc(),
            "output": ""
        }
        print(json.dumps(error_result))

if __name__ == "__main__":
    main()
""")
            
            # Build the Docker image
            try:
                image, logs = self.docker_client.images.build(
                    path=temp_dir,
                    tag=full_image_name,
                    rm=True,
                    forcerm=True,
                )
                logger.info(f"Successfully built Docker image {full_image_name}")
                return full_image_name
            except Exception as e:
                logger.error(f"Failed to build Docker image: {e}")
                
                # Fall back to pulling a standard Python image
                fallback_image = f"python:{self.python_version}-slim"
                logger.info(f"Falling back to standard image: {fallback_image}")
                self.docker_client.images.pull(fallback_image)
                return fallback_image
    
    def _check_for_malicious_code(self, code: str) -> List[str]:
        """
        Check for potentially malicious code patterns.
        
        Args:
            code: Python code to check
            
        Returns:
            List of detected violations
        """
        if not self.enable_malicious_code_detection:
            return []
        
        violations = []
        
        # Check code size
        if len(code) > MAX_CODE_SIZE:
            violations.append(f"Code exceeds maximum size limit of {MAX_CODE_SIZE} bytes")
        
        # Check for banned patterns
        for pattern in BANNED_PATTERNS:
            matches = re.findall(pattern, code)
            if matches:
                unique_matches = set(matches)
                for match in unique_matches:
                    violations.append(f"Banned pattern detected: {match}")
        
        # Try to parse the code to detect syntax errors
        try:
            ast_tree = ast.parse(code)
            
            # Walk the AST to find potentially dangerous operations
            for node in ast.walk(ast_tree):
                # Check for imports
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if name.name in BANNED_MODULES:
                            violations.append(f"Banned module import: {name.name}")
                
                # Check for from ... import
                elif isinstance(node, ast.ImportFrom):
                    if node.module in BANNED_MODULES:
                        violations.append(f"Banned module import: {node.module}")
                
                # Check for calls to banned functions
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in BANNED_FUNCTIONS:
                        violations.append(f"Banned function call: {node.func.id}")
        
        except SyntaxError as e:
            violations.append(f"Syntax error: {e}")
        except Exception as e:
            logger.warning(f"Error during AST parsing: {e}")
        
        return violations
    
    def execute_code(
        self,
        code: str,
        test_cases: List[Dict[str, Any]],
        timeout: Optional[int] = None,
        python_version: Optional[str] = None,
        memory_limit: Optional[str] = None,
        cpu_limit: Optional[float] = None,
    ) -> ExecutionResult:
        """
        Execute code in a secure sandbox and run test cases against it.
        
        Args:
            code: Python code to execute
            test_cases: List of test cases to run against the code
            timeout: Execution timeout in seconds (overrides default)
            python_version: Python version to use (overrides default)
            memory_limit: Memory limit for execution (overrides default)
            cpu_limit: CPU limit for execution (overrides default)
            
        Returns:
            ExecutionResult object with the results of execution
        """
        # Use default values if not specified
        timeout = timeout or self.timeout
        memory_limit = memory_limit or self.memory_limit
        cpu_limit = cpu_limit or self.cpu_limit
        
        # Generate a unique ID for this execution
        execution_id = str(uuid.uuid4())
        
        # Create execution directory
        execution_path = self.execution_dir / execution_id
        execution_path.mkdir(parents=True, exist_ok=True)
        
        # Log the execution attempt
        if self.log_executions:
            self._log_execution_attempt(execution_id, code, test_cases)
        
        # Check for malicious code
        code_violations = self._check_for_malicious_code(code)
        if code_violations:
            logger.warning(f"Malicious code detected in execution {execution_id}: {code_violations}")
            return ExecutionResult(
                success=False,
                passed=False,
                error="Malicious code detected",
                code_violations=code_violations
            )
        
        # Prepare code and test cases for execution
        code_b64 = base64.b64encode(code.encode("utf-8")).decode("utf-8")
        test_cases_b64 = base64.b64encode(json.dumps(test_cases).encode("utf-8")).decode("utf-8")
        
        # Set up Docker container
        container = None
        start_time = time.time()
        
        try:
            # Run the container with appropriate limits and isolation
            container = self.docker_client.containers.run(
                image=self.docker_image,
                detach=True,
                environment={
                    "CODE": code_b64,
                    "TEST_CASES": test_cases_b64,
                },
                mem_limit=memory_limit,
                nano_cpus=int(cpu_limit * 1e9),  # Convert CPU cores to nano CPUs
                network_disabled=self.network_disabled,
                read_only=self.read_only_filesystem,
                cap_drop=["ALL"],  # Drop all capabilities
                security_opt=["no-new-privileges"],  # Prevent privilege escalation
                tmpfs={"/tmp": "size=64m,exec,nodev,nosuid"},  # Temporary filesystem
                working_dir="/sandbox",
                remove=False,  # We'll remove it manually to ensure cleanup
            )
            
            # Register the container for cleanup
            self.containers[execution_id] = container
            
            # Wait for the container to complete or timeout
            try:
                exit_code = container.wait(timeout=timeout)["StatusCode"]
            except Exception:
                # Container timed out, kill it
                try:
                    container.kill()
                except Exception:
                    pass
                
                return ExecutionResult(
                    success=False,
                    passed=False,
                    error=f"Execution timed out after {timeout} seconds",
                    execution_time=time.time() - start_time
                )
            
            # Get container logs
            logs = container.logs().decode("utf-8", errors="replace")
            
            # Parse the result JSON
            try:
                result_json = json.loads(logs.strip())
                
                # Create the execution result
                execution_result = ExecutionResult(
                    success=result_json.get("success", False),
                    passed=result_json.get("passed", False),
                    error=result_json.get("error"),
                    output=result_json.get("output"),
                    execution_time=time.time() - start_time,
                    test_results=result_json.get("test_results", []),
                    code_violations=code_violations
                )
                
                # Get memory usage if available
                try:
                    stats = container.stats(stream=False)
                    memory_stats = stats.get("memory_stats", {})
                    memory_usage = memory_stats.get("usage", 0) / (1024 * 1024)  # Convert to MB
                    execution_result.memory_usage = memory_usage
                except Exception as e:
                    logger.warning(f"Failed to get container stats: {e}")
                
                return execution_result
            
            except json.JSONDecodeError:
                return ExecutionResult(
                    success=False,
                    passed=False,
                    error="Failed to parse execution result",
                    output=logs,
                    execution_time=time.time() - start_time,
                    code_violations=code_violations
                )
        
        except Exception as e:
            logger.error(f"Error during code execution: {e}")
            return ExecutionResult(
                success=False,
                passed=False,
                error=f"Execution error: {str(e)}",
                execution_time=time.time() - start_time,
                code_violations=code_violations
            )
        
        finally:
            # Clean up the container
            self._cleanup_container(execution_id)
            
            # Clean up the execution directory
            try:
                shutil.rmtree(execution_path)
            except Exception as e:
                logger.warning(f"Failed to remove execution directory: {e}")
    
    def execute_code_batch(
        self,
        codes: List[str],
        test_cases: List[Dict[str, Any]],
        timeout: Optional[int] = None,
        parallel: bool = True,
        max_parallel: int = 4,
    ) -> List[ExecutionResult]:
        """
        Execute multiple code samples against the same test cases.
        
        Args:
            codes: List of Python code samples to execute
            test_cases: List of test cases to run against each code sample
            timeout: Execution timeout in seconds (overrides default)
            parallel: Whether to execute code samples in parallel
            max_parallel: Maximum number of parallel executions
            
        Returns:
            List of ExecutionResult objects
        """
        if not codes:
            return []
        
        if not parallel:
            # Sequential execution
            return [self.execute_code(code, test_cases, timeout) for code in codes]
        
        # Parallel execution
        import concurrent.futures
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
            future_to_code = {
                executor.submit(self.execute_code, code, test_cases, timeout): i
                for i, code in enumerate(codes)
            }
            
            # Initialize results list with placeholders
            results = [None] * len(codes)
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_code):
                code_index = future_to_code[future]
                try:
                    results[code_index] = future.result()
                except Exception as e:
                    logger.error(f"Error executing code at index {code_index}: {e}")
                    results[code_index] = ExecutionResult(
                        success=False,
                        passed=False,
                        error=f"Execution error: {str(e)}"
                    )
        
        return results
    
    def _log_execution_attempt(
        self,
        execution_id: str,
        code: str,
        test_cases: List[Dict[str, Any]]
    ):
        """
        Log an execution attempt for auditing purposes.
        
        Args:
            execution_id: Unique ID for the execution
            code: Python code being executed
            test_cases: Test cases being run
        """
        if not self.log_executions:
            return
        
        log_dir = self.execution_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{execution_id}.json"
        
        log_data = {
            "execution_id": execution_id,
            "timestamp": time.time(),
            "code": code,
            "test_cases": test_cases,
        }
        
        try:
            with open(log_file, "w") as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to log execution attempt: {e}")
    
    def _cleanup_container(self, execution_id: str):
        """
        Clean up a container after execution.
        
        Args:
            execution_id: ID of the execution to clean up
        """
        container = self.containers.pop(execution_id, None)
        if container:
            try:
                container.remove(force=True)
            except Exception as e:
                logger.warning(f"Failed to remove container for execution {execution_id}: {e}")
    
    def cleanup(self):
        """Clean up all resources used by the executor."""
        # Clean up containers
        for execution_id, container in list(self.containers.items()):
            self._cleanup_container(execution_id)
        
        # Clean up execution directory if we created it
        if self.cleanup_execution_dir and self.execution_dir.exists():
            try:
                shutil.rmtree(self.execution_dir)
            except Exception as e:
                logger.warning(f"Failed to remove execution directory: {e}")


class PassKEvaluator:
    """
    Evaluator for Pass@k metrics using the safe code executor.
    
    This class provides functionality to evaluate generated code samples
    using the Pass@k metric, which measures how often at least one of k
    generated samples passes all test cases.
    """

    def __init__(
        self,
        executor: Optional[SafeCodeExecutor] = None,
        k_values: List[int] = [1, 5, 10, 100],
        timeout: int = DEFAULT_TIMEOUT,
        max_parallel: int = 4,
    ):
        """
        Initialize the Pass@k evaluator.
        
        Args:
            executor: SafeCodeExecutor instance (creates a new one if None)
            k_values: List of k values to compute Pass@k for
            timeout: Execution timeout in seconds
            max_parallel: Maximum number of parallel executions
        """
        self.executor = executor or SafeCodeExecutor(timeout=timeout)
        self.k_values = sorted(k_values)
        self.timeout = timeout
        self.max_parallel = max_parallel
    
    def evaluate(
        self,
        problem_id: str,
        generated_codes: List[str],
        test_cases: List[Dict[str, Any]],
        reference_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate generated code samples using Pass@k metric.
        
        Args:
            problem_id: Identifier for the problem
            generated_codes: List of generated code samples
            test_cases: List of test cases to run against the code
            reference_code: Reference solution (optional)
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating problem {problem_id} with {len(generated_codes)} generated samples")
        
        # Limit the number of samples to the maximum k value
        max_k = max(self.k_values)
        if len(generated_codes) > max_k:
            logger.info(f"Limiting evaluation to {max_k} samples")
            generated_codes = generated_codes[:max_k]
        
        # Execute all generated code samples
        results = self.executor.execute_code_batch(
            codes=generated_codes,
            test_cases=test_cases,
            timeout=self.timeout,
            parallel=True,
            max_parallel=self.max_parallel,
        )
        
        # Execute reference code if provided
        reference_result = None
        if reference_code:
            reference_result = self.executor.execute_code(
                code=reference_code,
                test_cases=test_cases,
                timeout=self.timeout,
            )
        
        # Calculate Pass@k metrics
        pass_at_k = {}
        passed_samples = [result.passed for result in results]
        num_passed = sum(passed_samples)
        
        for k in self.k_values:
            if k <= len(generated_codes):
                # Calculate Pass@k using the unbiased estimator
                n = len(generated_codes)
                c = num_passed
                
                if n == 0 or k == 0:
                    pass_at_k[f"pass@{k}"] = 0.0
                elif c == 0:
                    pass_at_k[f"pass@{k}"] = 0.0
                elif k == n:
                    pass_at_k[f"pass@{k}"] = float(c > 0)
                else:
                    # Unbiased estimator for Pass@k
                    # https://arxiv.org/pdf/2107.03374.pdf
                    if c >= k:
                        pass_at_k[f"pass@{k}"] = 1.0
                    else:
                        pass_at_k[f"pass@{k}"] = 1.0 - (
                            (n - c) * self._combinations(n - c, k) /
                            (n * self._combinations(n, k))
                        )
        
        # Prepare the evaluation result
        evaluation_result = {
            "problem_id": problem_id,
            "num_samples": len(generated_codes),
            "num_passed": num_passed,
            "pass_rate": num_passed / len(generated_codes) if generated_codes else 0.0,
            "metrics": pass_at_k,
            "sample_results": [
                {
                    "passed": result.passed,
                    "success": result.success,
                    "error": result.error,
                    "execution_time": result.execution_time,
                    "memory_usage": result.memory_usage,
                    "code_violations": result.code_violations,
                    "test_results": result.test_results,
                }
                for result in results
            ],
        }
        
        # Add reference code results if available
        if reference_result:
            evaluation_result["reference_result"] = {
                "passed": reference_result.passed,
                "success": reference_result.success,
                "error": reference_result.error,
                "execution_time": reference_result.execution_time,
                "memory_usage": reference_result.memory_usage,
            }
        
        return evaluation_result
    
    def _combinations(self, n: int, k: int) -> int:
        """
        Calculate the binomial coefficient (n choose k).
        
        Args:
            n: Total number of items
            k: Number of items to choose
            
        Returns:
            The binomial coefficient (n choose k)
        """
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
        
        k = min(k, n - k)
        c = 1
        for i in range(k):
            c = c * (n - i) // (i + 1)
        return c


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Safe Code Executor for HRM-CodeGen")
    parser.add_argument("--code", type=str, help="Python code to execute")
    parser.add_argument("--code-file", type=str, help="File containing Python code to execute")
    parser.add_argument("--test-cases", type=str, help="JSON file with test cases")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Execution timeout in seconds")
    parser.add_argument("--memory-limit", type=str, default=DEFAULT_MEMORY_LIMIT, help="Memory limit for execution")
    parser.add_argument("--cpu-limit", type=float, default=DEFAULT_CPU_LIMIT, help="CPU limit for execution")
    parser.add_argument("--python-version", type=str, default="3.10", help="Python version to use")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.getLogger("safe_code_executor").setLevel(logging.DEBUG)
    
    # Get code from file or argument
    code = args.code
    if args.code_file:
        with open(args.code_file, "r") as f:
            code = f.read()
    
    if not code:
        parser.error("Either --code or --code-file must be provided")
    
    # Get test cases
    test_cases = []
    if args.test_cases:
        with open(args.test_cases, "r") as f:
            test_cases = json.load(f)
    
    # Initialize executor
    executor = SafeCodeExecutor(
        python_version=args.python_version,
        timeout=args.timeout,
        memory_limit=args.memory_limit,
        cpu_limit=args.cpu_limit,
    )
    
    # Execute code
    result = executor.execute_code(code, test_cases, args.timeout)
    
    # Print or save results
    result_dict = {
        "success": result.success,
        "passed": result.passed,
        "error": result.error,
        "output": result.output,
        "execution_time": result.execution_time,
        "memory_usage": result.memory_usage,
        "test_results": result.test_results,
        "code_violations": result.code_violations,
    }
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result_dict, f, indent=2)
    else:
        print(json.dumps(result_dict, indent=2))
