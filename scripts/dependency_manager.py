#!/usr/bin/env python3
"""
Dependency Management Script
--------------------------
Handles dependency resolution, validation, and environment setup.
"""

import subprocess
import sys
import pkg_resources
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import logging
import json
from dataclasses import dataclass
import argparse
import platform
from concurrent.futures import ThreadPoolExecutor
import re
from datetime import datetime
@dataclass
class DependencyConfig:
    """Configuration for dependency management."""
    name: str
    version: str
    optional: bool = False
    alternatives: List[str] = None
    conflicts: List[str] = None
    
class DependencyManager:
    """Manages project dependencies and environments."""
    
    def __init__(
        self,
        project_root: Optional[Path] = None,
        log_file: Optional[str] = "dependency_manager.log"
    ):
        self.project_root = project_root or Path(__file__).parent.parent
        self.logger = self._setup_logging(log_file)
        
        # Initialize dependency tracking
        self.installed_packages: Dict[str, str] = {}
        self.missing_packages: List[str] = []
        self.conflicts: List[Tuple[str, str]] = []
        
    def _setup_logging(self, log_file: Optional[str]) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("DependencyManager")
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
        return logger
        
    def check_environment(self) -> bool:
        """Check Python environment and system requirements."""
        self.logger.info("Checking environment...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            self.logger.error(
                f"Python version {python_version.major}.{python_version.minor} "
                "is not supported. Please use Python 3.8 or higher."
            )
            return False
            
        # Check for virtual environment
        in_venv = hasattr(sys, 'real_prefix') or sys.base_prefix != sys.prefix
        if not in_venv:
            self.logger.warning(
                "Not running in a virtual environment. "
                "It's recommended to use a virtual environment."
            )
            
        # Check system dependencies
        system_deps = self._check_system_dependencies()
        if not system_deps:
            return False
            
        return True
        
    def _check_system_dependencies(self) -> bool:
        """Check system-level dependencies."""
        system = platform.system().lower()
        
        if system == "linux":
            return self._check_linux_dependencies()
        elif system == "darwin":
            return self._check_mac_dependencies()
        elif system == "windows":
            return self._check_windows_dependencies()
        else:
            self.logger.error(f"Unsupported operating system: {system}")
            return False
            
    def resolve_dependencies(
        self,
        requirements_files: List[str] = None
    ) -> bool:
        """Resolve project dependencies."""
        if requirements_files is None:
            requirements_files = [
                "requirements.txt",
                "requirements-dev.txt",
                "requirements-extra.txt"
            ]
            
        all_requirements = set()
        for req_file in requirements_files:
            req_path = self.project_root / req_file
            if not req_path.exists():
                self.logger.warning(f"Requirements file not found: {req_file}")
                continue
                
            with open(req_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        all_requirements.add(line)
                        
        return self._install_dependencies(all_requirements)
        
    def _install_dependencies(self, requirements: Set[str]) -> bool:
        """Install required dependencies."""
        self.logger.info("Installing dependencies...")
        
        success = True
        with ThreadPoolExecutor() as executor:
            futures = []
            for req in requirements:
                futures.append(
                    executor.submit(self._install_package, req)
                )
                
            for future in futures:
                if not future.result():
                    success = False
                    
        return success
        
    def _install_package(self, requirement: str) -> bool:
        """Install a single package."""
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", requirement],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.logger.info(f"Successfully installed {requirement}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install {requirement}: {str(e)}")
            return False
            
    def validate_installation(self) -> bool:
        """Validate installed packages."""
        self.logger.info("Validating installation...")
        
        try:
            installed = pkg_resources.working_set
            installed_packages = {pkg.key: pkg.version for pkg in installed}
            
            # Check core requirements
            with open(self.project_root / "requirements.txt") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        package = re.split('[><=~]', line)[0].strip()
                        if package not in installed_packages:
                            self.missing_packages.append(package)
                            
            if self.missing_packages:
                self.logger.error(
                    f"Missing packages: {', '.join(self.missing_packages)}"
                )
                return False
                
            self.logger.info("All required packages are installed")
            return True
            
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            return False
            
    def generate_report(self) -> Dict:
        """Generate dependency report."""
        return {
            "installed_packages": self.installed_packages,
            "missing_packages": self.missing_packages,
            "conflicts": self.conflicts,
            "python_version": sys.version,
            "platform": platform.platform(),
            "timestamp": datetime.now().isoformat()
        }
        
def main():
    parser = argparse.ArgumentParser(
        description="Manage project dependencies"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check environment and dependencies"
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install dependencies"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate installation"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate dependency report"
    )
    
    args = parser.parse_args()
    
    manager = DependencyManager()
    
    if args.check:
        manager.check_environment()
    if args.install:
        manager.resolve_dependencies()
    if args.validate:
        manager.validate_installation()
    if args.report:
        report = manager.generate_report()
        print(json.dumps(report, indent=2))
        
if __name__ == "__main__":
    main() 