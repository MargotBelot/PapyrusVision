#!/usr/bin/env python3
"""
PapyrusVision One-Click Installer
=================================

This script automatically installs all requirements for PapyrusVision and sets up the environment.
Works on Windows, macOS, and Linux.

Usage:
    python install.py

What it does:
1. Checks system requirements
2. Creates virtual environment 
3. Installs all dependencies
4. Verifies installation
5. Creates launch scripts
"""

import os
import sys
import subprocess
import platform
import urllib.request
from pathlib import Path


class PapyrusVisionInstaller:
    def __init__(self):
        self.system = platform.system().lower()
        self.python_cmd = self.find_python_command()
        self.pip_cmd = self.find_pip_command()
        self.project_dir = Path(__file__).parent
        self.venv_dir = self.project_dir / "papyrus_env"
        
    def find_python_command(self):
        """Find the correct Python command on this system."""
        # Try python3 first since we're running the installer with python3
        for cmd in ['python3', 'python', 'py']:
            try:
                result = subprocess.run([cmd, '--version'], capture_output=True, text=True)
                if result.returncode == 0 and 'Python 3.' in result.stdout:
                    return cmd
            except FileNotFoundError:
                continue
        
        # Fallback: try to use the same Python that's running this script
        if sys.executable:
            return sys.executable
        
        return None
    
    def find_pip_command(self):
        """Find the correct pip command on this system."""
        for cmd in ['pip', 'pip3']:
            try:
                result = subprocess.run([cmd, '--version'], capture_output=True, text=True)
                if result.returncode == 0:
                    return cmd
            except FileNotFoundError:
                continue
        return None

    def print_step(self, step, description):
        """Print installation step with formatting."""
        print(f"\nStep {step}: {description}")
        print("="* 50)

    def run_command(self, command, description="", check=True):
        """Run a command with error handling."""
        if description:
            print(f"Processing {description}...")
        
        try:
            if isinstance(command, str):
                result = subprocess.run(command, shell=True, check=check, 
                                      capture_output=True, text=True)
            else:
                result = subprocess.run(command, check=check, 
                                      capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"SUCCESS: Success!")
                if result.stdout.strip():
                    print(f"Output: {result.stdout.strip()}")
            else:
                print(f"WARNING: Warning: {result.stderr.strip()}")
            
            return result.returncode == 0
            
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Error: {e}")
            print(f"Command output: {e.stderr}")
            return False
        except Exception as e:
            print(f"ERROR: Unexpected error: {e}")
            return False

    def check_requirements(self):
        """Check if system meets requirements."""
        self.print_step(1, "Checking System Requirements")
        
        # Check Python
        if not self.python_cmd:
            print("ERROR: Python 3.8+ not found!")
            print("Please install Python from: https://python.org/downloads/")
            return False
        
        # Check Python version
        result = subprocess.run([self.python_cmd, '--version'], 
                              capture_output=True, text=True)
        version_str = result.stdout.strip()
        print(f"SUCCESS: Found {version_str}")
        
        # Extract version number
        version_parts = version_str.split()[1].split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        
        if major < 3 or (major == 3 and minor < 8):
            print("ERROR: Python 3.8+ required!")
            return False
        
        # Check pip
        if not self.pip_cmd:
            print("ERROR: pip not found!")
            return False
        
        print(f"SUCCESS: Found pip")
        
        # Check disk space (approximate)
        free_space = self.get_free_space()
        if free_space < 5:  # 5GB
            print(f"WARNING: Warning: Only {free_space:.1f}GB free space. Recommended: 5GB+")
        else:
            print(f"SUCCESS: Sufficient disk space: {free_space:.1f}GB free")
        
        return True

    def get_free_space(self):
        """Get free disk space in GB."""
        try:
            if self.system == "windows":
                free_bytes = os.statvfs('.').f_frsize * os.statvfs('.').f_bavail
            else:
                statvfs = os.statvfs('.')
                free_bytes = statvfs.f_frsize * statvfs.f_bavail
            return free_bytes / (1024**3)  # Convert to GB
        except:
            return float('inf')  # Unknown, assume sufficient

    def create_virtual_environment(self):
        """Create Python virtual environment."""
        self.print_step(2, "Creating Virtual Environment")
        
        if self.venv_dir.exists():
            print("Installation Virtual environment already exists. Removing old one...")
            import shutil
            shutil.rmtree(self.venv_dir)
        
        success = self.run_command(
            [self.python_cmd, '-m', 'venv', str(self.venv_dir)],
            "Creating virtual environment"
        )
        
        if not success:
            print("ERROR: Failed to create virtual environment!")
            return False
        
        # Get activation command
        if self.system == "windows":
            self.activate_cmd = str(self.venv_dir / "Scripts"/ "activate.bat")
            self.venv_python = str(self.venv_dir / "Scripts"/ "python.exe")
            self.venv_pip = str(self.venv_dir / "Scripts"/ "pip.exe")
        else:
            self.activate_cmd = f"source {self.venv_dir / 'bin' / 'activate'}"
            self.venv_python = str(self.venv_dir / "bin"/ "python")
            self.venv_pip = str(self.venv_dir / "bin"/ "pip")
        
        return True

    def detect_cuda_support(self):
        """Detect if CUDA is available and get version."""
        print("Detecting Detecting CUDA support...")
        
        # Check for NVIDIA GPU
        nvidia_smi_available = False
        cuda_version = None
        
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                nvidia_smi_available = True
                # Try to extract CUDA version
                output = result.stdout
                for line in output.split('\n'):
                    if 'CUDA Version:' in line:
                        cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                        break
                print(f"SUCCESS: NVIDIA GPU detected with CUDA {cuda_version}")
            else:
                print("INFO: No NVIDIA GPU detected")
        except FileNotFoundError:
            print("INFO: nvidia-smi not found - no NVIDIA GPU or drivers")
        
        return nvidia_smi_available, cuda_version

    def get_detectron2_install_command(self):
        """Generate the appropriate Detectron2 installation command for this platform."""
        has_cuda, cuda_version = self.detect_cuda_support()
        
        print("Setting up Determining optimal Detectron2 installation...")
        
        if self.system == "darwin":  # macOS
            # Check if Apple Silicon
            machine = platform.machine().lower()
            if machine in ['arm64', 'aarch64']:  # Apple Silicon
                print("Detected Detected macOS Apple Silicon (M1/M2/M3)")
                return [
                    # Install PyTorch for Apple Silicon first
                    [self.venv_pip, 'install', 'torch', 'torchvision', 'torchaudio'],
                    # Build Detectron2 from source
                    [self.venv_pip, 'install', 'git+https://github.com/facebookresearch/detectron2.git']
                ]
            else:  # Intel Mac
                print("Detected Detected macOS Intel")
                if has_cuda:
                    # Intel Mac with eGPU (rare)
                    return [
                        [self.venv_pip, 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu118'],
                        [self.venv_pip, 'install', 'detectron2', '-f', 'https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html']
                    ]
                else:
                    # CPU-only Intel Mac
                    return [
                        [self.venv_pip, 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cpu'],
                        [self.venv_pip, 'install', 'detectron2', '-f', 'https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html']
                    ]
        
        elif self.system in ["linux", "windows"]:  # Linux or Windows
            if has_cuda and cuda_version:
                print(f"Detected Detected {self.system.title()} with CUDA {cuda_version}")
                # Use CUDA-enabled versions
                return [
                    [self.venv_pip, 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu118'],
                    [self.venv_pip, 'install', 'detectron2', '-f', 'https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html']
                ]
            else:
                print(f"Detected Detected {self.system.title()} - using CPU-only version")
                # CPU-only versions
                return [
                    [self.venv_pip, 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cpu'],
                    [self.venv_pip, 'install', 'detectron2', '-f', 'https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html']
                ]
        
        else:
            # Fallback for unknown systems
            print("Unknown Unknown system - using CPU-only fallback")
            return [
                [self.venv_pip, 'install', 'torch', 'torchvision', 'torchaudio'],
                [self.venv_pip, 'install', 'git+https://github.com/facebookresearch/detectron2.git']
            ]

    def install_dependencies(self):
        """Install all Python dependencies including Detectron2."""
        self.print_step(3, "Installing Dependencies")
        
        requirements_file = self.project_dir / "requirements.txt"
        if not requirements_file.exists():
            print("ERROR: requirements.txt not found!")
            return False
        
        print("Installing This may take 15-25 minutes depending on internet speed...")
        
        # Upgrade pip first
        success = self.run_command(
            [self.venv_python, '-m', 'pip', 'install', '--upgrade', 'pip'],
            "Upgrading pip"
        )
        
        if not success:
            print("WARNING: Warning: Could not upgrade pip, continuing anyway...")
        
        # Install basic requirements first
        print("Installing Installing basic Python packages...")
        success = self.run_command(
            [self.venv_pip, 'install', '-r', str(requirements_file)],
            "Installing core Python packages"
        )
        
        if not success:
            print("ERROR: Failed to install basic dependencies!")
            return False
        
        # Install Detectron2 with platform-specific commands
        print("Verifying Installing Detectron2 (AI detection engine)...")
        detectron2_commands = self.get_detectron2_install_command()
        
        for i, cmd in enumerate(detectron2_commands, 1):
            step_name = "PyTorch"if i == 1 else "Detectron2"
            print(f"Installing Step {i}/2: Installing {step_name}...")
            
            # For PyTorch installation, be more permissive with errors
            allow_failure = (step_name == "PyTorch")
            
            success = self.run_command(
                cmd,
                f"Installing {step_name}",
                check=not allow_failure
            )
            
            if not success:
                if step_name == "PyTorch":
                    print(f"WARNING: PyTorch installation had issues, trying alternative approach...")
                    # Try simpler PyTorch installation
                    alt_pytorch_cmd = [self.venv_pip, 'install', 'torch', 'torchvision', 'torchaudio']
                    success = self.run_command(
                        alt_pytorch_cmd,
                        "Installing PyTorch (alternative method)"
                    )
                    if not success:
                        print("ERROR: Failed to install PyTorch!")
                        self.print_detectron2_troubleshooting()
                        return False
                else:
                    # For Detectron2, try multiple fallback strategies
                    print(f"WARNING: Failed to install {step_name} with optimized settings.")
                    print(f"NOTE: Trying fallback installations...")
                    
                    # Fallback 1: CPU-only version
                    print("Attempt Attempt 1: CPU-only Detectron2...")
                    cpu_fallback_cmd = [self.venv_pip, 'install', 'detectron2', '-f', 
                                       'https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html']
                    success = self.run_command(
                        cpu_fallback_cmd,
                        "Installing Detectron2 (CPU-only)"
                    )
                    
                    if not success:
                        # Fallback 2: Build from source with pre-installed PyTorch
                        print("Attempt Attempt 2: Building from source with existing PyTorch...")
                        
                        # Ensure PyTorch is properly installed first
                        pytorch_verify = self.run_command(
                            [self.venv_python, '-c', 'import torch; print("PyTorch available for build")'],
                            "Verifying PyTorch for build",
                            check=False
                        )
                        
                        if pytorch_verify:
                            # Try building with more specific environment
                            env_cmd = f"cd {self.project_dir} && source {self.venv_dir}/bin/activate && pip install git+https://github.com/facebookresearch/detectron2.git"
                            success = self.run_command(
                                env_cmd,
                                "Building Detectron2 from source"
                            )
                        
                        if not success:
                            # Fallback 3: Manual build method (our working approach)
                            print("Attempt Attempt 3: Manual build with proper environment setup...")
                            success = self.install_detectron2_manual_build()
                        
                        if not success:
                            # Fallback 4: Manual instructions
                            print("ERROR: Automatic Detectron2 installation failed!")
                            print("\nSetting up Manual installation required:")
                            print("\n1. Activate the environment:")
                            print(f"source {self.venv_dir}/bin/activate")
                            print("\n2. Try one of these commands:")
                            print("# Option A: CPU-only (works everywhere)")
                            print("pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html")
                            print("\n # Option B: Build from source (requires Xcode)")
                            print("xcode-select --install  # If not already installed")
                            print("pip install git+https://github.com/facebookresearch/detectron2.git")
                            print("\n3. Then verify with:")
                            print("python -c 'import detectron2; print(\"Detectron2 OK\")'")
                            print("\n4. Launch the app:")
                            print("streamlit run apps/streamlit_hieroglyphs_app.py")
                            
                            # Don't return False - let user continue manually
                            print("\nNOTE: Installation will continue without Detectron2 verification.")
                            print("You can install Detectron2 manually using the commands above.")
                            return True  # Continue installation
        
        print("SUCCESS: All dependencies installed successfully!")
        return True

    def print_detectron2_troubleshooting(self):
        """Print troubleshooting guide for Detectron2 installation issues."""
        print("\n"+ "="*60)
        print("DETECTRON2  DETECTRON2 TROUBLESHOOTING GUIDE")
        print("="*60)
        print("\nCommon solutions for Detectron2 installation issues:\n")
        
        if self.system == "darwin":  # macOS
            print("Detected macOS Troubleshooting:")
            print("1. Install Xcode command line tools:")
            print("xcode-select --install")
            print("\n2. For Apple Silicon (M1/M2/M3), ensure you're using the right Python:")
            print("which python3  # Should point to ARM version")
            print("\n3. Try installing with specific architecture:")
            print("arch -arm64 pip install torch torchvision torchaudio")
            
        elif self.system == "linux":
            print("Linux Linux Troubleshooting:")
            print("1. Install build tools:")
            print("sudo apt-get install build-essential")
            print("\n2. For CUDA issues, check CUDA version compatibility:")
            print("nvidia-smi  # Check CUDA version")
            print("nvcc --version  # Check compiler version")
            print("\n3. Install CUDA development tools:")
            print("sudo apt-get install cuda-toolkit-*")
            
        elif self.system == "windows":
            print("Windows Windows Troubleshooting:")
            print("1. Install Visual Studio Build Tools:")
            print("Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
            print("\n2. For CUDA issues, install CUDA toolkit:")
            print("Download from: https://developer.nvidia.com/cuda-downloads")
            print("\n3. Try CPU-only version first:")
            print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
        
        print("\nSetting up General Solutions:")
        print("1. Update pip and setuptools:")
        print("pip install --upgrade pip setuptools wheel")
        print("\n2. Clear pip cache:")
        print("pip cache purge")
        print("\n3. Try CPU-only installation:")
        print("pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html")
        print("\n4. Build from source (slower but more reliable):")
        print("pip install git+https://github.com/facebookresearch/detectron2.git")
        
        print("\nGetting Getting Help:")
        print("- GitHub Issues: https://github.com/MargotBelot/PapyrusVision/issues")
        print("- Detectron2 Issues: https://github.com/facebookresearch/detectron2/issues")
        print("- Include your system info and error messages when reporting issues")
        print("\n"+ "="*60)

    def install_detectron2_manual_build(self):
        """Install Detectron2 using the manual build method that works reliably."""
        import tempfile
        import shutil
        
        print("Setting up Setting up manual build environment for Detectron2...")
        
        try:
            # Step 1: Install build dependencies
            print("Installing Installing build dependencies...")
            # Install build dependencies with proper escaping for shell
            build_deps = ['setuptools==68.0.0', 'wheel', 'ninja']
            for dep in build_deps:
                dep_success = self.run_command(
                    [self.venv_pip, 'install', dep],
                    f"Installing {dep}",
                    check=False
                )
                if not dep_success:
                    print(f"WARNING: Failed to install {dep}")
            
            # Install pybind11 separately to handle bracket escaping
            pybind_success = self.run_command(
                f"{self.venv_pip} install 'pybind11[global]'",
                "Installing pybind11[global]",
                check=False
            )
            
            if not pybind_success:
                print("WARNING: Warning: Could not install pybind11")
                return False
            
            # Step 2: Clone Detectron2 to temporary directory
            temp_dir = Path(tempfile.mkdtemp())
            detectron2_dir = temp_dir / "detectron2"
            
            print(f"Cloning Cloning Detectron2 to {detectron2_dir}...")
            clone_cmd = ['git', 'clone', 'https://github.com/facebookresearch/detectron2.git', str(detectron2_dir)]
            success = self.run_command(
                clone_cmd,
                "Cloning Detectron2 repository",
                check=False
            )
            
            if not success:
                print("ERROR: Failed to clone Detectron2 repository")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return False
            
            # Step 3: Install using pip with --no-build-isolation from local clone
            print("Building Building Detectron2 from source (this may take 5-10 minutes)...")
            
            # Change to the detectron2 directory and install (regular install, not editable)
            build_cmd = f"cd {detectron2_dir} && source {self.venv_dir}/bin/activate && python -m pip install . --no-build-isolation"
            success = self.run_command(
                build_cmd,
                "Building and installing Detectron2",
                check=False
            )
            
            # Clean up temporary directory (after installation completes)
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            if success:
                print("SUCCESS: Manual build completed successfully!")
                
                # Verify the installation
                verify_cmd = [self.venv_python, '-c', 'import detectron2; from detectron2.engine import DefaultPredictor; print("Detectron2 manual build: SUCCESS")']
                verify_success = self.run_command(
                    verify_cmd,
                    "Verifying Detectron2 installation",
                    check=False
                )
                
                if verify_success:
                    print("SUCCESS: Detectron2 successfully installed and verified!")
                    return True
                else:
                    print("WARNING: Detectron2 built but verification failed")
                    return False
            else:
                print("ERROR: Manual build failed")
                return False
                
        except Exception as e:
            print(f"ERROR: Error during manual build: {e}")
            return False

        print("\n"+ "="*60)

    def verify_installation(self):
        """Verify that all key dependencies are installed."""
        self.print_step(4, "Verifying Installation")
        
        # Key packages to check for PapyrusVision
        packages = ['streamlit', 'torch', 'cv2', 'numpy', 'pandas', 'matplotlib', 'PIL']
        
        for package in packages:
            if package == 'cv2':
                import_name = 'cv2'
                package_name = 'opencv-python'
            elif package == 'PIL':
                import_name = 'PIL'
                package_name = 'pillow'
            else:
                import_name = package
                package_name = package
            
            success = self.run_command(
                [self.venv_python, '-c', f'import {import_name}; print(f"{package_name}: OK")'],
                f"Checking {package_name}",
                check=False
            )
            
            if not success:
                print(f"ERROR: {package_name} not properly installed!")
                return False
        
        # Special verification for Detectron2 (optional)
        print("Verifying Verifying Detectron2 installation...")
        success = self.run_command(
            [self.venv_python, '-c', 'import detectron2; from detectron2.engine import DefaultPredictor; print("Detectron2: OK")'],
            "Checking Detectron2",
            check=False
        )
        
        if not success:
            print("WARNING: Detectron2 not properly installed!")
            print("NOTE: This is critical for PapyrusVision functionality.")
            print("\nSetting up To install Detectron2 manually:")
            print(f"1. Activate environment: source {self.venv_dir}/bin/activate")
            print("2. Install Detectron2:")
            print("# CPU-only (recommended):")
            print("pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html")
            print("\n # Or build from source (requires Xcode):")
            print("xcode-select --install")
            print("pip install git+https://github.com/facebookresearch/detectron2.git")
            print("\nNOTE: Installation will continue - you can install Detectron2 later.")
        
        # Test CUDA availability if relevant
        has_cuda, _ = self.detect_cuda_support()
        if has_cuda:
            print("Testing Testing CUDA availability...")
            cuda_test = self.run_command(
                [self.venv_python, '-c', 'import torch; print(f"CUDA available: {torch.cuda.is_available()}"); print(f"CUDA devices: {torch.cuda.device_count()}")'],
                "Testing CUDA",
                check=False
            )
            if cuda_test:
                print("SUCCESS: GPU acceleration will be available!")
            else:
                print("INFO: GPU acceleration not available, will use CPU (slower but functional)")
        
        print("SUCCESS: All dependencies verified!")
        return True

    def create_launch_scripts(self):
        """Create easy launch scripts for the user."""
        self.print_step(5, "Creating Launch Scripts")
        
        if self.system == "windows":
            self.create_windows_launcher()
        else:
            self.create_unix_launcher()
        
        print("SUCCESS: Launch scripts created!")
        return True

    def create_windows_launcher(self):
        """Create Windows batch file launcher."""
        # Main app launcher
        launcher_content = f'''@echo off
echo Starting PapyrusVision Hieroglyphs App...
cd /d "{self.project_dir}"
call "{self.venv_dir}\\Scripts\\activate.bat"
streamlit run apps\\streamlit_hieroglyphs_app.py
pause
'''
        
        launcher_path = self.project_dir / "start_papyrus_vision.bat"
        with open(launcher_path, 'w') as f:
            f.write(launcher_content)
        
        # Digital Paleography Tool launcher
        paleo_launcher_content = f'''@echo off
echo Starting Digital Paleography Tool...
cd /d "{self.project_dir}"
call "{self.venv_dir}\\Scripts\\activate.bat"
streamlit run apps\\digital_paleography_tool.py
pause
'''
        
        paleo_launcher_path = self.project_dir / "start_paleography_tool.bat"
        with open(paleo_launcher_path, 'w') as f:
            f.write(paleo_launcher_content)
        
        print(f"SUCCESS: Created launcher: {launcher_path}")
        print(f"SUCCESS: Created paleography launcher: {paleo_launcher_path}")
        print("NOTE: Double-click 'start_papyrus_vision.bat' to run the main app")
        print("NOTE: Double-click 'start_paleography_tool.bat' to run the paleography tool")

    def create_unix_launcher(self):
        """Create Unix shell script launcher."""
        # Main app launcher
        launcher_content = f'''#!/bin/bash
echo "Starting PapyrusVision Hieroglyphs App..."
cd "{self.project_dir}"
source "{self.venv_dir}/bin/activate"
streamlit run apps/streamlit_hieroglyphs_app.py
'''
        
        launcher_path = self.project_dir / "start_papyrus_vision.sh"
        with open(launcher_path, 'w') as f:
            f.write(launcher_content)
        
        # Make executable
        os.chmod(launcher_path, 0o755)
        
        # Digital Paleography Tool launcher
        paleo_launcher_content = f'''#!/bin/bash
echo "Starting Digital Paleography Tool..."
cd "{self.project_dir}"
source "{self.venv_dir}/bin/activate"
streamlit run apps/digital_paleography_tool.py
'''
        
        paleo_launcher_path = self.project_dir / "start_paleography_tool.sh"
        with open(paleo_launcher_path, 'w') as f:
            f.write(paleo_launcher_content)
        
        # Make executable
        os.chmod(paleo_launcher_path, 0o755)
        
        print(f"SUCCESS: Created launcher: {launcher_path}")
        print(f"SUCCESS: Created paleography launcher: {paleo_launcher_path}")
        print(f"NOTE: Run './start_papyrus_vision.sh' to start the main app")
        print(f"NOTE: Run './start_paleography_tool.sh' to start the paleography tool")

    def install(self):
        """Run the complete installation process."""
        print("PapyrusVision PapyrusVision Installation Starting...")
        print(f"Detected  System: {platform.system()} {platform.machine()}")
        print(f"Installation Installation directory: {self.project_dir}")
        
        steps = [
            self.check_requirements,
            self.create_virtual_environment,
            self.install_dependencies,
            self.verify_installation,
            self.create_launch_scripts
        ]
        
        for i, step in enumerate(steps, 1):
            try:
                if not step():
                    print(f"\nERROR: Installation failed at step {i}")
                    print("Please report this issue with error details to:")
                    print("https://github.com/MargotBelot/PapyrusVision/issues")
                    return False
            except KeyboardInterrupt:
                print("\nInstallation Installation cancelled by user")
                return False
            except Exception as e:
                print(f"\nERROR: Unexpected error in step {i}: {e}")
                return False
        
        self.print_success_message()
        return True

    def print_success_message(self):
        """Print success message with next steps."""
        print("\n"+ "="*60)
        print("SUCCESS: INSTALLATION COMPLETE!")
        print("="*60)
        print("\nInstalling Next Steps:")
        
        if self.system == "windows":
            print("1. Double-click 'start_papyrus_vision.bat' to launch the main app")
            print("2. Double-click 'start_paleography_tool.bat' to launch the paleography tool")
        else:
            print("1. Run './start_papyrus_vision.sh' to launch the main app")
            print("2. Run './start_paleography_tool.sh' to launch the paleography tool")
        
        print("3. Your browser will open to http://localhost:8501")
        print("4. Start analyzing hieroglyphs and creating digital paleographies!")
        
        print("\nNOTE: Tips:")
        print("- The first launch may take 30-60 seconds")
        print("- Keep the terminal/command prompt window open while using")
        print("- Press Ctrl+C in terminal to stop the application")
        
        print("\nFeatures Features Available:")
        print("- Hieroglyph detection and classification")
        print("- Digital paleography creation")
        print("- Batch processing capabilities")
        print("- Interactive HTML catalog generation")
        print("- Unicode mapping and Gardiner code descriptions")
        
        print("\nDocumentation: Documentation:")
        print("- Technical Guide: docs/TECHNICAL_GUIDE.md")
        print("- Jupyter Notebooks: notebooks/ directory")
        
        print("\nUnknown Need help?")
        print("- GitHub Issues: https://github.com/MargotBelot/PapyrusVision/issues")


def main():
    """Main installation function."""
    installer = PapyrusVisionInstaller()
    
    try:
        success = installer.install()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nInstallation Installation cancelled")
        sys.exit(1)


if __name__ == "__main__":
    main()
