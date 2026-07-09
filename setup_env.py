#!/usr/bin/env python3
import subprocess
import shutil
import sys
import os

def check_command_exists(command):
    """Check if a system command is available in the PATH."""
    return shutil.which(command) is not None

def run_command(args, env_name=None):
    """Run a command using subprocess. Optional conda run context."""
    if env_name:
        # Wrap command inside 'conda run -n <env_name>' to execute inside the specific environment
        args = ["conda", "run", "-n", env_name, "--no-capture-output"] + args
    
    print(f"Running: {' '.join(args)}")
    try:
        result = subprocess.run(args, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False

def check_conda_env_exists(env_name):
    """Check if a specific conda environment already exists."""
    try:
        output = subprocess.check_output(["conda", "env", "list"]).decode("utf-8")
        return env_name in output
    except Exception as e:
        print(f"Could not verify conda environments: {e}")
        return False

def check_nvidia_gpu():
    """Detect if an NVIDIA GPU and driver are available."""
    return check_command_exists("nvidia-smi")

def main():
    env_name = "surv_env"
    
    print("==================================================")
    # 1. Verify Conda is installed
    if not check_command_exists("conda"):
        print("[Error] 'conda' command not found. Please install Miniconda or Anaconda first.")
        print("Download link: https://docs.conda.io/en/latest/miniconda.html")
        sys.exit(1)
    print("[OK] Conda is installed.")

    # 2. Check/Create Conda Environment
    if check_conda_env_exists(env_name):
        print(f"[OK] Conda environment '{env_name}' already exists.")
    else:
        print(f"Creating conda environment '{env_name}' with Python 3.10...")
        create_cmd = ["conda", "create", "-n", env_name, "python=3.10", "-y"]
        if not run_command(create_cmd):
            print("[Error] Failed to create conda environment.")
            sys.exit(1)
        print(f"[OK] Conda environment '{env_name}' successfully created.")

    # 3. Detect GPU and Install PyTorch
    has_gpu = check_nvidia_gpu()
    if has_gpu:
        print("[OK] NVIDIA GPU detected. Installing PyTorch with CUDA 11.8 support...")
        # PyTorch installation command with CUDA
        pytorch_cmd = ["conda", "install", "pytorch", "torchvision", "torchaudio", "pytorch-cuda=11.8", "-c", "pytorch", "-c", "nvidia", "-y", "-n", env_name]
    else:
        print("[Info] No NVIDIA GPU detected. Installing CPU-only version of PyTorch...")
        # PyTorch installation command for CPU
        pytorch_cmd = ["conda", "install", "pytorch", "torchvision", "torchaudio", "cpuonly", "-c", "pytorch", "-y", "-n", env_name]

    if not run_command(pytorch_cmd):
        print("[Error] PyTorch installation failed. Trying alternative pip installation...")
        if has_gpu:
            pip_torch = ["pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"]
        else:
            pip_torch = ["pip", "install", "torch", "torchvision", "torchaudio"]
        
        if not run_command(pip_torch, env_name=env_name):
            print("[Error] Alternative PyTorch installation also failed.")
            sys.exit(1)

    print("[OK] PyTorch installed successfully.")

    # 4. Install other required packages via pip
    required_packages = [
        "opencv-python",
        "psutil",
        "websockets",
        "pillow",
        "transformers",
        "accelerate",
        "numpy"
    ]
    
    print(f"Installing project dependencies: {', '.join(required_packages)}...")
    pip_cmd = ["pip", "install"] + required_packages
    if not run_command(pip_cmd, env_name=env_name):
        print("[Error] Dependency installation failed.")
        sys.exit(1)

    print("==================================================")
    print("[SUCCESS] Environment and dependency setup completed!")
    print("\nTo start using the environment, run:")
    print(f"    conda activate {env_name}")
    print("\nTo run the surveillance system:")
    print("    python surv.py --source video.mp4")
    print("==================================================")

if __name__ == "__main__":
    main()
