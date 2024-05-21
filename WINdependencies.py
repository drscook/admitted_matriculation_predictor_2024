import subprocess
import sys

def pcks_install():
 
    packages = [
        'dill',
        'python-dotenv', 
        'oracledb',
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'ipython', 
        'codetiming',
        'miceforest',
        'flaml',
        'scikit-learn==1.4.1.post1',
        # 'hashlib'
    ]

    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    pcks_install()
    print("All packages installed successfully.")