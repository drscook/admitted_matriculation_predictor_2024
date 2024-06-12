import subprocess
import sys

def pcks_install():
 
    packages = [
        'dill',
        'python-dotenv', 
        'oracledb',
        'numpy==1.26.4',
        'pandas==2.2.1',
        'matplotlib',
        'ipython', 
        'codetiming',
        'miceforest==5.7.0',
        'flaml==2.1.2',
        'scikit-learn==1.4.1.post1',
        'tabulate',
        # 'hashlib'
        'xgboost',
        'openpyxl',
    ]

    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    pcks_install()
    print("All packages installed successfully.")