from pathlib import Path
import os

from setuptools import find_packages, setup


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
os.chdir(THIS_DIR)

README_PATH = REPO_ROOT / "README.md"
README = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else "quant_system"

REQUIREMENTS = [
    "pandas>=2.1",
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "lightgbm>=4.0",
    "xgboost>=2.0",
    "hmmlearn>=0.3.0",
    "hdbscan>=0.8.33",
    "requests>=2.31",
    "python-dotenv>=1.0",
    "streamlit>=1.28",
    "altair>=5.2",
    "plotly>=5.18",
    "websockets>=11.0",
    "fastapi>=0.109",
    "uvicorn>=0.24",
    "jinja2>=3.1",
    "tqdm>=4.66",
    "psutil>=5.9",
    "matplotlib>=3.8",
    "seaborn>=0.13",
]

PACKAGE_DATA = {
    "quant_system": [
        "config/**/*.yaml",
        "config/*.yaml",
        "config/*.env",
        "replay_export/*.html",
        "replay_export/*.css",
        "replay_export/*.js",
        "dashboard/styles/*.css",
        "dashboard/components/js/**/*.js",
        "dashboard/components/js/**/*.css",
        "dashboard/components/js/**/*.html",
        "dashboard/components/execution_panel/*.js",
        "dashboard/components/execution_panel/*.css",
    ]
}


setup(
    name="quant_system",
    version="1.0.0",
    author="Nischal Neupane",
    description="Multi-timeframe quant trading framework with research, backtest, forward, live, and dashboard layers.",
    long_description=README,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(where="..", exclude=("tests", "notebooks", "research")),
    package_dir={"": ".."},
    package_data=PACKAGE_DATA,
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=REQUIREMENTS,
    entry_points={
        "console_scripts": [
            "quant-backtest=quant_system.cli.backtest_cli:main",
            "quant-forward=quant_system.cli.forward_cli:main",
            "quant-live=quant_system.cli.live_cli:main",
            "quant-train=quant_system.cli.train_cli:main",
            "quant-train-orchestrator=quant_system.train_orchestrator:main",
            "quant-terminal-api=quant_system.cli.terminal_api_cli:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    zip_safe=False,
)
