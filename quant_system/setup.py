from setuptools import setup, find_packages
from pathlib import Path

# Read long description
this_dir = Path(__file__).parent
readme = (this_dir / "README.md").read_text(encoding="utf-8")

# Core dependencies
requirements = [
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

setup(
    name="quant_system",
    version="1.0.0",
    author="Your Name",
    author_email="you@example.com",
    description="Institutional-Grade Multi-Asset AI Quant Trading Framework",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/your/repo",  # adjust
    packages=find_packages(exclude=("tests", "notebooks", "research")),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "quant-backtest=quant_system.cli.backtest_cli:main",
            "quant-forward=quant_system.cli.forward_cli:main",
            "quant-live=quant_system.cli.live_cli:main",
            "quant-train=quant_system.cli.train_cli:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    zip_safe=False,
)
