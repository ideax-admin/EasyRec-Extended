from setuptools import setup, find_packages

setup(
    name="easyrec-extended",
    version="0.1.0",
    description="Extended EasyRec framework with Policy-driven recommendation engine",
    author="IdeaX Business",
    author_email="dev@ideax-business.com",
    url="https://github.com/ideax-business/EasyRec-Extended",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "tensorflow>=2.5.0",
        "protobuf>=3.19.0",
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "pydantic>=1.8.0",
        "asyncio-contextmanager>=1.0.0",
        "grpcio>=1.40.0",
        "grpcio-tools>=1.40.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-asyncio>=0.18.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
