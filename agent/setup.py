from setuptools import setup, find_packages
from Cython.Build import cythonize

# Boilerplate for integrating with PyTest
from setuptools.command.test import test
import sys
import os

print("Checking for RDKit... ", end="")
try:
    from rdkit import Chem
except ImportError:
    print("ERROR!")
    print("ERROR: RDKit not found. It is required and must be installed seperately.")
    print("See: https://www.rdkit.org/docs/Install.html")
    try:
        import conda

        print("")
        print("Since you have conda installed, you might want to try:")
        print("  conda install -c conda-forge rdkit")
    except:
        pass
    exit(-1)
print("SUCCESS!")


class PyTest(test):
    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]

    def initialize_options(self):
        test.initialize_options(self)
        self.pytest_args = ""

    def run_tests(self):
        import shlex
        import pytest

        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


# The actual setup metadata
setup(
    name="keter",
    version="0.0.1",
    description="Highly scalable coronavirus science platform",
    long_description=open("README.rst").read(),
    keywords="machine_learning artificial_intelligence medicine devops",
    author="JJ Ben-Joseph",
    author_email="jbenjoseph@iqt.org",
    python_requires=">=3.6",
    url="https://www.github.com/bnext-iqt/keter",
    license="Apache",
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    install_requires=[
        "Cython",
        "gensim",
        "auto-sklearn",
        "protobuf<4.0.0",
        "jinja2<3.0.0",
        "betterproto[compiler]",
        "pyarrow",
        "selfies",
        "fire",
        "flair",
        "aioredis",
        "psutil",
        "aiohttp_cors",
        "redis",
        "aiohttp",
        "grpcio",
        "msgpack",
        "ray",
        "dvc[s3]",
        "flask",
        "deepchem",
        "tensorflow==2.3.*",
    ],
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "keter = keter.__main__:main",
        ],
    },
    ext_modules=cythonize("keter/operations.pyx", language_level="3"),
    cmdclass={"test": PyTest},
)
