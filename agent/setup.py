from setuptools import setup, find_packages
from Cython.Build import cythonize

# Boilerplate for integrating with PyTest
from setuptools.command.test import test
import sys
import os

exit_due_to_failure = False

print("Checking for RDKit... ", end="")
try:
    from rdkit import Chem

    print("SUCCESS!")
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
    exit_due_to_failure = True

print("Checking for PyTorch... ", end="")
try:
    import torch

    print("SUCCESS!")
except ImportError:
    print("ERROR!")
    print("ERROR: PyTorch not found. It is required and must be installed seperately.")
    print("See: https://pytorch.org/get-started/locally/")
    try:
        import conda

        print("")
        print("Since you have conda installed, you might want to try:")
        print("  conda install pytorch torchvision torchaudio -c pytorch")
    except:
        pass
    exit_due_to_failure = True

print("Checking for Swig... ", end="")
from shutil import which
if which("swig"):
    print("SUCCESS!")
else:
    print("ERROR!")
    print("ERROR: Swig not found. It is required and must be installed seperately.")
    print("See: https://www.dev2qa.com/how-to-install-swig-on-macos-linux-and-windows/")
    try:
        import conda

        print("")
        print("Since you have conda installed, you might want to try:")
        print("  conda install -c anaconda swig")
    except:
        pass
    exit_due_to_failure = True

if exit_due_to_failure:
    exit(-1)


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


with open("requirements.txt") as fd:
    install_requires = fd.read().splitlines()

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
    install_requires=install_requires,
    tests_require=["pytest"],
    entry_points={"console_scripts": ["keter = keter.__main__:main",],},
    ext_modules=cythonize("keter/operations.pyx", language_level="3"),
    cmdclass={"test": PyTest},
)
