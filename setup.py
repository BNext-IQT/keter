from setuptools import setup, find_packages

# Boilerplate for integrating with PyTest
from setuptools.command.test import test
import sys
import os


class PyTest(test):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        test.initialize_options(self)
        self.pytest_args = ''

    def run_tests(self):
        import shlex
        import pytest
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)

# The actual setup metadata
setup(
    name='keter',
    version='0.0.1',
    description='Bioinformatics platform',
    long_description=open("README.rst").read(),
    keywords='machine_learning artificial_intelligence medicine devops',
    author='JJ Ben-Joseph',
    author_email='jbenjoseph@iqt.org',
    python_requires='>=3.6',
    url='https://www.github.com/bnext-iqt/keter',
    license='Apache',
    classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Operating System :: OS Independent'
    ],
    packages=find_packages(),
    install_requires=['rq', 'redis'],
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'keter = keter.__main__:main',
        ],
    },
    cmdclass={'test': PyTest}
)