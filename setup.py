# -*- coding: utf-8 -*-

"""
setup.py - Luke Bouma (bouma.luke@gmail.com) - Nov 2021
"""
__version__ = '0.0.0'

import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import shlex
        import pytest

        if not self.pytest_args:
            targs = []
        else:
            targs = shlex.split(self.pytest_args)

        errno = pytest.main(targs)
        sys.exit(errno)


def readme():
    with open('README.md') as f:
        return f.read()

INSTALL_REQUIRES = [
    'numpy>=1.4.0',
    'scipy',
    'astropy>=1.3',
    'matplotlib',
]


###############
## RUN SETUP ##
###############

# run setup.
setup(
    name='Notch_and_LoCoR',
    version=__version__,
    description=('Notch and LoCoR fork -- implementation by Aaron Rizzuto & the THYME team.'),
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Development Status :: 1 - Alpha',
        'License :: OSI Approved :: MIT License',
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    keywords='astronomy',
    url='https://github.com/arizzuto/Notch_and_LoCoR',
    author='Aaron Rizzuto',
    author_email='aaron.rizzuto23@gmail.com',
    license='MIT',
    packages=[
        'notch_and_locor',
    ],
    install_requires=INSTALL_REQUIRES,
    tests_require=['pytest==3.8.2',],
    cmdclass={'test':PyTest},
    include_package_data=True,
    zip_safe=False,
)
