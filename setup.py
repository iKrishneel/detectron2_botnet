#! /usr/bin/env python

from setuptools import setup
from setuptools import find_packages


try:
    with open('README.md', 'r') as f:
        readme = f.read()
except Exception:
    readme = str('')


# write all package dependencies here
install_requires = [
    'bottleneck-transformer-pytorch',
    # 'detectron2',
    'coloredlogs',
    'numpy',
    'matplotlib',
    'opencv-python',
    'pillow',
    'torch',
    'torchvision',
    'tqdm',
    'scipy',
    'pytest',
]

setup(
    name='detectron2_botnet',
    version='0.0.1',
    long_description=readme,
    packages=find_packages(),
    zip_safe=False,
    install_requires=install_requires,
    test_suite='tests',
)
