from setuptools import setup, find_packages
from pathlib import Path

# Read long description
long_description = Path("README.md").read_text(encoding="utf-8")

# Try reading the requirements file
try:
    requirements = Path("requirements.txt").read_text(encoding="utf-8").splitlines()
except FileNotFoundError:
    requirements = []

setup(
    name='pyOASIS',
    version='1.0.3',
    author='Giorgio Picanço',
    author_email='giorgiopicanco@gmail.com',
    description='Open-Access System for Ionospheric Studies (OASIS)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/giorgiopicanco/OASIS',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'pyOASIS': ['glonass_channels.dat'],
    },
    install_requires=requirements,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
    ],
    python_requires='>=3.8',
)
