from setuptools import setup, find_packages

setup(
    name="max30102-gui",
    version="1.0.0",
    description="GUI for MAX30102 pulse sensor using PyQt5",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "PyQt5>=5.15",
        "matplotlib>=3.5",
        "numpy>=1.21",
        "scipy>=1.7",
        "smbus2>=0.4.2"
    ],
    entry_points={
        "console_scripts": [
            "max30102-gui=main:main"
        ]
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
