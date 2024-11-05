from setuptools import setup, find_packages

setup(
    name='heartwise_statplots',  # Replace with your package name
    version='0.1.0',           # Initial version number
    author='Jacques Delfrate',
    author_email='jacques.delfrate@gmail.com',
    description='Package containing utilities for HeartWise projects',
    long_description=open('README.md').read(),  # Use your README.md as the long description
    long_description_content_type='text/markdown',
    url='https://github.com/HeartWise-AI/HeartWise_StatPlots.git',  # Replace with your GitHub repo URL
    packages=find_packages(),  # Automatically find packages in your project
    install_requires=[
        'numpy',
        'pandas',
        'seaborn',
        'scikit-learn',
        'pytest',
        'tqdm',
    ],
    classifiers=[
        # Choose appropriate classifiers from https://pypi.org/classifiers/
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Replace if using a different license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify your Python version compatibility
)