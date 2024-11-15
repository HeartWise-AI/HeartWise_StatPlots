from setuptools import setup, find_packages

setup(
    name='heartwise_statplots', 
    version='0.1.2',           
    author='Jacques Delfrate',
    author_email='jacques.delfrate@gmail.com',
    description='Package containing utilities for HeartWise projects',
    long_description=open('README.md').read(),  
    long_description_content_type='text/markdown',
    url='https://github.com/HeartWise-AI/HeartWise_StatPlots.git',  
    packages=find_packages(),  
    install_requires=[
        'numpy',
        'pandas',
        'seaborn',
        'scikit-learn',
        'pytest',
        'tqdm',
        'transformers',
        'pydicom'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  
)