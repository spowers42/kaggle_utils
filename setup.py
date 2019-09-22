import setuptools

with open("README.md", 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="kaggle-utils-spp",
    version='0.0.1',
    author='Scott P. Powers',
    author_email='spowers.42@gmail.com',
    description='A small set of utilities for Kaggle competitions',
    url='',
    packages=setuptools.find_packages(),
    install_requires=[
        'pandas',
        'fastai',
        'tqdm',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
