import setuptools

## Following example from
## https://github.com/pypa/sampleproject/blob/master/setup.py

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepvirfinder",
    version="1.0.0",
    author="Jie Ren",
    author_email="renj@usc.edu",
    description="Identifying viruses from metagenomic data by deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # Using my fork for now
    url="https://github.com/papanikos/DeepVirFinder",
    keywords="machine learning, bioinformatics, metagenomics, viromics",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
                ],
    include_package_data=True,
    # works with python 3.6
    # 3.8 throws some errors so isolated env it is, for now
    python_requires='>=3.6',

    # The fine grained versions are the ones
    # used during development, installed with
    # $ conda create -n dvf python=3.6 biopython numpy theano keras scikit-learn
    install_requires=[
                    'biopython>=1.77',
                    'keras==2.3.1',
                    'numpy==1.17.0',
                    'scikit-learn==0.23.1',
                    'tensorflow==1.14.0',
                    'theano==1.0.4',
                    ],
    scripts=[
        'deepvirfinder/dvf.py',
        'deepvirfinder/encode.py',
        'deepvirfinder/training.py',
        ],

    # Could not make it work like this
    # TO DO
#    entry_points={
#        'console_scripts' : [
#           'dvf=dvf.py',
#            ],
#        }
    )

