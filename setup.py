from setuptools import setup
import os
import io

DESCRIPTION = "User can create a Machine Learning model without coding"
URL = 'https://github.com/vijaytakbhate2002/ML-Builder.git'
EMAIL = 'vijaytakbhate20@gmail.com'
AUTHOR = 'Vijay Dipak Takbhate'
REQUIRES_PYTHON = '>=3.12.4'

cwd = os.path.abspath(os.path.dirname(__file__))

def list_reqs(fname='requirements.txt'):
    try:
        with io.open(os.path.join(cwd, fname), encoding='utf-8') as f:
            return f.read().splitlines()
    except FileNotFoundError:
        pass

with open('VERSION', 'r') as f:
    version = f.readline()

try:
    with io.open(os.path.join(cwd, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name="ML Builder",
    version = version,
    description = DESCRIPTION,
    long_description= long_description,
    long_description_content_type= 'str',
    author= AUTHOR,
    author_email= EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    package_data={'ML-Builder': ['VERSION']},
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],    
)

