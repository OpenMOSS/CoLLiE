import os

from setuptools import setup, find_packages

with open('requirements.txt', encoding='utf-8') as f:
    reqs = f.read()
    
setup(name='collie',
      version='0.0.1',
      packages=find_packages(),
      install_requires=reqs.strip().split('\n'))