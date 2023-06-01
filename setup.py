from setuptools import setup, find_packages

with open('requirements.txt', encoding='utf-8') as f:
    reqs = f.read()
    
setup(name='collie',
      version='0.1.1',
      packages=find_packages())