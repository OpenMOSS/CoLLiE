from setuptools import setup, find_packages

with open('requirements.txt', encoding='utf-8') as f:
    reqs = f.read()
    
setup(name='collie',
      version='1.0.0',
      description="CoLLiE: Collaborative Tuning of Large Language Models in an Efficient Way",
      author="OpenLMLab",
      author_email="yanghang@pjlab.org.cn",
      packages=find_packages(),
      install_requires=reqs.splitlines(),
      python_requires='>=3.8')