from setuptools import setup
from setuptools import find_packages

setup(name='ggc',
      version='0.0.1',
      description='Geometric Graph Construction',
      author='Yifan Qian',
      author_email='haczqyf@gmail.com',
      url='https://haczqyf.github.io/',
      download_url='https://github.com/haczqyf/ggc',
      license='MIT',
      install_requires=['numpy>=1.18.1',
                        'scikit-learn>=0.22',
                        'networkx>=2.4',
                        'scipy>=1.4.1',
                        'seaborn>=0.10.0',
                        'pandas>=1.0.2',
                        'matplotlib>=3.2.0'],
      package_data={'ggc': ['README.md']},
      packages=find_packages())
