from setuptools import setup, find_packages

__version__ = '0.0.5'

install_requires = ['gym', 'numpy', 'spheropy>=0.0.4']

packages = [package for package in find_packages() if package.startswith('gym_')]

setup(
    name='gym_sphero',
    version=__version__,
    install_requires=install_requires,
    packages=packages
)