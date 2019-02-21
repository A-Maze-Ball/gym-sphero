from setuptools import setup

__version__ = '0.0.1'

install_requires = ['gym', 'numpy', 'spheropy>=0.0.3']

setup(
    name='gym_sphero',
    version=__version__,
    install_requires=install_requires
)