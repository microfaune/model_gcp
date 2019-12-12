from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['librosa', 'google-cloud-storage']


setup(
    name='deploy_bird_detection',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    version='0.1',
    scripts=['predictor.py'])
