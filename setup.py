
from setuptools import setup

setup(name='stable_nalu',
      version='0.1.0',
      description='Implementation of NALU with stable training',
      url='https://github.com/AndreasMadsen/publication-stable-nalu',
      author='Andreas Madsen',
      author_email='amwebdk@gmail.com',
      license='MIT',
      packages=['stable_nalu'],
      install_requires=[
          'numpy',
          'tqdm',
          'torch',
          'scipy',
          'pandas',
          'tensorflow',
          'torchvision',
          'tensorboard',
          'tensorboardX',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)