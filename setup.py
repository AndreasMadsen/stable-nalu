
from setuptools import setup

setup(name='grumbel_nalu',
      version='0.1.0',
      description='Implementation of NALU with Grumbel Weight',
      url='https://github.com/AndreasMadsen/publication-grumbel-nalu',
      author='Andreas Madsen',
      author_email='amwebdk@gmail.com',
      license='MIT',
      packages=['grumbel_nalu'],
      install_requires=[
          'numpy',
          'tqdm',
          'torch',
          'scipy',
          'tensorboardX',
          'tensorboard'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)