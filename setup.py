from setuptools import setup
from pykospacing import __version__


setup(name='pykospacing',
      python_requires='==3.6.*',
      version=__version__,
      url='https://github.com/haven-jeon/PyKoSpacing',
      license='GPL-3',
      author='Heewon Jeon',
      author_email='madjakarta@gmail.com',
      description='Python package for automatic Korean word spacing.',
      packages=['pykospacing', ],
      long_description=open('README.md', encoding='utf-8').read(),
      zip_safe=False,
      include_package_data=True,

      install_requires=[
          'tensorflow >= 1.4.0, <= 1.6.0',
          'keras >= 2.1.5',
          'h5py >= 2.7.1',
          'argparse >= 1.4.0',
      ],

      entry_points={
          'console_scripts': [
              'pykos = pykospacing.pykos:main',
          ],
      },
      )
