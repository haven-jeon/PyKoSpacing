from setuptools import setup


setup(name='pykospacing',
      python_requires='>=3.6',
      version=0.5,
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
          'tensorflow == 2.5.0',
          'h5py == 3.1.0',
          'argparse >= 1.4.0',
      ],

      entry_points={
          'console_scripts': [
              'pykos = pykospacing.pykos:main',
          ],
      },
      )
