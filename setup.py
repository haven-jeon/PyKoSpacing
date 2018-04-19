from setuptools import setup, find_packages

 
setup(name='pykospacing',

      version='0.1',

      url='https://github.com/haven-jeon/PyKoSpacing',

      license='GPL-3',

      author='Heewon Jeon',

      author_email='madjakarta@gmail.com',

      description='Python package for automatic Korean word spacing.',

      packages=['pykospacing',],

      long_description=open('README.md', encoding='utf-8').read(),

      zip_safe=False,

      include_package_data=True,

      install_requires=['tensorflow>=1.4.0', 'keras>=2.1.5', 'h5py>=2.7.0'],
      )


