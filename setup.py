from setuptools import setup
import os, platform

def requirements():
    if platform.system() == 'Darwin':
        return []

    def open_req(req_file):
        with open(os.path.join(os.path.dirname(os.path.abspath("__file__")), req_file)) as f:
            return f.read().splitlines()

    return open_req('requirements.txt')

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

      install_requires=requirements(),

      entry_points={
          'console_scripts': [
              'pykos = pykospacing.pykos:main',
          ],
      },
      )
