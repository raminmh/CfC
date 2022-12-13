from setuptools import setup
import setuptools

from pathlib import Path
d = Path(__file__).parent
long_description = (d / "README.md").read_text()


setup(
    name='cfc_model',
    version='1.0.5',    
    description='An easy-to-use api for the closed-form continuous models in tensorflow and pytorch.',
    url='https://github.com/nightvision04/CfC',
    author='Daniel Scott',
    author_email='danscottlearns@gmail.com',
    install_requires=['pandas',
                      'numpy',                     
                      ],
    
    packages=setuptools.find_packages(),
    include_package_data=True,
    long_description=long_description,
    long_description_content_type='text/markdown',

    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    py_modules=['cfc_model'],
)
