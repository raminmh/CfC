from setuptools import setup
import setuptools

setup(
    name='cfc_model',
    version='1.0.2',    
    description='An easy-to-use api for the closed-form continuous models in tensorflow and pytorch.',
    url='https://github.com/nightvision04/CfC',
    author='Daniel Scott',
    author_email='danscottlearns@gmail.com',
    install_requires=['pandas',
                      'numpy',                     
                      ],
    
    packages=setuptools.find_packages(),
    include_package_data=True,

    classifiers=[
        'Programming Language :: Python :: 3.9',
    ],
    py_modules=['cfc_model'],
)
