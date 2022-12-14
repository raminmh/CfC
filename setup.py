from setuptools import setup
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    ]

setup(
    name='cfc_model',
    version='1.0.9',    
    description='An easy-to-use api for the closed-form continuous models in tensorflow',
    url='https://github.com/nightvision04/CfC',
    author='Daniel Scott',
    author_email='danscottlearns@gmail.com',
    install_requires=['pandas',
                      'numpy'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    
    packages=setuptools.find_packages(),
    include_package_data=True,

    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    py_modules=['cfc_model'],
)
