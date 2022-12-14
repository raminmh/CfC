from setuptools import setup
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

    CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: BSD License
Programming Language :: C
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Topic :: Software Development :: Libraries
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX :: Linux
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""
    
setup(
    name='cfc_model',
    version='1.1.0',    
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
    license='Apache'

    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    py_modules=['cfc_model'],
)
