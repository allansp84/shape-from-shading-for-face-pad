from setuptools import setup, find_packages

setup(
    name='antispoofing.sfsnet',
    version=open('version.txt').read().rstrip(),
    url='',
    license='',
    author='',
    author_email='allansp84@gmail.com',
    description='',
    long_description=open('README.md').read(),

    packages=find_packages(where='antispoofing.sfsnet', exclude=['tests']),

    install_requires=open('requirements.txt').read().splitlines(),
    dependency_links=['git+https://github.com/davisking/dlib.git@c2a9dee846cad7b20998ee0f3fae79b6bcb67d43#egg=dlib'],

    entry_points={
        'console_scripts': [
            'sfsnet.py = antispoofing.sfsnet.scripts.sfsnet:main',
        ],
    },

)
