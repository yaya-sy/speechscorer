from setuptools import setup

with open("README", 'r') as f:
    long_description = f.read()

setup(
   name='speechscorer',
   version='1.0',
   description='A useful module',
   license="MIT",
   long_description=long_description,
   author='Man Foo',
   author_email='yayasyscore@gmail.com',
   url="http://www.foopackage.example/",
   packages=['speechscorer'],  #same as name
   install_requires=['wheel', 'bar', 'greek'], #external packages as dependencies
    entry_points={'console_scripts': ['speechscore = speechscore.main:main']},
)
