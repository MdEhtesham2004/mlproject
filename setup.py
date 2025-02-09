from setuptools import find_packages, setup
from typing import List

MINUS_E_DOT='-e .' 
def get_requirements(file_path:str)->List[str]:
    """this function will returns the list of requriements from the file"""
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n',"")  for req in requirements]
        if MINUS_E_DOT in requirements:
            requirements.remove(MINUS_E_DOT)
    return requirements

setup(
    name='mlproject',
    version = '0.1',
    author='Mohammed Ehtesham',
    author_email='ehteshammd089@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)

