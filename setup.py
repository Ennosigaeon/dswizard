from setuptools import setup, find_namespace_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='dswizard',
    version='0.1.0',
    description='DataScience Wizard for automatic assembly of machine learning pipelines',
    author='Marc Zoeller',
    author_email='m.zoeller@usu.de',
    url="https://gitlab.usu-research.ml/research/automl/dswizard",
    license='BSD 3-Clause License',
    classifiers=['Development Status :: 4 - Beta'],
    packages=find_namespace_packages(include=['dswizard.*']),
    python_requires='>=3.5',
    install_requires=requirements,
    keywords=['distributed', 'optimization', 'multifidelity'],
    test_suite="tests"
)
