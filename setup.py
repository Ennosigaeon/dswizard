from setuptools import setup, find_packages

setup(
    name='dswizard',
    version='0.1.0',
    description='DataScience Wizard for automatic assembly of machine learning pipelines',
    author='Marc Zoeller',
    author_email='m.zoeller@usu.de',
    url="https://gitlab.usu-research.ml/research/automl/hpbandster",
    license='BSD 3-Clause License',
    classifiers=['Development Status :: 4 - Beta'],
    packages=find_packages(),
    python_requires='>=3',
    install_requires=['Pyro4', 'serpent', 'ConfigSpace', 'numpy', 'statsmodels', 'scipy', 'netifaces', 'scikit-learn',
                      'IPython', 'matplotlib', 'pandas'],
    extras_require={
        'docu': ['sphinx', 'sphinx_rtd_theme', 'sphinx_gallery'],
    },
    keywords=['distributed', 'optimization', 'multifidelity'],
    test_suite="tests"
)
