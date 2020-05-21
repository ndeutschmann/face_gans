from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='A short description of the project.',
    author='nicolas deutschmann',
    license='MIT',
    entry_points='''
        [console_scripts]
        dcg32-invert=src.models.train_model.dcgan32.command_line:dcgan32_inversion
    ''',
)
