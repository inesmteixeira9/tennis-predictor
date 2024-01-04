from setuptools import find_packages, setup

setup(
    name='tennis-predictor',
    packages=find_packages(),
    version='1.0.0',
    description='Tennis betting decision-making model that gives you a recommendation on betting based on the potential profit. The model uses betting odds, player rankings and surface as features to make predictions on the match outcome. The given prediction, alongside with the current betting odds, are used to provide a recomendation on betting.',
    author='ines',
    license='',
    url='https://github.com/inesmteixeira9/tennis-predictor',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
