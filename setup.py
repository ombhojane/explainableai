from setuptools import setup, find_packages

setup(
    name='explainableai',
    version='0.1.2',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'shap',
        'matplotlib',
        'seaborn',
        'plotly',
        'ipywidgets',
    ],
    entry_points={
        'console_scripts': [
            # Define command-line scripts here if needed
            # "explain=explainableai:main",
        ],
    },
    author=['Om Bhojane', 'Palak Boricha'],
    author_email='ombhojane05@gmail.com',
    description='A package for Explainable AI',
    long_description=open('README.txt').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ombhojane/explainableai',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
