from setuptools import setup, find_packages
import os

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='explainableai',
    version='0.10',
    packages=find_packages(exclude=['tests*']),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'shap',
        'matplotlib',
        'seaborn',
        'plotly',
        'ipywidgets',
        'lime',
        'reportlab',
        'google-generativeai',
        'python-dotenv',
        'scipy',
        'pillow',
        'colorama',
        'dask'
    ],
        extras_require={
        'dev': [
            'pytest',
            'flake8',
            'black',
            'mypy',
        ],
    },
    entry_points={
        'console_scripts': [
            'explainableai=explainableai.main:main',
        ],
    },
    author='Om Bhojane, Palak Boricha',
    author_email='ombhojane05@gmail.com',
    description='A comprehensive package for Explainable AI and model interpretation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ombhojane/explainableai',
    project_urls={
        'Bug Tracker': 'https://github.com/ombhojane/explainableai/issues',
        'Source Code': 'https://github.com/ombhojane/explainableai',
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='explainableai, explainable ai, interpretable ml, model interpretability, '
              'feature importance, shap, lime, model explanation, ai transparency, '
              'machine learning, deep learning, artificial intelligence, data science, '
              'model insights, feature analysis, model debugging, ai ethics, '
              'responsible ai, xai, model visualization',
    python_requires='>=3.7',
    include_package_data=True,
    package_data={
        'explainableai': ['data/*.csv', 'templates/*.html'],
    },
)