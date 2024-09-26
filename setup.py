from setuptools import setup, find_packages
import os

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='explainableai',
    version='0.1.6',
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
        'pillow'
    ],
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
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='explainable ai, machine learning, model interpretation, data science',
    python_requires='>=3.7',
    include_package_data=True,
    package_data={
        'explainableai': ['data/*.csv', 'templates/*.html'],
    },
)