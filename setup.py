from setuptools import setup, find_packages

setup(
    name='explainable_ai',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your package dependencies here
    ],
    entry_points={
        'console_scripts': [
            # Define command-line scripts here if needed
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for Explainable AI',
    long_description=open('README.txt').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/explainable_ai',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
