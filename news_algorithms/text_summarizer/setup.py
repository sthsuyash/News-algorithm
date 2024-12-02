from setuptools import setup, find_packages

setup(
    name='nepali_text_summarization',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'scikit-learn',
        'gensim',
    ],
    author='Suyash Shrestha',
    author_email='sthasuyash11@gmail.com',
    description='A package for Nepali language text summarization using Extractive approach.',
)
