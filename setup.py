from setuptools import setup, find_packages

setup(
    name='Applied-Graph-Optimization-for-Forex-Market',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    author='Ilan Aliouchouche',
    install_requires=[
        'numpy==1.26.4',
        'pandas==2.2.2',
        'networkx==3.1',
        'matplotlib==3.8',
        'tqdm==4.66.3',
    ],
    python_requires='>=3.10',
)
