from setuptools import setup, find_packages

setup(
    name='Applied-Graph-Optimization-for-Forex-Market',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    author='Ilan Aliouchouche',
    install_requires=[
        'numpy',
        'pandas',
    ],
    python_requires='>=3.10',
)
