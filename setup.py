from setuptools import setup, find_packages  # type: ignore

setup(
    name='Applied-Graph-Optimization-for-Forex-Market',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    author='Ilan Aliouchouche',
    install_requires=[
        'numpy==1.26.4',
        'pandas==2.2.2',
    ],
    python_requires='>=3.10',
)
