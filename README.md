# Profitable Path Detection in Currency Graph

## Project Description
This project is designed to evaluate the performance of different algorithms for detecting profitable paths in currency graphs. By modeling a currency exchange graph, the project aims to identify arbitrage opportunities by maximizing profits through cycles in the graph.

## Project Structure
- `src/optimal_paths.py`: the module containing each algorithm
- `src/algo_benchmark.py`: the code of the benchmark, executable with a `csv` as output (located in `app/bench_app`)
- `app/bench_app/dashboard.py`: launches a dashboard to view the metrics of each algorithm's performance (with the output `csv` of `algo_benchmark.py`)
- `tests/`: includes unit tests for verifying the functionality of each module

## Usage Guide

### Cloning the Project and install dependencies
To clone the project, use the following command:
```bash
git clone https://github.com/ilanaliouchouche/Profitable-Path-Detection-in-Currency-Graphs.git

cd Profitable-Path-Detection-in-Currency-Graphs

pip install -e .
```

### Launching the Dashboard
To view the metrics dashboard run:
```bash
python app/bench_app/dashboard.py
```

### Testing and Visualizing an algorithm
To see a path research execution example with an interactive, you can modify the main function in `src/optimal_paths.py` to specify the desired algorithm and run (you can run it directly):
```bash
python src/optimal_paths.py
```

### Running Test
Unit tests are located in the `tests/` directory and can be executed to validate code functionality:
```bash
python -m unittest discover tests
```

### CI/CD Actions
This project includes two GitHub Actions:
1. Code quality check with `flake8`
2. Unit tests automatic execution

**(See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for more information)**

### Contributors

- Ilan Aliouchouche
- David Ribeiro
- Jiren Ren

