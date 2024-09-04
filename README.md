
# Drill Path Optimizer

The Drill Path Optimizer is a Python application designed to find the optimal path for drilling operations using various optimization algorithms such as Nearest Neighbor, 2-opt, 3-opt, and Linear Programming with Gurobi. It provides an intuitive GUI for selecting input files and running optimizations.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Algorithms and Methods](#algorithms-and-methods)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Nearest Neighbor Heuristic**: A greedy algorithm that constructs an initial feasible solution by choosing the closest unvisited point.
- **2-opt and 3-opt Local Search**: Algorithms that iteratively improve the initial solution by reversing segments of the tour to reduce the overall distance.
- **Integration with Gurobi Optimizer**: Uses Gurobi for finding optimal solutions via Linear Programming techniques.
- **Graphical User Interface (GUI)**: A user-friendly interface built with `tkinter` that allows users to select directories, specify time limits, and start the optimization process with ease.
- **Real-Time Status Updates**: Provides real-time updates and displays results using a status panel and a results table.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/Project1-DrillPathOptimizer.git
   cd Project1-DrillPathOptimizer
   ```

2. **Install the required dependencies**:

   Use `pip` to install the necessary Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

   Ensure you have the following installed:
   - Python 3.x
   - `gurobipy`
   - `matplotlib`
   - `networkx`
   - `Pillow`

3. **Set up Gurobi**:

   Install Gurobi and set up a license by following the instructions on the [Gurobi website](https://www.gurobi.com/documentation/).

## Usage

1. **Run the GUI application**:

   Start the main application by running the following command:

   ```bash
   python scripts/final_solution.py
   ```

2. **Choose a directory**:

   Use the "Choose Directory" button in the GUI to select a directory containing input files with point coordinates.

3. **Enter the time limit**:

   Specify the available time in minutes for the optimization process.

4. **Start the optimization**:

   Click the "Optimize" button to begin the optimization. The application will display real-time updates and results in the GUI.

## Project Structure

```markdown
├── README.md                   # Project overview and documentation
├── LICENSE                     # Licensing information
├── data/                       # Example input data or instructions for obtaining data
├── scripts/                    # Python scripts for the project
├── results/                    # Generated results or output
└── assets/                     # Images or assets for documentation
```

## Algorithms and Methods

- **Nearest Neighbor**: Constructs an initial tour by repeatedly visiting the nearest unvisited point.
- **2-opt and 3-opt**: Local search algorithms that improve the initial solution by reversing segments of the tour.
- **Linear Programming with Gurobi**: Solves the optimization problem using LP techniques to find the most optimal path.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Gurobi Optimization](https://www.gurobi.com/)
- Other libraries or resources
