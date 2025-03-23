# simple_cfd

# Navier-Stokes Simulation Web Interface

This project provides a **web-based interface** for running a Navier-Stokes fluid dynamics simulation. Users can input simulation parameters (e.g., grid resolution, Reynolds number, shape, etc.) through a web form. The backend runs the simulation using a C++ program, generates `.vtk` files for each step, and provides a downloadable `.zip` file containing all results.

---

## Features

- **Web Interface**: Users can input simulation settings via a simple HTML form.
- **Simulation Execution**: The backend runs a C++ Navier-Stokes solver with the provided inputs.
- **Results Download**: The simulation results (`.vtk` files and summaries) are bundled into a `.zip` file for easy download.
- **Customizable Inputs**: Users can specify grid resolution, Reynolds number, time step, number of steps, plot interval, and shape.

---

## Prerequisites

Before running the project, ensure you have the following installed:

1. **Python 3.x**: Required for the Flask backend.
2. **Flask**: Install using `pip install flask`.
3. **C++ Compiler**: Ensure `g++` is installed to compile the C++ code.
4. **OpenMP**: Required for parallelization in the C++ code.

---

## Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>