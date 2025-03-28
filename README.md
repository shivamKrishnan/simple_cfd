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
4. **OpenMP**: Required for parallelization in the C++ code (can check by running openmp_check.cpp).

---

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/shivamKrishnan/simple_cfd.git
   cd simple_cfd
   ```

 2. Create and Activate a Virtual Environment
   It is recommended to use a Python virtual environment (`venv`) to isolate dependencies.
   
   - **On Windows**:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

   - **On Linux/macOS**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. Install Dependencies
   Install Flask and any required dependencies inside the virtual environment:
   ```bash
   pip install flask
   ```

4. Directory Structure
   Ensure your project directory looks like this:
   ```
   📦 simple_cfd
   ├── 📝 input_params.txt
   ├── ⚖️ LICENSE
   ├🌀 navier_stokes_2d.cpp
   ├🌀 navier_stokes_3d.cpp
   ├🌀 openmp_check.cpp
   ├📖 README.md
   ├🐍 server.py
   ├📁 input
   ├📁 output
   ├📁 static
   │   ├📜 script.js
   │   └🎨 styles.css
   └📁 templates
      └🖥️ index.html
   ```

5. Compile the C++ Code
   The Flask backend will automatically compile the C++ code, but you can manually compile it using:

   ```bash
   g++ -fopenmp navier_stokes_2d.cpp -o navier_stokes -O2
   ```
   ```bash
   g++ -fopenmp navier_stokes_3d.cpp -o navier_stokes_3d -O2
   ```

---

## Running the Web Application

1. **Start the Flask Server**:
   Run the Flask server:
   ```bash
   python server.py
   ```

2. **Access the Web Interface**:
   Open your browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

3. **Input Simulation Parameters**:
   Fill out the form with the desired simulation settings:
   - Grid Resolution (nx, ny)
   - Reynolds Number
   - Time Step (dt)
   - Number of Steps
   - Plot Interval
   - Shape (e.g., Circle, Square, Airfoil, etc.)

4. **Run the Simulation**:
   Click the "Run Simulation" button. The backend will:
   - Save the inputs to `input_params.txt`.
   - Run the C++ simulation.
   - Generate `.vtk` files for each step.
   - Create a `.zip` file containing all results.

5. **Download Results**:
   After the simulation completes, the browser will automatically download the `simulation_results.zip` file.

---

## Example Inputs

Here’s an example of valid inputs for the simulation:

- **Grid Resolution**: `201 101`
- **Reynolds Number**: `200`
- **Time Step**: `0.004`
- **Number of Steps**: `10000`
- **Plot Interval**: `200`
- **Shape**: `CIRCLE`

---

## Output Files

The simulation generates the following files:

1. **Step Files**:
   - `step_<step_number>.vtk`: VTK files for each step, containing pressure, velocity, and mask fields.

2. **Summary Files**:
   - `summary_<step_number>.txt`: Summary statistics for each step (e.g., maximum velocity, average pressure).

3. **Overall Summary**:
   - `overall_summary.txt`: Aggregated statistics for the entire simulation.

4. **Zip File**:
   - `simulation_results.zip`: A compressed archive containing all step files and summaries.

---

## Customizing the Simulation

To modify the simulation (e.g., add new shapes or change the domain size), edit the following files:

1. **C++ Code**:
   - `navier_stokes_2d.cpp`: Modify the `setupMask()` function to add new shapes or change the simulation logic.

2. **Web Interface**:
   - `templates/index.html`: Modify the HTML form to add new input fields or change the layout.

3. **Backend**:
   - `server.py`: Modify the Flask backend to handle additional inputs or change the simulation execution logic.

---

## Troubleshooting

1. **C++ Compilation Errors**:
   - Ensure `g++` and OpenMP are installed.
   - Check for syntax errors in the C++ code.

2. **Flask Server Errors**:
   - Ensure Flask is installed (`pip install flask`).
   - Check the terminal for error messages when running `python server.py`.

3. **Simulation Errors**:
   - Ensure the input parameters are valid (e.g., positive numbers for grid resolution, time step, etc.).
   - Check the `output/` directory for generated files and logs.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- This project uses the **Navier-Stokes equations** to simulate fluid flow.
- The web interface is built using **Flask** and **HTML/JavaScript**.
- The C++ code is parallelized using **OpenMP**.

---

## Contact

For questions or feedback, please contact:
- **Shivam Krisnan**: [shivamkrishnan15@gmail.com](mailto:shivamkrishnan15@gmail.com)
- **GitHub**: [shivamKrishnan](https://github.com/shivamKrishnan)

---

Enjoy simulating fluid dynamics with this web-based interface! 🚀

