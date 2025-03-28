from flask import Flask, request, send_file, render_template, jsonify
import subprocess
import os
import zipfile
import shutil

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run_simulation", methods=["POST"])
def run_simulation():
    # Get user inputs from the form
    simulation_type = request.form["simulationType"]
    domainX = float(request.form["domainX"])
    domainY = float(request.form["domainY"])
    shapeRadius = float(request.form["shapeRadius"])
    nx = int(request.form["nx"])
    ny = int(request.form["ny"])
    reynolds = float(request.form["reynolds"])
    dt = float(request.form["dt"])
    num_steps = int(request.form["num_steps"])
    plot_interval = int(request.form["plot_interval"])
    shape = request.form["shape"]

    # Handle STL file upload for 3D simulations
    if simulation_type == "3D":
        stl_file = request.files["stlFile"]
        if stl_file:
            stl_file.save("input.stl")  # Save the uploaded STL file
        domainZ = domainY  # Assuming a cubic domain for simplicity
        nz = ny  # Same resolution in Z as Y
    else:
        domainZ = 0  # Not used for 2D
        nz = 0  # Not used for 2D

    # Write input parameters to input_params.txt
    with open("input_params.txt", "w") as f:
        # Modify this section to handle 2D and 3D cases differently
        if simulation_type == "3D":
            f.write(f"{domainX} {domainY} {domainZ}\n")
            f.write(f"{shapeRadius}\n")
            f.write(f"{nx} {ny} {nz}\n")
        else:
            # For 2D, only write the actual X and Y values
            f.write(f"{domainX} {domainY}\n")
            f.write(f"{shapeRadius}\n")
            f.write(f"{nx} {ny}\n")
        
        f.write(f"{reynolds}\n")
        f.write(f"{dt}\n")
        f.write(f"{num_steps}\n")
        f.write(f"{plot_interval}\n")
        f.write(f"{shape}\n")

    # Clear the output directory before running the simulation
    output_dir = "output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Prepare the summary file
    summary_path = os.path.join(output_dir, "overall_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Simulation Details:\n")
        f.write(f"Dimension: {simulation_type}\n")
        f.write(f"Domain Size: {domainX} x {domainY}")
        if simulation_type == "3D":
            f.write(f" x {domainZ}")
        f.write("\n")
        f.write(f"Shape: {shape}\n")
        f.write(f"Shape Radius/Size: {shapeRadius}\n")
        f.write(f"Grid Resolution: {nx} x {ny}")
        if simulation_type == "3D":
            f.write(f" x {nz}")
        f.write("\n")
        f.write(f"Reynolds Number: {reynolds}\n")
        f.write(f"Time Step (dt): {dt}\n")
        f.write(f"Total Simulation Steps: {num_steps}\n")
        f.write(f"Plot Interval: {plot_interval}\n")

    # Run the C++ simulation
    try:
        # Compile and run the C++ code
        if simulation_type == "3D":
            subprocess.run(["g++", "-fopenmp", "navier_stokes_3d.cpp", "-o", "navier_stokes", "-O2"], check=True)
        else:
            subprocess.run(["g++", "-fopenmp", "navier_stokes_2d.cpp", "-o", "navier_stokes", "-O2"], check=True)
        
        # Run the simulation
        subprocess.run(["./navier_stokes"], check=True)

        # Append completion status
        with open(summary_path, "a") as f:
            f.write("\nSimulation Status: Completed Successfully")

    except subprocess.CalledProcessError as e:
        # Write error to summary file
        with open(summary_path, "a") as f:
            f.write(f"\nSimulation Status: Failed")

    # Create a zip file of the output directory with compression
    zip_filename = "simulation_results.zip"
    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                zipf.write(os.path.join(root, file), 
                           os.path.relpath(os.path.join(root, file), output_dir))

    # Provide the zip file for download
    return send_file(zip_filename, as_attachment=True)

@app.route("/get_summary")
def get_summary():
    # Read and return the contents of overall_summary.txt
    try:
        with open("output/overall_summary.txt", "r") as f:
            summary = f.read()
        return summary
    except FileNotFoundError:
        return "Summary not available. Run a simulation first.", 404

if __name__ == "__main__":
    app.run(debug=True)