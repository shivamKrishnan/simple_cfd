from flask import Flask, request, send_file, render_template
import subprocess
import os
import zipfile
import shutil  # Import shutil for directory operations

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run_simulation", methods=["POST"])
def run_simulation():
    # Get user inputs from the form
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

    # Write input parameters to input_params.txt
    with open("input_params.txt", "w") as f:
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
        shutil.rmtree(output_dir)  # Remove the entire directory
    os.makedirs(output_dir)  # Recreate the directory

    # Run the C++ simulation
    try:
        # Compile and run the C++ code
        subprocess.run(["g++", "-fopenmp", "navier_stokes_2d.cpp", "-o", "navier_stokes", "-O2"], check=True)
        subprocess.run(["./navier_stokes"], check=True)
    except subprocess.CalledProcessError as e:
        return f"Error running simulation: {e}", 500

    # Create a zip file of the output directory
    zip_filename = "simulation_results.zip"
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), output_dir))

    # Provide the zip file for download
    return send_file(zip_filename, as_attachment=True)

def create_blank_input_params():
    # Create a blank input_params.txt file if it doesn't exist
    if not os.path.exists("input_params.txt"):
        open("input_params.txt", "w").close()

if __name__ == "__main__":
    # Create a blank input_params.txt if it doesn't exist
    create_blank_input_params()
    app.run(debug=True)