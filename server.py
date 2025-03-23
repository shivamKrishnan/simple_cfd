from flask import Flask, request, send_file, render_template
import subprocess
import os
import zipfile

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run_simulation", methods=["POST"])
def run_simulation():
    # Get user inputs from the form
    nx = int(request.form["nx"])
    ny = int(request.form["ny"])
    reynolds = float(request.form["reynolds"])
    dt = float(request.form["dt"])
    num_steps = int(request.form["num_steps"])
    plot_interval = int(request.form["plot_interval"])
    shape = request.form["shape"]

    # Run the C++ simulation
    try:
        # Compile and run the C++ code
        subprocess.run(["g++", "-fopenmp", "navier_stokes_2d.cpp", "-o", "navier_stokes", "-O2"], check=True)
        subprocess.run(["./navier_stokes", str(nx), str(ny), str(reynolds), str(dt), str(num_steps), str(plot_interval), shape], check=True)
    except subprocess.CalledProcessError as e:
        return f"Error running simulation: {e}", 500

    # Create a zip file of the output directory
    zip_filename = "simulation_results.zip"
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        for root, dirs, files in os.walk("output"):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), "output"))

    # Provide the zip file for download
    return send_file(zip_filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)