#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <omp.h> // For OpenMP parallelization

class NavierStokesSolver
{
public:
    enum Shape
    {
        CIRCLE,
        SQUARE,
        AIRFOIL,
        CAR,
        DIAMOND,
        TRIANGLE,
        ELLIPSE,
        ROUNDED_RECTANGLE,
        STAR,
        HEXAGON,
        CRESCENT,
        HEART,
        CROSS,
        TRAPEZOID,
        PARABOLA,
        POLYGON,
    };
    // Getter methods for private member variables
    int getNy() const { return ny; }
    int getNx() const { return nx; }
    const std::vector<std::vector<int>> &getMask() const { return mask; }
    const std::vector<std::vector<double>> &getU() const { return u; }
    const std::vector<std::vector<double>> &getV() const { return v; }
    const std::vector<std::vector<double>> &getP() const { return p; }

private:
    // Domain parameters
    double Lx, Ly;       // Domain size
    double cx, cy, r;    // Shape center and radius (or size)
    int nx, ny;          // Number of grid points
    double dx, dy;       // Grid spacing
    double Re;           // Reynolds number
    double dt;           // Time step
    int max_iter;        // Maximum iterations for pressure solver
    double tol;          // Convergence tolerance
    Shape shape;         // Shape type
    bool use_les = true; // Enable/disable LES
    double cs = 0.1;     // Smagorinsky constant (typical value)

    // Fields
    std::vector<std::vector<double>> u;      // x-velocity
    std::vector<std::vector<double>> v;      // y-velocity
    std::vector<std::vector<double>> p;      // pressure
    std::vector<std::vector<double>> u_prev; // previous u
    std::vector<std::vector<double>> v_prev; // previous v
    std::vector<std::vector<int>> mask;      // Fluid/solid mask
    std::vector<double> x, y;                // Grid coordinates
    std::vector<std::vector<double>> nu_t;   // Turbulent eddy viscosity

    // Compute turbulent viscosity using Smagorinsky model
    void computeTurbulentViscosity()
    {
        // Resize turbulent viscosity field
        nu_t.resize(ny, std::vector<double>(nx, 0.0));

        // Compute strain rate tensor components and eddy viscosity
#pragma omp parallel for collapse(2)
        for (int i = 1; i < ny - 1; ++i)
        {
            for (int j = 1; j < nx - 1; ++j)
            {
                if (mask[i][j] == 1)
                {
                    // Strain rate tensor components
                    double S11 = (u[i][j + 1] - u[i][j - 1]) / (2 * dx);
                    double S22 = (v[i + 1][j] - v[i - 1][j]) / (2 * dy);

                    // Shear strain rate
                    double S12 = 0.5 * ((u[i + 1][j] - u[i - 1][j]) / (2 * dy) +
                                        (v[i][j + 1] - v[i][j - 1]) / (2 * dx));

                    // Magnitude of strain rate tensor
                    double S_mag = std::sqrt(2.0 * (S11 * S11 + S22 * S22 + 2 * S12 * S12));

                    // Filter width (grid spacing)
                    double delta = std::sqrt(dx * dy);

                    // Compute eddy viscosity (Smagorinsky model)
                    nu_t[i][j] = (cs * delta) * (cs * delta) * S_mag;
                }
            }
        }
    }

public:
    NavierStokesSolver(
        std::pair<double, double> domain_size = {10.0, 10.0},
        std::pair<double, double> shape_center = {5.0, 5.0},
        double shape_radius = 1.0,
        Shape shape_type = CIRCLE,
        int nx = 101, int ny = 101,
        double reynolds = 100.0,
        double dt = 0.01,
        int max_iter = 1000,
        double tol = 1e-5) : Lx(domain_size.first), Ly(domain_size.second),
                             cx(shape_center.first), cy(shape_center.second),
                             r(shape_radius), shape(shape_type), nx(nx), ny(ny), Re(reynolds),
                             dt(dt), max_iter(max_iter), tol(tol)
    {
        // Calculate grid spacing
        dx = Lx / (nx - 1);
        dy = Ly / (ny - 1);

        // Initialize grid coordinates
        x.resize(nx);
        y.resize(ny);
        for (int i = 0; i < nx; ++i)
            x[i] = i * dx;
        for (int j = 0; j < ny; ++j)
            y[j] = j * dy;

        // Initialize fields with proper dimensions
        u.resize(ny, std::vector<double>(nx, 0.0));
        v.resize(ny, std::vector<double>(nx, 0.0));
        p.resize(ny, std::vector<double>(nx, 0.0));
        u_prev.resize(ny, std::vector<double>(nx, 0.0));
        v_prev.resize(ny, std::vector<double>(nx, 0.0));
        mask.resize(ny, std::vector<int>(nx, 1));
        nu_t.resize(ny, std::vector<double>(nx, 0.0));

        // Set up the shape mask
        setupMask();

        // Set initial boundary conditions
        for (int i = 0; i < ny; ++i)
        {
            u[i][0] = 1.0; // Inlet velocity (uniform flow)
        }
    }

    // Add method to toggle LES
    void setLESMode(bool enable)
    {
        use_les = enable;
    }

    // Add method to set Smagorinsky constant
    void setSmagorinskyConstant(double constant)
    {
        cs = constant;
    }

    void setupMask()
    {
        // Create mask: 1 for fluid cells, 0 for solid cells (inside shape)
        for (int i = 0; i < ny; ++i)
        {
            for (int j = 0; j < nx; ++j)
            {
                double x_dist = x[j] - cx;
                double y_dist = y[i] - cy;

                switch (shape)
                {
                case CIRCLE:
                    if (x_dist * x_dist + y_dist * y_dist <= r * r)
                    {
                        mask[i][j] = 0;
                    }
                    break;
                case SQUARE:
                    if (std::abs(x_dist) <= r && std::abs(y_dist) <= r)
                    {
                        mask[i][j] = 0;
                    }
                    break;
                case AIRFOIL:
                    // Simple airfoil shape (NACA 0012 profile)
                    if (x_dist >= -r && x_dist <= r)
                    {
                        double t = 0.12; // Thickness
                        double yt = 5 * t * (0.2969 * std::sqrt(std::abs(x_dist) / r) - 0.1260 * (x_dist / r) - 0.3516 * std::pow(x_dist / r, 2) + 0.2843 * std::pow(x_dist / r, 3) - 0.1015 * std::pow(x_dist / r, 4));
                        if (std::abs(y_dist) <= yt)
                        {
                            mask[i][j] = 0;
                        }
                    }
                    break;
                case CAR:
                {
                    // Car dimensions (relative to radius r)
                    double car_length = 2.0 * r;
                    double car_height = r;
                    double hood_length = 0.5 * r;
                    double cabin_length = 0.7 * r;
                    double trunk_length = 0.8 * r;
                    double ground_clearance = 0.15 * r;

                    // Basic car shape components
                    double car_body_bottom = cy - car_height / 2 + ground_clearance;
                    double car_body_top = cy + car_height / 2;
                    double car_front = cx - car_length / 2;
                    double car_back = cx + car_length / 2;
                    double hood_end = car_front + hood_length;
                    double windshield_start = hood_end;
                    double cabin_end = windshield_start + cabin_length;
                    double trunk_start = cabin_end;

                    // Check if point is inside car body
                    if (x_dist >= -car_length / 2 && x_dist <= car_length / 2)
                    {
                        double upper_boundary;
                        double lower_boundary = car_body_bottom;

                        // Calculate the upper boundary based on car section
                        double x_pos = x[j];

                        if (x_pos <= hood_end)
                        {
                            // Hood (flat)
                            upper_boundary = cy;
                        }
                        else if (x_pos <= cabin_end)
                        {
                            // Cabin with curved windshield and roof
                            double t = (x_pos - windshield_start) / cabin_length;

                            // Create curve for windshield and roof
                            // Using a simple quadratic curve for windshield
                            if (t < 0.3)
                            {
                                // Windshield section (rising)
                                double curve_t = t / 0.3;
                                upper_boundary = cy + (car_height / 2 - car_height / 4) * curve_t * curve_t;
                            }
                            else
                            {
                                // Roof section (slightly curved)
                                upper_boundary = car_body_top;
                            }
                        }
                        else
                        {
                            // Trunk (sloping down slightly)
                            double t = (x_pos - trunk_start) / trunk_length;
                            upper_boundary = car_body_top - (car_height / 6) * t;
                        }

                        // Add wheels (circles)
                        double front_wheel_x = car_front + r * 0.3;
                        double rear_wheel_x = car_back - r * 0.3;
                        double wheel_y = car_body_bottom - r * 0.1;
                        double wheel_radius = r * 0.2;

                        bool in_front_wheel = (pow(x[j] - front_wheel_x, 2) + pow(y[i] - wheel_y, 2)) <= pow(wheel_radius, 2);
                        bool in_rear_wheel = (pow(x[j] - rear_wheel_x, 2) + pow(y[i] - wheel_y, 2)) <= pow(wheel_radius, 2);

                        // Check if point is inside main car body or wheels
                        if ((y[i] >= lower_boundary && y[i] <= upper_boundary) || in_front_wheel || in_rear_wheel)
                        {
                            mask[i][j] = 0;
                        }
                    }
                    break;
                }
                case DIAMOND:
                {
                    // Diamond shape: a rotated square
                    double diamond_size = r * std::sqrt(2); // Diagonal length
                    if (std::abs(x_dist) + std::abs(y_dist) <= diamond_size)
                    {
                        mask[i][j] = 0;
                    }
                    break;
                }
                case TRIANGLE:
                {
                    // Triangle shape: equilateral triangle pointing upwards
                    double triangle_height = r * std::sqrt(3); // Height of the triangle
                    double half_base = r;                      // Half of the base length

                    // Check if the point is inside the triangle
                    if (y_dist <= triangle_height && std::abs(x_dist) <= half_base * (1 - y_dist / triangle_height))
                    {
                        mask[i][j] = 0;
                    }
                    break;
                }
                case ELLIPSE:
                {
                    double a = r;     // Semi-major axis
                    double b = r / 2; // Semi-minor axis
                    if ((x_dist * x_dist) / (a * a) + (y_dist * y_dist) / (b * b) <= 1.0)
                    {
                        mask[i][j] = 0;
                    }
                    break;
                }
                case ROUNDED_RECTANGLE:
                {
                    double width = 2.0 * r;
                    double height = r;
                    double corner_radius = 0.2 * r;

                    // Check if the point is inside the main rectangle (excluding corners)
                    if (std::abs(x_dist) <= width / 2 - corner_radius && std::abs(y_dist) <= height / 2 - corner_radius)
                    {
                        mask[i][j] = 0;
                    }
                    // Check if the point is inside one of the rounded corners
                    else if (std::abs(x_dist) > width / 2 - corner_radius && std::abs(y_dist) > height / 2 - corner_radius)
                    {
                        double corner_x = std::abs(x_dist) - (width / 2 - corner_radius);
                        double corner_y = std::abs(y_dist) - (height / 2 - corner_radius);
                        if (corner_x * corner_x + corner_y * corner_y <= corner_radius * corner_radius)
                        {
                            mask[i][j] = 0;
                        }
                    }
                    break;
                }
                case STAR:
                {
                    double outer_radius = r;
                    double inner_radius = r / 2;
                    int num_points = 5; // Number of star points

                    double angle = std::atan2(y_dist, x_dist);
                    double radius = std::sqrt(x_dist * x_dist + y_dist * y_dist);

                    // Star equation: alternating between outer and inner radii
                    double star_radius = outer_radius * inner_radius /
                                         (std::sqrt(outer_radius * outer_radius * std::pow(std::cos(angle * num_points / 2), 2) +
                                                    inner_radius * inner_radius * std::pow(std::sin(angle * num_points / 2), 2)));

                    if (radius <= star_radius)
                    {
                        mask[i][j] = 0;
                    }
                    break;
                }
                case HEXAGON:
                {
                    double side_length = r;                // Side length of the hexagon
                    double height = side_length * sqrt(3); // Height of the hexagon

                    // Check if the point is inside the hexagon
                    if (std::abs(x_dist) <= side_length / 2 &&
                        std::abs(y_dist) <= (height / 2) &&
                        std::abs(y_dist) * 2 + std::abs(x_dist) * sqrt(3) <= height)
                    {
                        mask[i][j] = 0;
                    }
                    break;
                }
                case CRESCENT:
                {
                    double outer_radius = r;
                    double inner_radius = 0.7 * r;
                    double offset = 0.5 * r; // Offset of the inner circle

                    // Check if the point is inside the outer circle but outside the inner circle
                    if (x_dist * x_dist + y_dist * y_dist <= outer_radius * outer_radius &&
                        (x_dist - offset) * (x_dist - offset) + y_dist * y_dist > inner_radius * inner_radius)
                    {
                        mask[i][j] = 0;
                    }
                    break;
                }
                case HEART:
                {
                    double heart_radius = r;
                    double x_norm = x_dist / heart_radius;
                    double y_norm = y_dist / heart_radius;

                    // Heart equation: (x^2 + y^2 - 1)^3 - x^2 * y^3 <= 0
                    if (std::pow(x_norm * x_norm + y_norm * y_norm - 1, 3) -
                            x_norm * x_norm * std::pow(y_norm, 3) <=
                        0)
                    {
                        mask[i][j] = 0;
                    }
                    break;
                }
                case CROSS:
                {
                    double arm_width = 0.3 * r;
                    double arm_length = r;

                    // Check if the point is inside the vertical or horizontal arm
                    if ((std::abs(x_dist) <= arm_width && std::abs(y_dist) <= arm_length) ||
                        (std::abs(y_dist) <= arm_width && std::abs(x_dist) <= arm_length))
                    {
                        mask[i][j] = 0;
                    }
                    break;
                }
                case TRAPEZOID:
                {
                    double top_width = r;
                    double bottom_width = 1.5 * r;
                    double height = r;

                    // Check if the point is inside the trapezoid
                    double slope = (bottom_width - top_width) / (2 * height);
                    if (std::abs(y_dist) <= height / 2 &&
                        std::abs(x_dist) <= top_width / 2 + slope * (height / 2 - y_dist))
                    {
                        mask[i][j] = 0;
                    }
                    break;
                }
                case PARABOLA:
                {
                    double a = 0.1; // Controls the width of the parabola
                    if (y_dist >= a * x_dist * x_dist - r)
                    {
                        mask[i][j] = 0;
                    }
                    break;
                }
                case POLYGON:
                {
                    std::vector<std::pair<double, double>> vertices = {
                        {cx - r, cy - r},
                        {cx + r, cy - r},
                        {cx + r, cy + r},
                        {cx - r, cy + r}}; // Example: square

                    bool inside = false;
                    for (size_t k = 0, l = vertices.size() - 1; k < vertices.size(); l = k++)
                    {
                        if (((vertices[k].second > y[i]) != (vertices[l].second > y[i])) &&
                            (x[j] < (vertices[l].first - vertices[k].first) * (y[i] - vertices[k].second) /
                                            (vertices[l].second - vertices[k].second) +
                                        vertices[k].first))
                        {
                            inside = !inside;
                        }
                    }
                    if (inside)
                    {
                        mask[i][j] = 0;
                    }
                    break;
                }
                }
            }
        }
    }

    std::vector<std::vector<double>> pressurePoisson()
    {
        // Create the right-hand side of the pressure equation
        std::vector<std::vector<double>> b(ny, std::vector<double>(nx, 0.0));

        // Compute source term for pressure Poisson equation
#pragma omp parallel for collapse(2)
        for (int i = 1; i < ny - 1; ++i)
        {
            for (int j = 1; j < nx - 1; ++j)
            {
                if (mask[i][j] == 1)
                { // Only for fluid cells
                    b[i][j] = ((u[i][j + 1] - u[i][j - 1]) / (2 * dx) +
                               (v[i + 1][j] - v[i - 1][j]) / (2 * dy)) /
                              dt;
                }
            }
        }

        // Iterative solution (Jacobi method with relaxation)
        std::vector<std::vector<double>> p_next = p;
        double error;

        for (int k = 0; k < max_iter; ++k)
        {
            std::vector<std::vector<double>> p_old = p_next;
            error = 0.0;

            double relaxation = 0.8; // Relaxation factor (0.5-0.8 is often stable)

#pragma omp parallel for collapse(2) reduction(max : error)
            for (int i = 1; i < ny - 1; ++i)
            {
                for (int j = 1; j < nx - 1; ++j)
                {
                    if (mask[i][j] == 1)
                    { // Only update fluid cells
                        // Count number of fluid neighbors
                        double n_fluid = mask[i + 1][j] + mask[i - 1][j] +
                                         mask[i][j + 1] + mask[i][j - 1];

                        if (n_fluid > 0)
                        { // At least one fluid neighbor
                            double p_new = ((p_old[i + 1][j] * mask[i + 1][j] +
                                             p_old[i - 1][j] * mask[i - 1][j]) /
                                                (dx * dx) +
                                            (p_old[i][j + 1] * mask[i][j + 1] +
                                             p_old[i][j - 1] * mask[i][j - 1]) /
                                                (dy * dy) -
                                            b[i][j]) /
                                           (n_fluid * (1 / (dx * dx) + 1 / (dy * dy)));

                            // Apply relaxation
                            p_next[i][j] = relaxation * p_new + (1 - relaxation) * p_old[i][j];

                            // Check for extremely large values and limit them
                            if (std::isnan(p_next[i][j]))
                            {
                                p_next[i][j] = 0.0; // Reset to zero if NaN
                            }
                            else if (p_next[i][j] > 1e10)
                            {
                                p_next[i][j] = 1e10; // Cap extremely large positive values
                            }
                            else if (p_next[i][j] < -1e10)
                            {
                                p_next[i][j] = -1e10; // Cap extremely large negative values
                            }

                            // Compute local error for convergence check
                            double local_error = std::abs(p_next[i][j] - p_old[i][j]);
                            error = std::max(error, local_error);
                        }
                    }
                }
            }

            // Apply boundary conditions for pressure
#pragma omp parallel for
            for (int i = 0; i < ny; ++i)
            {
                p_next[i][0] = p_next[i][1];           // Left boundary
                p_next[i][nx - 1] = p_next[i][nx - 2]; // Right boundary
            }

#pragma omp parallel for
            for (int j = 0; j < nx; ++j)
            {
                p_next[0][j] = p_next[1][j];           // Bottom boundary
                p_next[ny - 1][j] = p_next[ny - 2][j]; // Top boundary
            }

            // Check convergence
            if (error < tol)
            {
                std::cout << "Pressure solver converged in " << k + 1 << " iterations" << std::endl;
                break;
            }

            if (k == max_iter - 1)
            {
                std::cout << "Warning: Pressure solver did not converge. Final error: " << error << std::endl;
            }
        }

        return p_next;
    }

    void solveStep()
    {
        // Compute turbulent viscosity if LES is enabled
        if (use_les)
        {
            computeTurbulentViscosity();
        }

        // Store previous velocities
        u_prev = u;
        v_prev = v;

        // Intermediate velocity fields
        std::vector<std::vector<double>> u_temp = u;
        std::vector<std::vector<double>> v_temp = v;

        // Calculate intermediate velocity (with turbulent viscosity)
#pragma omp parallel for collapse(2)
        for (int i = 1; i < ny - 1; ++i)
        {
            for (int j = 1; j < nx - 1; ++j)
            {
                if (mask[i][j] == 1)
                {
                    // Effective viscosity (laminar + turbulent)
                    double nu_eff = (use_les ? 1.0 / Re + nu_t[i][j] : 1.0 / Re);

                    // x-momentum equation with turbulent viscosity
                    u_temp[i][j] = u[i][j] + dt * (
                                                      // Viscous terms (with effective viscosity)
                                                      nu_eff * ((u[i + 1][j] - 2 * u[i][j] + u[i - 1][j]) / (dy * dy) +
                                                                (u[i][j + 1] - 2 * u[i][j] + u[i][j - 1]) / (dx * dx)) -
                                                      // Convective terms
                                                      u[i][j] * (u[i][j + 1] - u[i][j - 1]) / (2 * dx) -
                                                      v[i][j] * (u[i + 1][j] - u[i - 1][j]) / (2 * dy));

                    // y-momentum equation with turbulent viscosity
                    v_temp[i][j] = v[i][j] + dt * (
                                                      // Viscous terms (with effective viscosity)
                                                      nu_eff * ((v[i + 1][j] - 2 * v[i][j] + v[i - 1][j]) / (dy * dy) +
                                                                (v[i][j + 1] - 2 * v[i][j] + v[i][j - 1]) / (dx * dx)) -
                                                      // Convective terms
                                                      u[i][j] * (v[i][j + 1] - v[i][j - 1]) / (2 * dx) -
                                                      v[i][j] * (v[i + 1][j] - v[i - 1][j]) / (2 * dy));
                }
            }
        }

        // Solve pressure Poisson equation
        p = pressurePoisson();

        // Correct velocities with pressure gradient
#pragma omp parallel for collapse(2)
        for (int i = 1; i < ny - 1; ++i)
        {
            for (int j = 1; j < nx - 1; ++j)
            {
                if (mask[i][j] == 1)
                { // Only update fluid cells
                    u[i][j] = u_temp[i][j] - dt * (p[i][j + 1] - p[i][j - 1]) / (2 * dx);
                    v[i][j] = v_temp[i][j] - dt * (p[i + 1][j] - p[i - 1][j]) / (2 * dy);
                }
            }
        }

        // Apply boundary conditions

        // Left boundary (inlet): fixed uniform velocity
#pragma omp parallel for
        for (int i = 0; i < ny; ++i)
        {
            u[i][0] = 1.0;
            v[i][0] = 0.0;
        }

        // Right boundary (outlet): zero gradient
#pragma omp parallel for
        for (int i = 0; i < ny; ++i)
        {
            u[i][nx - 1] = u[i][nx - 2];
            v[i][nx - 1] = v[i][nx - 2];
        }

        // Top and bottom boundaries: no-slip
#pragma omp parallel for
        for (int j = 0; j < nx; ++j)
        {
            u[0][j] = 0.0;
            v[0][j] = 0.0;
            u[ny - 1][j] = 0.0;
            v[ny - 1][j] = 0.0;
        }

        // Shape boundary: enforce zero velocity inside and on the shape
#pragma omp parallel for collapse(2)
        for (int i = 0; i < ny; ++i)
        {
            for (int j = 0; j < nx; ++j)
            {
                if (mask[i][j] == 0)
                {
                    u[i][j] = 0.0;
                    v[i][j] = 0.0;
                }
            }
        }
    }

    void saveToVTK(int step)
    {
        // Create a VTK file for the current step
        std::string filename = "output/step_" + std::to_string(step) + ".vtk";
        std::ofstream file(filename);

        if (!file.is_open())
        {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }

        // VTK header
        file << "# vtk DataFile Version 3.0" << std::endl;
        file << "Navier-Stokes solution at step " << step << std::endl;
        file << "ASCII" << std::endl;
        file << "DATASET STRUCTURED_GRID" << std::endl;
        file << "DIMENSIONS " << nx << " " << ny << " 1" << std::endl;
        file << "POINTS " << nx * ny << " float" << std::endl;

        // Write point coordinates
        for (int i = 0; i < ny; ++i)
        {
            for (int j = 0; j < nx; ++j)
            {
                file << std::fixed << std::setprecision(6) << x[j] << " " << y[i] << " 0.0" << std::endl;
            }
        }

        // Write pressure field
        file << "POINT_DATA " << nx * ny << std::endl;
        file << "SCALARS pressure float 1" << std::endl;
        file << "LOOKUP_TABLE default" << std::endl;
        for (int i = 0; i < ny; ++i)
        {
            for (int j = 0; j < nx; ++j)
            {
                file << std::fixed << std::setprecision(6) << p[i][j] << std::endl;
            }
        }

        // Write velocity field as vectors
        file << "VECTORS velocity float" << std::endl;
        for (int i = 0; i < ny; ++i)
        {
            for (int j = 0; j < nx; ++j)
            {
                file << std::fixed << std::setprecision(6) << u[i][j] << " " << v[i][j] << " 0.0" << std::endl;
            }
        }

        // Write mask field
        file << "SCALARS mask float 1" << std::endl;
        file << "LOOKUP_TABLE default" << std::endl;
        for (int i = 0; i < ny; ++i)
        {
            for (int j = 0; j < nx; ++j)
            {
                file << std::fixed << std::setprecision(6) << static_cast<float>(mask[i][j]) << std::endl;
            }
        }

        // Write turbulent viscosity field if LES is enabled
        if (use_les)
        {
            file << "SCALARS turbulent_viscosity float 1" << std::endl;
            file << "LOOKUP_TABLE default" << std::endl;
            for (int i = 0; i < ny; ++i)
            {
                for (int j = 0; j < nx; ++j)
                {
                    file << std::fixed << std::setprecision(6) << nu_t[i][j] << std::endl;
                }
            }
        }

        file.close();
    }

    void saveSummary(int step, double elapsed_time)
    {
        std::ofstream file("output/summary_" + std::to_string(step) + ".txt");
        if (file.is_open())
        {
            // Calculate some flow statistics
            double max_u = 0.0, max_v = 0.0, avg_p = 0.0;
            int fluid_cells = 0;

            for (int i = 0; i < ny; ++i)
            {
                for (int j = 0; j < nx; ++j)
                {
                    if (mask[i][j] == 1)
                    {
                        max_u = std::max(max_u, std::abs(u[i][j]));
                        max_v = std::max(max_v, std::abs(v[i][j]));
                        avg_p += p[i][j];
                        fluid_cells++;
                    }
                }
            }

            if (fluid_cells > 0)
            {
                avg_p /= fluid_cells;
            }

            // Write summary statistics
            file << "Simulation Summary for Step " << step << std::endl;
            file << "-------------------------------" << std::endl;
            file << "Domain size: " << Lx << " x " << Ly << std::endl;
            file << "Grid resolution: " << nx << " x " << ny << std::endl;
            file << "Reynolds number: " << Re << std::endl;
            file << "Time step: " << dt << std::endl;
            file << "Current time: " << step * dt << std::endl;
            file << "Elapsed computation time: " << elapsed_time << " seconds" << std::endl;
            file << "Maximum u velocity: " << max_u << std::endl;
            file << "Maximum v velocity: " << max_v << std::endl;
            file << "Average pressure: " << avg_p << std::endl;
            file << "Fluid cells: " << fluid_cells << " (" << (double)fluid_cells / (nx * ny) * 100.0 << "% of domain)" << std::endl;
            file << "LES Model: " << (use_les ? "Enabled" : "Disabled") << std::endl;
            if (use_les)
            {
                file << "Smagorinsky Constant: " << cs << std::endl;
            }

            file.close();
        }
        else
        {
            std::cerr << "Error: Could not open summary file." << std::endl;
        }
    }

    void simulate(int num_steps = 1000, int plot_interval = 100)
    {
        // Create output directory if it doesn't exist
#ifdef _WIN32
        system("mkdir output 2> nul");
#else
        system("mkdir -p output");
#endif

        // Save initial state
        saveToVTK(0);

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int step = 1; step <= num_steps; ++step)
        {
            solveStep();

            // Save results at specified intervals
            if (step % plot_interval == 0 || step == num_steps)
            {
                auto current_time = std::chrono::high_resolution_clock::now();
                double elapsed_seconds = std::chrono::duration<double>(current_time - start_time).count();

                std::cout << "Step " << step << "/" << num_steps
                          << " (Time: " << elapsed_seconds << " s)" << std::endl;

                saveToVTK(step);
                saveSummary(step, elapsed_seconds);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(end_time - start_time).count();
        std::cout << "Simulation completed in " << total_time << " seconds" << std::endl;
    }

    double computeDragCoefficient()
    {
        double drag_force = 0.0;

        // Sum pressure forces on the shape (only consider cells adjacent to the shape)
        for (int i = 1; i < ny - 1; ++i)
        {
            for (int j = 1; j < nx - 1; ++j)
            {
                // If this is a fluid cell and at least one neighbor is a solid cell
                if (mask[i][j] == 1 && (mask[i + 1][j] == 0 || mask[i - 1][j] == 0 ||
                                        mask[i][j + 1] == 0 || mask[i][j - 1] == 0))
                {

                    // Calculate normal vector for this boundary cell
                    double nx_local = 0.0, ny_local = 0.0;
                    if (mask[i + 1][j] == 0)
                        ny_local += 1.0;
                    if (mask[i - 1][j] == 0)
                        ny_local -= 1.0;
                    if (mask[i][j + 1] == 0)
                        nx_local += 1.0;
                    if (mask[i][j - 1] == 0)
                        nx_local -= 1.0;

                    // Normalize
                    double norm = std::sqrt(nx_local * nx_local + ny_local * ny_local);
                    if (norm > 0)
                    {
                        nx_local /= norm;
                        ny_local /= norm;
                    }

                    // Add pressure force in x-direction
                    drag_force += p[i][j] * nx_local * dy;
                }
            }
        }

        // Compute drag coefficient
        // Cd = 2*F_drag / (density * U^2 * D)
        // We use density = 1 and U = 1 for simplicity
        double drag_coefficient = 2.0 * drag_force / (1.0 * 1.0 * 2.0 * r);

        return drag_coefficient;
    }
};

void displayShapes()
{
    std::cout << "Available shapes:\n";
    std::cout << "1. Circle\n";
    std::cout << "2. Square\n";
    std::cout << "3. Airfoil\n";
    std::cout << "4. Car\n";
    std::cout << "5. Diamond\n";
    std::cout << "6. Triangle\n";
    std::cout << "7. Ellipse\n";
    std::cout << "8. Rounded Rectangle\n";
    std::cout << "9. Star\n";
    std::cout << "10. Hexagon\n";
    std::cout << "11. Crescent\n";
    std::cout << "12. Heart\n";
    std::cout << "13. Cross\n";
    std::cout << "14. Trapezoid\n";
    std::cout << "15. Parabola\n";
    std::cout << "16. Polygon\n";
}

// Function to get user input for simulation parameters
void getUserInput(double &domainX, double &domainY, double &shapeRadius,
                  int &nx, int &ny, double &reynolds, double &dt,
                  int &numSteps, int &plotInterval, bool &useLES, double &smagorinskyConstant)
{
    std::cout << "Enter domain size (Lx Ly): ";
    std::cin >> domainX >> domainY;

    std::cout << "Enter shape radius (or size): ";
    std::cin >> shapeRadius;

    std::cout << "Enter grid resolution (nx ny): ";
    std::cin >> nx >> ny;

    std::cout << "Enter Reynolds number: ";
    std::cin >> reynolds;

    std::cout << "Enter time step (dt): ";
    std::cin >> dt;

    std::cout << "Enter number of time steps: ";
    std::cin >> numSteps;

    std::cout << "Enter plot interval: ";
    std::cin >> plotInterval;

    std::cout << "Use Large Eddy Simulation (LES)? (1 for Yes, 0 for No): ";
    std::cin >> useLES;

    if (useLES)
    {
        std::cout << "Enter Smagorinsky constant (default 0.1): ";
        std::cin >> smagorinskyConstant;
    }
}

// Function to display user settings in the console
void displayUserSettings(int shapeChoice, double domainX, double domainY, double shapeRadius,
                         int nx, int ny, double reynolds, double dt,
                         int numSteps, int plotInterval,
                         bool useLES, double smagorinskyConstant)
{
    std::cout << "\nUser Settings:\n";
    std::cout << "-------------------------------\n";
    std::cout << "Shape: " << shapeChoice << "\n";
    std::cout << "Domain size: " << domainX << " x " << domainY << "\n";
    std::cout << "Shape radius: " << shapeRadius << "\n";
    std::cout << "Grid resolution: " << nx << " x " << ny << "\n";
    std::cout << "Reynolds number: " << reynolds << "\n";
    std::cout << "Time step: " << dt << "\n";
    std::cout << "Number of time steps: " << numSteps << "\n";
    std::cout << "Plot interval: " << plotInterval << "\n";
    std::cout << "LES Turbulence Model: " << (useLES ? "Enabled" : "Disabled") << "\n";
    if (useLES)
    {
        std::cout << "Smagorinsky Constant: " << smagorinskyConstant << "\n";
    }
    std::cout << "-------------------------------\n";
}

// Function to generate an overall summary of the simulation
void generateOverallSummary(int numSteps, double totalTime, double finalDragCoefficient,
                            double maxU, double maxV, double avgPressure, int fluidCells,
                            bool useLES, double smagorinskyConstant)
{
    std::ofstream summaryFile("output/overall_summary.txt");
    if (!summaryFile.is_open())
    {
        std::cerr << "Error: Could not create overall summary file." << std::endl;
        return;
    }

    summaryFile << "Overall Simulation Summary\n";
    summaryFile << "-------------------------------\n";
    summaryFile << "Total number of steps: " << numSteps << "\n";
    summaryFile << "Total computation time: " << totalTime << " seconds\n";
    summaryFile << "Final drag coefficient: " << finalDragCoefficient << "\n";
    summaryFile << "Maximum u velocity: " << maxU << "\n";
    summaryFile << "Maximum v velocity: " << maxV << "\n";
    summaryFile << "Average pressure: " << avgPressure << "\n";
    summaryFile << "Fluid cells: " << fluidCells << "\n";
    summaryFile << "LES Model: " << (useLES ? "Enabled" : "Disabled") << "\n";
    if (useLES)
    {
        summaryFile << "Smagorinsky Constant: " << smagorinskyConstant << "\n";
    }
    summaryFile << "-------------------------------\n";

    summaryFile.close();
}

int main(int argc, char *argv[])
{
    // Read input parameters from input_params.txt
    std::ifstream input_file("input_params.txt");
    if (!input_file.is_open())
    {
        std::cerr << "Error: Could not open input_params.txt" << std::endl;
        return 1;
    }

    double domainX, domainY, shapeRadius, reynolds, dt, smagorinskyConstant;
    int nx, ny, numSteps, plotInterval;
    bool useLES;
    std::string shapeStr;

    input_file >> domainX >> domainY;
    input_file >> shapeRadius;
    input_file >> nx >> ny;
    input_file >> reynolds;
    input_file >> dt;
    input_file >> numSteps;
    input_file >> plotInterval;
    input_file >> shapeStr;
    input_file >> useLES;
    input_file >> smagorinskyConstant;

    input_file.close();

    // Map shape string to enum
    NavierStokesSolver::Shape shape;
    if (shapeStr == "CIRCLE")
        shape = NavierStokesSolver::CIRCLE;
    else if (shapeStr == "SQUARE")
        shape = NavierStokesSolver::SQUARE;
    else if (shapeStr == "AIRFOIL")
        shape = NavierStokesSolver::AIRFOIL;
    else if (shapeStr == "CAR")
        shape = NavierStokesSolver::CAR;
    else if (shapeStr == "DIAMOND")
        shape = NavierStokesSolver::DIAMOND;
    else if (shapeStr == "TRIANGLE")
        shape = NavierStokesSolver::TRIANGLE;
    else if (shapeStr == "ELLIPSE")
        shape = NavierStokesSolver::ELLIPSE;
    else if (shapeStr == "ROUNDED_RECTANGLE")
        shape = NavierStokesSolver::ROUNDED_RECTANGLE;
    else if (shapeStr == "STAR")
        shape = NavierStokesSolver::STAR;
    else if (shapeStr == "HEXAGON")
        shape = NavierStokesSolver::HEXAGON;
    else if (shapeStr == "CRESCENT")
        shape = NavierStokesSolver::CRESCENT;
    else if (shapeStr == "HEART")
        shape = NavierStokesSolver::HEART;
    else if (shapeStr == "CROSS")
        shape = NavierStokesSolver::CROSS;
    else if (shapeStr == "TRAPEZOID")
        shape = NavierStokesSolver::TRAPEZOID;
    else if (shapeStr == "PARABOLA")
        shape = NavierStokesSolver::PARABOLA;
    else if (shapeStr == "POLYGON")
        shape = NavierStokesSolver::POLYGON;
    else
    {
        std::cerr << "Invalid shape in input_params.txt" << std::endl;
        return 1;
    }

    // Define shape center (1/4 of the domain from the left, middle vertically)
    std::pair<double, double> shapeCenter = {domainX / 4, domainY / 2};

    // Create and run the solver
    NavierStokesSolver solver(
        {domainX, domainY},
        shapeCenter,
        shapeRadius,
        shape,
        nx, ny,
        reynolds,
        dt);

    // Configure LES if enabled
    if (useLES)
    {
        solver.setLESMode(true);
        solver.setSmagorinskyConstant(smagorinskyConstant);
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    solver.simulate(numSteps, plotInterval);
    auto end_time = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double>(end_time - start_time).count();

    // Calculate and print drag coefficient
    double cd = solver.computeDragCoefficient();
    std::cout << "Drag coefficient: " << cd << std::endl;

    // Generate overall summary
    double maxU = 0.0, maxV = 0.0, avgPressure = 0.0;
    int fluidCells = 0;

    // Calculate final flow statistics
    for (int i = 0; i < solver.getNy(); ++i)
    {
        for (int j = 0; j < solver.getNx(); ++j)
        {
            if (solver.getMask()[i][j] == 1)
            {
                maxU = std::max(maxU, std::abs(solver.getU()[i][j]));
                maxV = std::max(maxV, std::abs(solver.getV()[i][j]));
                avgPressure += solver.getP()[i][j];
                fluidCells++;
            }
        }
    }
    if (fluidCells > 0)
    {
        avgPressure /= fluidCells;
    }

    // Generate overall summary
    generateOverallSummary(numSteps, totalTime, cd, maxU, maxV, avgPressure, fluidCells, useLES, smagorinskyConstant);

    return 0;
}