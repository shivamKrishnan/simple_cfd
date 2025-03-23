#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <string>
#include <tuple>
#include <limits>
#include <functional>
#include <cstring>
#include <sstream>
#include <cstdint>

// Define M_PI if not already defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Structure to represent a triangle in the STL file
struct Triangle
{
    double v1[3], v2[3], v3[3]; // Vertices of the triangle
};

// Improved STL Parser Function (supports both binary and ASCII formats)
std::vector<Triangle> readSTL(const std::string &filename)
{
    std::vector<Triangle> triangles;
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return triangles;
    }

    // Try to determine if the file is ASCII or binary
    char header[80] = {0};
    file.read(header, 80);

    // Check if file begins with "solid" (ASCII STL)
    bool isASCII = (strncmp(header, "solid", 5) == 0);

    // Return to beginning of file
    file.seekg(0, std::ios::beg);

    if (isASCII)
    {
        // Process ASCII STL file
        std::string line;
        int vertexCount = 0;
        Triangle currentTriangle;

        while (std::getline(file, line))
        {
            // Trim whitespace from the beginning of the line
            line.erase(0, line.find_first_not_of(" \t"));

            if (line.substr(0, 6) == "vertex")
            {
                // Extract vertex coordinates
                double x, y, z;
                std::istringstream iss(line.substr(6));
                if (!(iss >> x >> y >> z))
                {
                    std::cerr << "Error: Failed to parse vertex coordinates" << std::endl;
                    continue;
                }

                // Assign to the appropriate vertex in the triangle
                if (vertexCount % 3 == 0)
                    std::copy(std::initializer_list<double>{x, y, z}.begin(), std::initializer_list<double>{x, y, z}.end(), currentTriangle.v1);
                else if (vertexCount % 3 == 1)
                    std::copy(std::initializer_list<double>{x, y, z}.begin(), std::initializer_list<double>{x, y, z}.end(), currentTriangle.v2);
                else
                {
                    std::copy(std::initializer_list<double>{x, y, z}.begin(), std::initializer_list<double>{x, y, z}.end(), currentTriangle.v3);
                    triangles.push_back(currentTriangle);
                }
                vertexCount++;
            }
            else if (line.substr(0, 10) == "endsolid")
            {
                // End of solid, break if we've found any triangles
                if (!triangles.empty())
                    break;
            }
        }
    }
    else
    {
        // Process binary STL file
        file.seekg(80, std::ios::beg); // Skip header

        uint32_t triangleCount;
        file.read(reinterpret_cast<char *>(&triangleCount), sizeof(uint32_t));

        // Reserve space for all triangles
        triangles.reserve(triangleCount);

        // Binary STL format: normal (12 bytes) + 3 vertices (36 bytes) + attribute (2 bytes) = 50 bytes per triangle
        for (uint32_t i = 0; i < triangleCount; i++)
        {
            float normal[3];
            file.read(reinterpret_cast<char *>(normal), 12); // Skip normal vector

            Triangle tri;
            file.read(reinterpret_cast<char *>(tri.v1), 12);
            file.read(reinterpret_cast<char *>(tri.v2), 12);
            file.read(reinterpret_cast<char *>(tri.v3), 12);

            // Skip attribute byte count
            uint16_t attributeByteCount;
            file.read(reinterpret_cast<char *>(&attributeByteCount), 2);

            triangles.push_back(tri);
        }
    }

    file.close();

    std::cout << "Loaded " << triangles.size() << " triangles from STL file" << std::endl;

    return triangles;
}

// Function to calculate the bounding box of the STL model
void calculateBoundingBox(const std::vector<Triangle> &triangles, double &min_x, double &max_x, double &min_y, double &max_y, double &min_z, double &max_z)
{
    min_x = min_y = min_z = std::numeric_limits<double>::max();
    max_x = max_y = max_z = std::numeric_limits<double>::lowest();

    for (const auto &tri : triangles)
    {
        min_x = std::min(min_x, std::min(tri.v1[0], std::min(tri.v2[0], tri.v3[0])));
        max_x = std::max(max_x, std::max(tri.v1[0], std::max(tri.v2[0], tri.v3[0])));
        min_y = std::min(min_y, std::min(tri.v1[1], std::min(tri.v2[1], tri.v3[1])));
        max_y = std::max(max_y, std::max(tri.v1[1], std::max(tri.v2[1], tri.v3[1])));
        min_z = std::min(min_z, std::min(tri.v1[2], std::min(tri.v2[2], tri.v3[2])));
        max_z = std::max(max_z, std::max(tri.v1[2], std::max(tri.v2[2], tri.v3[2])));
    }

    // Debug: Print the bounding box
    std::cout << "Bounding Box: (" << min_x << ", " << min_y << ", " << min_z << ") to ("
              << max_x << ", " << max_y << ", " << max_z << ")" << std::endl;
}

// Enhanced method to translate and scale the STL model with validation
void translateAndScaleSTL(std::vector<Triangle> &triangles, double translate_x, double translate_y, double translate_z, double scale_factor)
{
    // Validate scale factor (prevent zero or negative scaling)
    if (scale_factor <= 0.0)
    {
        std::cerr << "Warning: Invalid scale factor (" << scale_factor << "). Using absolute value." << std::endl;
        scale_factor = std::abs(scale_factor);
        if (scale_factor < 1e-6)
            scale_factor = 1e-6;
    }

    // Process each triangle
    for (auto &tri : triangles)
    {
        // Process each vertex
        for (int i = 0; i < 3; ++i)
        {
            double *vertex = (i == 0) ? tri.v1 : (i == 1) ? tri.v2
                                                          : tri.v3;

            // Translate and scale coordinates
            vertex[0] = (vertex[0] + translate_x) * scale_factor;
            vertex[1] = (vertex[1] + translate_y) * scale_factor;
            vertex[2] = (vertex[2] + translate_z) * scale_factor;
        }
    }

    std::cout << "STL model translated by (" << translate_x << ", " << translate_y << ", "
              << translate_z << ") and scaled by " << scale_factor << std::endl;
}

// Enhanced method to check if a point is inside the solid (ray casting algorithm)
bool pointInSolid(const double p[3], const std::vector<Triangle> &triangles)
{
    // Early return if no triangles
    if (triangles.empty())
    {
        return false;
    }

    // Use a more robust direction for the ray
    const double ray[3] = {1.0, 0.1, 0.1}; // Slightly off-axis to avoid edge cases
    int intersections = 0;
    double eps = 1e-8;

    for (const auto &tri : triangles)
    {
        // Edge vectors
        double edge1[3] = {tri.v2[0] - tri.v1[0], tri.v2[1] - tri.v1[1], tri.v2[2] - tri.v1[2]};
        double edge2[3] = {tri.v3[0] - tri.v1[0], tri.v3[1] - tri.v1[1], tri.v3[2] - tri.v1[2]};

        // Begin Möller–Trumbore algorithm
        double h[3], s[3], q[3];
        double a, f, u, v, t;

        // Calculate h = ray × edge2
        h[0] = ray[1] * edge2[2] - ray[2] * edge2[1];
        h[1] = ray[2] * edge2[0] - ray[0] * edge2[2];
        h[2] = ray[0] * edge2[1] - ray[1] * edge2[0];

        // Calculate a = edge1 · h
        a = edge1[0] * h[0] + edge1[1] * h[1] + edge1[2] * h[2];

        // If ray is parallel to triangle (a is close to 0)
        if (a > -eps && a < eps)
            continue;

        f = 1.0 / a;

        // Calculate s = p - v1
        s[0] = p[0] - tri.v1[0];
        s[1] = p[1] - tri.v1[1];
        s[2] = p[2] - tri.v1[2];

        // Calculate u = f * (s · h)
        u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);

        // If u is outside the triangle
        if (u < 0.0 || u > 1.0)
            continue;

        // Calculate q = s × edge1
        q[0] = s[1] * edge1[2] - s[2] * edge1[1];
        q[1] = s[2] * edge1[0] - s[0] * edge1[2];
        q[2] = s[0] * edge1[1] - s[1] * edge1[0];

        // Calculate v = f * (ray · q)
        v = f * (ray[0] * q[0] + ray[1] * q[1] + ray[2] * q[2]);

        // If v is outside the triangle or u+v > 1
        if (v < 0.0 || u + v > 1.0)
            continue;

        // Calculate t = f * (edge2 · q)
        t = f * (edge2[0] * q[0] + edge2[1] * q[1] + edge2[2] * q[2]);

        // If intersection is in positive ray direction and not too close to the origin
        if (t > eps)
            intersections++;
    }

    // Odd number of intersections means the point is inside
    return (intersections % 2) == 1;
}

// Enum for shape types
enum class ShapeType
{
    SPHERE,
    CUBE,
    CYLINDER,
    CUSTOM
};

// Function to get the shape function based on the shape type
std::function<bool(double, double, double)> getShapeFunction(ShapeType type, double cx, double cy, double cz, double r)
{
    switch (type)
    {
    case ShapeType::SPHERE:
        return [cx, cy, cz, r](double x, double y, double z)
        {
            return (x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz) <= r * r;
        };
    case ShapeType::CUBE:
        return [cx, cy, cz, r](double x, double y, double z)
        {
            return std::abs(x - cx) <= r && std::abs(y - cy) <= r && std::abs(z - cz) <= r;
        };
    case ShapeType::CYLINDER:
        return [cx, cy, cz, r](double x, double y, double z)
        {
            return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r && std::abs(z - cz) <= r;
        };
    default:
        return [](double x, double y, double z)
        { return false; };
    }
}

// Improved VTK Writer Function
void saveToVTK(int step,
               const std::vector<std::vector<std::vector<double>>> &u,
               const std::vector<std::vector<std::vector<double>>> &v,
               const std::vector<std::vector<std::vector<double>>> &w,
               const std::vector<std::vector<std::vector<double>>> &p,
               const std::vector<std::vector<std::vector<int>>> &mask,
               const std::vector<double> &x,
               const std::vector<double> &y,
               const std::vector<double> &z)
{
    const int nx = x.size();
    const int ny = y.size();
    const int nz = z.size();

    // Create a VTK file for the current step
    std::string filename = "output/step_" + std::to_string(step) + ".vti";
    std::ofstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // XML VTI format (ImageData)
    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";

    // Define the grid dimensions and spacing
    double dx = (nx > 1) ? (x[1] - x[0]) : 1.0;
    double dy = (ny > 1) ? (y[1] - y[0]) : 1.0;
    double dz = (nz > 1) ? (z[1] - z[0]) : 1.0;

    file << "  <ImageData WholeExtent=\"0 " << (nx - 1) << " 0 " << (ny - 1) << " 0 " << (nz - 1)
         << "\" Origin=\"" << x[0] << " " << y[0] << " " << z[0]
         << "\" Spacing=\"" << dx << " " << dy << " " << dz << "\">\n";

    file << "    <Piece Extent=\"0 " << (nx - 1) << " 0 " << (ny - 1) << " 0 " << (nz - 1) << "\">\n";

    // Point data (at cell centers)
    file << "      <PointData>\n";

    // Pressure scalar field
    file << "        <DataArray type=\"Float32\" Name=\"pressure\" format=\"ascii\">\n";
    for (int k = 0; k < nz; ++k)
    {
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                file << p[k][j][i] << " ";
            }
            file << "\n";
        }
    }
    file << "        </DataArray>\n";

    // Velocity vector field
    file << "        <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int k = 0; k < nz; ++k)
    {
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                file << u[k][j][i] << " " << v[k][j][i] << " " << w[k][j][i] << " ";
            }
            file << "\n";
        }
    }
    file << "        </DataArray>\n";

    // Velocity magnitude for easy visualization
    file << "        <DataArray type=\"Float32\" Name=\"velocity_magnitude\" format=\"ascii\">\n";
    for (int k = 0; k < nz; ++k)
    {
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                double mag = std::sqrt(u[k][j][i] * u[k][j][i] + v[k][j][i] * v[k][j][i] + w[k][j][i] * w[k][j][i]);
                file << mag << " ";
            }
            file << "\n";
        }
    }
    file << "        </DataArray>\n";

    // Mask field
    file << "        <DataArray type=\"Int8\" Name=\"mask\" format=\"ascii\">\n";
    for (int k = 0; k < nz; ++k)
    {
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                file << mask[k][j][i] << " ";
            }
            file << "\n";
        }
    }
    file << "        </DataArray>\n";

    file << "      </PointData>\n";
    file << "    </Piece>\n";
    file << "  </ImageData>\n";
    file << "</VTKFile>\n";

    file.close();

    // Also generate a lightweight legacy VTK file for compatibility
    std::string legacy_filename = "output/step_" + std::to_string(step) + ".vtk";
    std::ofstream legacy_file(legacy_filename);

    if (legacy_file.is_open())
    {
        // VTK header
        legacy_file << "# vtk DataFile Version 3.0\n";
        legacy_file << "Navier-Stokes solution at step " << step << "\n";
        legacy_file << "ASCII\n";
        legacy_file << "DATASET STRUCTURED_POINTS\n";
        legacy_file << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
        legacy_file << "ORIGIN " << x[0] << " " << y[0] << " " << z[0] << "\n";
        legacy_file << "SPACING " << dx << " " << dy << " " << dz << "\n";

        // Point data
        legacy_file << "POINT_DATA " << nx * ny * nz << "\n";

        // Pressure field
        legacy_file << "SCALARS pressure float 1\n";
        legacy_file << "LOOKUP_TABLE default\n";
        for (int k = 0; k < nz; ++k)
        {
            for (int j = 0; j < ny; ++j)
            {
                for (int i = 0; i < nx; ++i)
                {
                    legacy_file << p[k][j][i] << "\n";
                }
            }
        }

        // Velocity field
        legacy_file << "VECTORS velocity float\n";
        for (int k = 0; k < nz; ++k)
        {
            for (int j = 0; j < ny; ++j)
            {
                for (int i = 0; i < nx; ++i)
                {
                    legacy_file << u[k][j][i] << " " << v[k][j][i] << " " << w[k][j][i] << "\n";
                }
            }
        }

        // Mask field
        legacy_file << "SCALARS mask int 1\n";
        legacy_file << "LOOKUP_TABLE default\n";
        for (int k = 0; k < nz; ++k)
        {
            for (int j = 0; j < ny; ++j)
            {
                for (int i = 0; i < nx; ++i)
                {
                    legacy_file << mask[k][j][i] << "\n";
                }
            }
        }

        legacy_file.close();
    }
}

class NavierStokesSolver3D
{
private:
    // Domain parameters
    double Lx, Ly, Lz; // Domain size
    int nx, ny, nz;    // Number of grid points
    double dx, dy, dz; // Grid spacing
    double Re;         // Reynolds number
    double dt;         // Time step
    int max_iter;      // Maximum iterations for pressure solver
    double tol;        // Convergence tolerance
    double r;          // Radius or half-length of the shape

    // Fields
    std::vector<std::vector<std::vector<double>>> u;      // x-velocity
    std::vector<std::vector<std::vector<double>>> v;      // y-velocity
    std::vector<std::vector<std::vector<double>>> w;      // z-velocity
    std::vector<std::vector<std::vector<double>>> p;      // pressure
    std::vector<std::vector<std::vector<double>>> u_prev; // previous u
    std::vector<std::vector<std::vector<double>>> v_prev; // previous v
    std::vector<std::vector<std::vector<double>>> w_prev; // previous w
    std::vector<std::vector<std::vector<int>>> mask;      // Fluid/solid mask
    std::vector<double> x, y, z;                          // Grid coordinates

    // Function to determine if a point is inside the solid object
    std::function<bool(double, double, double)> isSolid;

public:
    NavierStokesSolver3D(
        std::tuple<double, double, double> domain_size = {10.0, 10.0, 10.0},
        std::function<bool(double, double, double)> isSolid = [](double x, double y, double z)
        { return false; }, // Default: no solid
        int nx = 101, int ny = 101, int nz = 101,
        double reynolds = 100.0,
        double dt = 0.01,
        int max_iter = 1000,
        double tol = 1e-5,
        double r = 0.5 // Radius or half-length of the shape
        ) : Lx(std::get<0>(domain_size)), Ly(std::get<1>(domain_size)), Lz(std::get<2>(domain_size)),
            nx(nx), ny(ny), nz(nz), Re(reynolds),
            dt(dt), max_iter(max_iter), tol(tol), r(r), isSolid(isSolid)
    {
        // Calculate grid spacing
        dx = Lx / (nx - 1);
        dy = Ly / (ny - 1);
        dz = Lz / (nz - 1);

        // Initialize grid coordinates
        x.resize(nx);
        y.resize(ny);
        z.resize(nz);
        for (int i = 0; i < nx; ++i)
            x[i] = i * dx;
        for (int j = 0; j < ny; ++j)
            y[j] = j * dy;
        for (int k = 0; k < nz; ++k)
            z[k] = k * dz;

        // Initialize fields with proper dimensions
        u.resize(nz, std::vector<std::vector<double>>(ny, std::vector<double>(nx, 0.0)));
        v.resize(nz, std::vector<std::vector<double>>(ny, std::vector<double>(nx, 0.0)));
        w.resize(nz, std::vector<std::vector<double>>(ny, std::vector<double>(nx, 0.0)));
        p.resize(nz, std::vector<std::vector<double>>(ny, std::vector<double>(nx, 0.0)));
        u_prev.resize(nz, std::vector<std::vector<double>>(ny, std::vector<double>(nx, 0.0)));
        v_prev.resize(nz, std::vector<std::vector<double>>(ny, std::vector<double>(nx, 0.0)));
        w_prev.resize(nz, std::vector<std::vector<double>>(ny, std::vector<double>(nx, 0.0)));
        mask.resize(nz, std::vector<std::vector<int>>(ny, std::vector<int>(nx, 1)));

        // Set up the mask based on the user-defined shape
        setupMask();

        // Set initial boundary conditions
        for (int k = 0; k < nz; ++k)
        {
            for (int j = 0; j < ny; ++j)
            {
                u[k][j][0] = 1.0; // Inlet velocity (uniform flow)
            }
        }
    }

    void setupMask()
    {
        int solid_cells = 0;
        for (int k = 0; k < nz; ++k)
        {
            for (int j = 0; j < ny; ++j)
            {
                for (int i = 0; i < nx; ++i)
                {
                    double p[3] = {x[i], y[j], z[k]};
                    if (isSolid(p[0], p[1], p[2]))
                    {
                        mask[k][j][i] = 0;
                        solid_cells++;
                    }
                }
            }
        }
        // Debug: Print the number of solid cells
        std::cout << "Solid Cells: " << solid_cells << std::endl;
    }

    std::vector<std::vector<std::vector<double>>> pressurePoisson()
    {
        // Create the right-hand side of the pressure equation
        std::vector<std::vector<std::vector<double>>> b(nz, std::vector<std::vector<double>>(ny, std::vector<double>(nx, 0.0)));

// Compute source term for pressure Poisson equation
#pragma omp parallel for collapse(3)
        for (int k = 1; k < nz - 1; ++k)
        {
            for (int j = 1; j < ny - 1; ++j)
            {
                for (int i = 1; i < nx - 1; ++i)
                {
                    if (mask[k][j][i] == 1)
                    {
                        b[k][j][i] = ((u[k][j][i + 1] - u[k][j][i - 1]) / (2 * dx) +
                                      (v[k][j + 1][i] - v[k][j - 1][i]) / (2 * dy) +
                                      (w[k + 1][j][i] - w[k - 1][j][i]) / (2 * dz)) /
                                     dt;
                    }
                }
            }
        }

        // Iterative solution (Jacobi method)
        std::vector<std::vector<std::vector<double>>> p_next = p;
        double error;

        for (int iter = 0; iter < max_iter; ++iter)
        {
            std::vector<std::vector<std::vector<double>>> p_old = p_next;
            error = 0.0;

#pragma omp parallel for collapse(3) reduction(max : error)
            for (int k = 1; k < nz - 1; ++k)
            {
                for (int j = 1; j < ny - 1; ++j)
                {
                    for (int i = 1; i < nx - 1; ++i)
                    {
                        if (mask[k][j][i] == 1)
                        {
                            // Count number of fluid neighbors
                            double n_fluid = mask[k + 1][j][i] + mask[k - 1][j][i] +
                                             mask[k][j + 1][i] + mask[k][j - 1][i] +
                                             mask[k][j][i + 1] + mask[k][j][i - 1];

                            if (n_fluid > 0)
                            {
                                p_next[k][j][i] = ((p_old[k + 1][j][i] * mask[k + 1][j][i] +
                                                    p_old[k - 1][j][i] * mask[k - 1][j][i]) /
                                                       (dz * dz) +
                                                   (p_old[k][j + 1][i] * mask[k][j + 1][i] +
                                                    p_old[k][j - 1][i] * mask[k][j - 1][i]) /
                                                       (dy * dy) +
                                                   (p_old[k][j][i + 1] * mask[k][j][i + 1] +
                                                    p_old[k][j][i - 1] * mask[k][j][i - 1]) /
                                                       (dx * dx) -
                                                   b[k][j][i]) /
                                                  (n_fluid * (1 / (dx * dx) + 1 / (dy * dy) + 1 / (dz * dz)));

                                // Compute local error for convergence check
                                double local_error = std::abs(p_next[k][j][i] - p_old[k][j][i]);
                                error = std::max(error, local_error);
                            }
                        }
                    }
                }
            }

// Apply boundary conditions for pressure
#pragma omp parallel for collapse(2)
            for (int k = 0; k < nz; ++k)
            {
                for (int j = 0; j < ny; ++j)
                {
                    p_next[k][j][0] = p_next[k][j][1];           // Left boundary
                    p_next[k][j][nx - 1] = p_next[k][j][nx - 2]; // Right boundary (fixed pressure)
                }
            }

#pragma omp parallel for collapse(2)
            for (int k = 0; k < nz; ++k)
            {
                for (int i = 0; i < nx; ++i)
                {
                    p_next[k][0][i] = p_next[k][1][i];           // Bottom boundary
                    p_next[k][ny - 1][i] = p_next[k][ny - 2][i]; // Top boundary
                }
            }

#pragma omp parallel for collapse(2)
            for (int j = 0; j < ny; ++j)
            {
                for (int i = 0; i < nx; ++i)
                {
                    p_next[0][j][i] = p_next[1][j][i];           // Front boundary
                    p_next[nz - 1][j][i] = p_next[nz - 2][j][i]; // Back boundary
                }
            }

            // Check convergence
            if (error < tol)
            {
                std::cout << "Pressure solver converged in " << iter + 1 << " iterations" << std::endl;
                break;
            }

            if (iter == max_iter - 1)
            {
                std::cout << "Warning: Pressure solver did not converge. Final error: " << error << std::endl;
            }
        }

        return p_next;
    }

    void solveStep()
    {
        // Store previous velocities
        u_prev = u;
        v_prev = v;
        w_prev = w;

        // Intermediate velocity fields
        std::vector<std::vector<std::vector<double>>> u_temp = u;
        std::vector<std::vector<std::vector<double>>> v_temp = v;
        std::vector<std::vector<std::vector<double>>> w_temp = w;

// Calculate intermediate velocity (without pressure gradient)
#pragma omp parallel for collapse(3)
        for (int k = 1; k < nz - 1; ++k)
        {
            for (int j = 1; j < ny - 1; ++j)
            {
                for (int i = 1; i < nx - 1; ++i)
                {
                    if (mask[k][j][i] == 1)
                    {
                        // x-momentum equation
                        u_temp[k][j][i] = u[k][j][i] + dt * (
                                                                // Viscous terms
                                                                1 / Re * ((u[k + 1][j][i] - 2 * u[k][j][i] + u[k - 1][j][i]) / (dz * dz) + (u[k][j + 1][i] - 2 * u[k][j][i] + u[k][j - 1][i]) / (dy * dy) + (u[k][j][i + 1] - 2 * u[k][j][i] + u[k][j][i - 1]) / (dx * dx)) -
                                                                // Convective terms
                                                                u[k][j][i] * (u[k][j][i + 1] - u[k][j][i - 1]) / (2 * dx) -
                                                                v[k][j][i] * (u[k][j + 1][i] - u[k][j - 1][i]) / (2 * dy) -
                                                                w[k][j][i] * (u[k + 1][j][i] - u[k - 1][j][i]) / (2 * dz));

                        // y-momentum equation
                        v_temp[k][j][i] = v[k][j][i] + dt * (
                                                                // Viscous terms
                                                                1 / Re * ((v[k + 1][j][i] - 2 * v[k][j][i] + v[k - 1][j][i]) / (dz * dz) + (v[k][j + 1][i] - 2 * v[k][j][i] + v[k][j - 1][i]) / (dy * dy) + (v[k][j][i + 1] - 2 * v[k][j][i] + v[k][j][i - 1]) / (dx * dx)) -
                                                                // Convective terms
                                                                u[k][j][i] * (v[k][j][i + 1] - v[k][j][i - 1]) / (2 * dx) -
                                                                v[k][j][i] * (v[k][j + 1][i] - v[k][j - 1][i]) / (2 * dy) -
                                                                w[k][j][i] * (v[k + 1][j][i] - v[k - 1][j][i]) / (2 * dz));

                        // z-momentum equation
                        w_temp[k][j][i] = w[k][j][i] + dt * (
                                                                // Viscous terms
                                                                1 / Re * ((w[k + 1][j][i] - 2 * w[k][j][i] + w[k - 1][j][i]) / (dz * dz) + (w[k][j + 1][i] - 2 * w[k][j][i] + w[k][j - 1][i]) / (dy * dy) + (w[k][j][i + 1] - 2 * w[k][j][i] + w[k][j][i - 1]) / (dx * dx)) -
                                                                // Convective terms
                                                                u[k][j][i] * (w[k][j][i + 1] - w[k][j][i - 1]) / (2 * dx) -
                                                                v[k][j][i] * (w[k][j + 1][i] - w[k][j - 1][i]) / (2 * dy) -
                                                                w[k][j][i] * (w[k + 1][j][i] - w[k - 1][j][i]) / (2 * dz));
                    }
                }
            }
        }

        // Solve pressure Poisson equation
        p = pressurePoisson();

// Correct velocities with pressure gradient
#pragma omp parallel for collapse(3)
        for (int k = 1; k < nz - 1; ++k)
        {
            for (int j = 1; j < ny - 1; ++j)
            {
                for (int i = 1; i < nx - 1; ++i)
                {
                    if (mask[k][j][i] == 1)
                    {
                        u[k][j][i] = u_temp[k][j][i] - dt * (p[k][j][i + 1] - p[k][j][i - 1]) / (2 * dx);
                        v[k][j][i] = v_temp[k][j][i] - dt * (p[k][j + 1][i] - p[k][j - 1][i]) / (2 * dy);
                        w[k][j][i] = w_temp[k][j][i] - dt * (p[k + 1][j][i] - p[k - 1][j][i]) / (2 * dz);
                    }
                }
            }
        }

// Apply boundary conditions

// Left boundary (inlet): fixed uniform velocity
#pragma omp parallel for collapse(2)
        for (int k = 0; k < nz; ++k)
        {
            for (int j = 0; j < ny; ++j)
            {
                u[k][j][0] = 1.0;
                v[k][j][0] = 0.0;
                w[k][j][0] = 0.0;
            }
        }

// Right boundary (outlet): fixed pressure, zero gradient for velocity
#pragma omp parallel for collapse(2)
        for (int k = 0; k < nz; ++k)
        {
            for (int j = 0; j < ny; ++j)
            {
                p[k][j][nx - 1] = 0.0;             // Fixed pressure (e.g., atmospheric pressure)
                u[k][j][nx - 1] = u[k][j][nx - 2]; // Zero gradient for velocity
                v[k][j][nx - 1] = v[k][j][nx - 2]; // Zero gradient for velocity
                w[k][j][nx - 1] = w[k][j][nx - 2]; // Zero gradient for velocity
            }
        }

// Top and bottom boundaries: no-slip
#pragma omp parallel for collapse(2)
        for (int k = 0; k < nz; ++k)
        {
            for (int i = 0; i < nx; ++i)
            {
                u[k][0][i] = 0.0;
                v[k][0][i] = 0.0;
                w[k][0][i] = 0.0;
                u[k][ny - 1][i] = 0.0;
                v[k][ny - 1][i] = 0.0;
                w[k][ny - 1][i] = 0.0;
            }
        }

// Front and back boundaries: no-slip
#pragma omp parallel for collapse(2)
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                u[0][j][i] = 0.0;
                v[0][j][i] = 0.0;
                w[0][j][i] = 0.0;
                u[nz - 1][j][i] = 0.0;
                v[nz - 1][j][i] = 0.0;
                w[nz - 1][j][i] = 0.0;
            }
        }

// Sphere boundary: enforce zero velocity inside and on the sphere
#pragma omp parallel for collapse(3)
        for (int k = 0; k < nz; ++k)
        {
            for (int j = 0; j < ny; ++j)
            {
                for (int i = 0; i < nx; ++i)
                {
                    if (mask[k][j][i] == 0)
                    {
                        u[k][j][i] = 0.0;
                        v[k][j][i] = 0.0;
                        w[k][j][i] = 0.0;
                    }
                }
            }
        }
    }

    void saveToVTK(int step)
    {
        // Call the standalone VTK writer function
        ::saveToVTK(step, u, v, w, p, mask, x, y, z);
    }

    void saveSummary(int step, double elapsed_time)
    {
        std::ofstream file("output/summary_" + std::to_string(step) + ".txt");
        if (file.is_open())
        {
            // Calculate some flow statistics
            double max_u = 0.0, max_v = 0.0, max_w = 0.0, avg_p = 0.0;
            int fluid_cells = 0;

            for (int k = 0; k < nz; ++k)
            {
                for (int j = 0; j < ny; ++j)
                {
                    for (int i = 0; i < nx; ++i)
                    {
                        if (mask[k][j][i] == 1)
                        {
                            max_u = std::max(max_u, std::abs(u[k][j][i]));
                            max_v = std::max(max_v, std::abs(v[k][j][i]));
                            max_w = std::max(max_w, std::abs(w[k][j][i]));
                            avg_p += p[k][j][i];
                            fluid_cells++;
                        }
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
            file << "Domain size: " << Lx << " x " << Ly << " x " << Lz << std::endl;
            file << "Grid resolution: " << nx << " x " << ny << " x " << nz << std::endl;
            file << "Reynolds number: " << Re << std::endl;
            file << "Time step: " << dt << std::endl;
            file << "Current time: " << step * dt << std::endl;
            file << "Elapsed computation time: " << elapsed_time << " seconds" << std::endl;
            file << "Maximum u velocity: " << max_u << std::endl;
            file << "Maximum v velocity: " << max_v << std::endl;
            file << "Maximum w velocity: " << max_w << std::endl;
            file << "Average pressure: " << avg_p << std::endl;
            file << "Fluid cells: " << fluid_cells << " (" << (double)fluid_cells / (nx * ny * nz) * 100.0 << "% of domain)" << std::endl;

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
        (void)system("mkdir output 2> nul");
#else
        (void)system("mkdir -p output");
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

        // Sum pressure forces on the sphere (only consider cells adjacent to the sphere)
        for (int k = 1; k < nz - 1; ++k)
        {
            for (int j = 1; j < ny - 1; ++j)
            {
                for (int i = 1; i < nx - 1; ++i)
                {
                    // If this is a fluid cell and at least one neighbor is a solid cell
                    if (mask[k][j][i] == 1 && (mask[k + 1][j][i] == 0 || mask[k - 1][j][i] == 0 ||
                                               mask[k][j + 1][i] == 0 || mask[k][j - 1][i] == 0 ||
                                               mask[k][j][i + 1] == 0 || mask[k][j][i - 1] == 0))
                    {
                        // Calculate normal vector for this boundary cell
                        double nx_local = 0.0, ny_local = 0.0, nz_local = 0.0;
                        if (mask[k + 1][j][i] == 0)
                            nz_local += 1.0;
                        if (mask[k - 1][j][i] == 0)
                            nz_local -= 1.0;
                        if (mask[k][j + 1][i] == 0)
                            ny_local += 1.0;
                        if (mask[k][j - 1][i] == 0)
                            ny_local -= 1.0;
                        if (mask[k][j][i + 1] == 0)
                            nx_local += 1.0;
                        if (mask[k][j][i - 1] == 0)
                            nx_local -= 1.0;

                        // Normalize
                        double norm = std::sqrt(nx_local * nx_local + ny_local * ny_local + nz_local * nz_local);
                        if (norm > 0)
                        {
                            nx_local /= norm;
                            ny_local /= norm;
                            nz_local /= norm;
                        }

                        // Add pressure force in x-direction
                        drag_force += p[k][j][i] * nx_local * dy * dz;
                    }
                }
            }
        }

        // Compute drag coefficient
        // Cd = 2*F_drag / (density * U^2 * A)
        // We use density = 1 and U = 1 for simplicity
        double drag_coefficient = 2.0 * drag_force / (1.0 * 1.0 * M_PI * r * r);

        return drag_coefficient;
    }
};

int main(int argc, char *argv[])
{
    // Get number of threads from command line or use default
    int num_threads = 4; // Default value
    if (argc > 1)
    {
        num_threads = std::atoi(argv[1]);
    }

    // Set number of OpenMP threads
    omp_set_num_threads(num_threads);
    std::cout << "Running with " << num_threads << " threads" << std::endl;

    // Define simulation parameters
    std::tuple<double, double, double> domain_size = {10.0, 5.0, 5.0}; // Cuboid dimensions
    int nx = 101, ny = 51, nz = 51;                                    // Grid resolution
    double reynolds = 150.0;                                           // Reynolds number
    double dt = 0.005;                                                 // Time step
    int num_steps = 10000;                                             // Number of simulation steps
    int plot_interval = 200;                                           // Plotting interval
    int max_iter = 5000;                                               // Maximum iterations for pressure solver
    double tol = 1e-5;                                                 // Convergence tolerance
    double r = 0.8;                                                    // Radius or half-length of the shape

    // Let the user choose the shape
    std::cout << "Choose shape (0: Sphere, 1: Cube, 2: Cylinder, 3: Custom from file): ";
    int shape_choice;
    std::cin >> shape_choice;

    std::function<bool(double, double, double)> isSolid;

    if (shape_choice == 3)
    {
        std::string filename;
        std::cout << "Enter the path to the STL file: ";
        std::cin >> filename;

        auto triangles = readSTL(filename);

        // Calculate bounding box of the original STL model
        double min_x, max_x, min_y, max_y, min_z, max_z;
        calculateBoundingBox(triangles, min_x, max_x, min_y, max_y, min_z, max_z);

        // Define the desired position (2.5, 2.5, 2.5)
        double desired_x = 2.5, desired_y = 2.5, desired_z = 2.5;

        // Calculate the translation vector to move the STL model to the desired position
        double translate_x = desired_x - (max_x + min_x) / 2.0;
        double translate_y = desired_y - (max_y + min_y) / 2.0;
        double translate_z = desired_z - (max_z + min_z) / 2.0;

        // Define the scaling factor (1/4th of the domain length in x-direction)
        double scale_factor = 1;

        // Translate and scale the STL model
        translateAndScaleSTL(triangles, translate_x, translate_y, translate_z, scale_factor);

        // Update the bounding box after translation and scaling
        calculateBoundingBox(triangles, min_x, max_x, min_y, max_y, min_z, max_z);

        // Define the solid function using the translated and scaled STL model
        isSolid = [triangles](double x, double y, double z)
        {
            double p[3] = {x, y, z};
            return pointInSolid(p, triangles);
        };
    }
    else
    {
        // Use predefined shapes
        double cx = 2.5, cy = 2.5, cz = 2.5; // Center of the shape
        isSolid = getShapeFunction(static_cast<ShapeType>(shape_choice), cx, cy, cz, r);
    }

    // Create and run the solver
    NavierStokesSolver3D solver(
        domain_size,
        isSolid,
        nx, ny, nz,
        reynolds,
        dt,
        max_iter,
        tol,
        r // Pass radius to the solver
    );

    solver.simulate(num_steps, plot_interval);

    // Calculate and print drag coefficient
    double cd = solver.computeDragCoefficient();
    std::cout << "Drag coefficient: " << cd << std::endl;

    return 0;
}