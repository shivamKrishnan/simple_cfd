```mermaid
flowchart TD
    %% Frontend (User Interface)
    A[User] -->|1. Sets Parameters| B["Web Interface<br>(HTML/CSS/JS)"]
    B -->|2. POST Request| C["Flask Server<br>(server.py)"]
    C -->|3. Routes Request| E{"3D or 2D?"}

    %% Computational Core
    subgraph Computational Core
        E -->|3D| F["3D Solver Module<br>(navier_stokes_3d.cpp)"]
        F --> H["3D Geometry Processor<br>(STL handling)"]
        E -->|2D| G["2D Solver Module<br>(navier_stokes_2d.cpp)"]
        G --> J["2D Shape Generator"]
        H & J -->|4. Raw Data| K[[ ]]
    end

    %% Post-Processing (VTK Generation)
    K -->|5. Simulation Data| M["VTK Output Generator<br>(vtk_writer.cpp)"]
    M -->|6. Generates| L["Simulation Results<br>(VTK files)"]
    L -->|7a. Download| B
    L -->|7b. Display Summary| B --> A

    %% Subgraph Definitions
    subgraph Frontend
        B
    end

    subgraph Backend
        C
        E
    end

    subgraph Post-Processing
        M
        L
    end

    %% Styling
    style A fill:#ffd700,stroke:#333,stroke-width:2px
    style B fill:#87cefa,stroke:#333
    style C fill:#ff7f50,stroke:#333
    style E fill:#90ee90,stroke:#333
    style ComputationalCore fill:#f0f8ff,stroke:#333,stroke-width:2px,stroke-dasharray:5 5
    style F fill:#add8e6,stroke:#333
    style G fill:#add8e6,stroke:#333
    style H fill:#dda0dd,stroke:#333
    style J fill:#98fb98,stroke:#333
    style K fill:#ffffff,stroke:#ffffff,stroke-width:0  %% Fully invisible
    style M fill:#ffa07a,stroke:#333
   ```
