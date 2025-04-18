```mermaid
flowchart TD
    A[User] -->|Sets Parameters| B["Web Interface\n(HTML/CSS/JS)"]
    B -->|POST Request| C["Flask Server\n(server.py)"]
    C -->|Writes| D["input_params.txt"]
    C -->|Routes| E{"3D or 2D?"}
    
    E -->|3D| F["3D Solver\nnavier_stokes_3d.cpp"]
    F -->|STL Processing| H["Voxelize Mesh"]
    H --> I[3D Simulation]
    
    E -->|2D| G["2D Solver\nnavier_stokes_2d.cpp"]
    G --> J[2D Simulation]
    
    subgraph "3D Solver Modules"
        F1["STL Reader\n- readSTL()\n- preprocessSTL()"]
        F2["Voxelizer\n- pointInSolid()\n- voxelizeSTL()"]
        F3["Solver Core\n- pressurePoisson()\n- solveStep()"]
        F4["Physics\n- computeDragCoefficient()"]
        F5["Output\n- saveToVTK()\n- saveSummary()"]
        F1 --> F2 --> F3 --> F4 --> F5
    end
    
    subgraph "2D Solver Modules"
        G1["Shape Generator\n- setupMask()\n(16 shapes)"]
        G2["Solver Core\n- pressurePoisson()\n- solveStep()"]
        G3["LES Model\n- computeTurbulentViscosity()"]
        G4["Physics\n- computeDragCoefficient()"]
        G5["Output\n- saveToVTK()\n- saveSummary()"]
        G1 --> G2 --> G3 --> G4 --> G5
    end
    
    I & J --> K["VTK Output Files"]
    K --> L["ZIP Results"]
    L -->|Download| B
    K --> M["Summary Data"]
    M -->|Display| B --> A

    subgraph Frontend
        B
    end

    subgraph Backend
        C
        D
        E
        F
        G
        H
        I
        J
        K
        L
        M
    end

    style A fill:#ffd700,stroke:#333,stroke-width:2px
    style B fill:#87cefa,stroke:#333
    style C fill:#ff7f50,stroke:#333
    style D fill:#f0e68c,stroke:#333
    style E fill:#90ee90,stroke:#333
    style F fill:#add8e6,stroke:#333,stroke-dasharray:5
    style G fill:#add8e6,stroke:#333,stroke-dasharray:5
    style H fill:#dda0dd,stroke:#333
    style I fill:#98fb98,stroke:#333
    style J fill:#98fb98,stroke:#333
    style K fill:#ffa07a,stroke:#333
    style L fill:#ffb6c1,stroke:#333
    style M fill:#e6e6fa,stroke:#333
    style F1 fill:#d4f1f9,stroke:#333
    style F2 fill:#d4f1f9,stroke:#333
    style F3 fill:#d4f1f9,stroke:#333
    style F4 fill:#d4f1f9,stroke:#333
    style F5 fill:#d4f1f9,stroke:#333
    style G1 fill:#e8f8f5,stroke:#333
    style G2 fill:#e8f8f5,stroke:#333
    style G3 fill:#e8f8f5,stroke:#333
    style G4 fill:#e8f8f5,stroke:#333
    style G5 fill:#e8f8f5,stroke:#333
```
