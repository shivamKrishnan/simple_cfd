```mermaid
flowchart TD
    %% Main branches
    Start("Solver Selection") --> Mode{"3D or 2D Mode?"}
    Mode -->|3D| ThreeD[3D Solver]
    Mode -->|2D| TwoD[2D Solver]
    
    %% 3D Solver Branch
    subgraph "3D Solver Pipeline"
        ThreeD --> STL["STL Processing<br>- readSTL()"]
        STL --> Voxel["Voxelization"]
        Voxel --> InitCond3D["Initialize Conditions"]
        InitCond3D --> MainLoop3D["Main Time Stepping Loop"]
        
        MainLoop3D --> CoreSolver3D["Core CFD Solver<br>(Pressure-Velocity Coupling)"]
        CoreSolver3D --> Output3D["Output Generation"]
        
        %% Loop back
        Output3D -->|"t < t_end"| MainLoop3D
        Output3D -->|"t >= t_end"| Finish3D["Finalize Results"]
    end
    
    %% 2D Solver Branch
    subgraph "2D Solver Pipeline"
        TwoD --> Shape["Shape Generation<br>(16 predefined shapes)"]
        Shape --> InitCond2D["Initialize Conditions"]
        InitCond2D --> MainLoop2D["Main Time Stepping Loop"]
        
        MainLoop2D --> CoreSolver2D["Core CFD Solver<br>(Pressure-Velocity Coupling)"]
        CoreSolver2D --> Output2D["Output Generation"]
        
        %% Loop back
        Output2D -->|"t < t_end"| MainLoop2D
        Output2D -->|"t >= t_end"| Finish2D["Finalize Results"]
    end
    
    %% Final output
    Finish3D & Finish2D --> Results["Generate Result Files"]
    
    style Start fill:#f9f9f9,stroke:#333,stroke-width:2px
    style Mode fill:#90ee90,stroke:#333
    style ThreeD fill:#add8e6,stroke:#333
    style TwoD fill:#add8e6,stroke:#333
    style STL fill:#d4f1f9,stroke:#333
    style Voxel fill:#d4f1f9,stroke:#333
    style InitCond3D fill:#d4f1f9,stroke:#333
    style MainLoop3D fill:#ffd700,stroke:#333
    style Shape fill:#e8f8f5,stroke:#333
    style InitCond2D fill:#e8f8f5,stroke:#333
    style MainLoop2D fill:#ffd700,stroke:#333
    style Results fill:#ffa07a,stroke:#333
    style CoreSolver3D fill:#d4f1f9,stroke:#333
    style Output3D fill:#d4f1f9,stroke:#333
    style Finish3D fill:#d4f1f9,stroke:#333
    style CoreSolver2D fill:#e8f8f5,stroke:#333
    style Output2D fill:#e8f8f5,stroke:#333
    style Finish2D fill:#e8f8f5,stroke:#333
    ```
