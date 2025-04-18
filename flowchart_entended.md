```mermaid
flowchart TD
    %% Main branches
    Start("Solver Selection") --> Mode{"3D or 2D Mode?"}
    Mode -->|3D| ThreeD[3D Solver]
    Mode -->|2D| TwoD[2D Solver]
    
    %% 3D Solver Branch
    subgraph "3D Solver Pipeline"
        ThreeD --> STL["STL Processing\n- readSTL()\n- preprocessSTL()"]
        STL --> Voxel["Voxelization\n- pointInSolid()\n- voxelizeSTL()"]
        Voxel --> InitCond3D["Initialize Conditions\n- setupBoundaries()\n- initializeFields()"]
        InitCond3D --> MainLoop3D["Main Time Stepping Loop"]
        
        MainLoop3D --> Predictor["Predictor Step\n- computeIntermediate()"]
        Predictor --> Pressure3D["Pressure Correction\n- pressurePoisson()"]
        Pressure3D --> Corrector["Corrector Step\n- updateVelocities()"]
        Corrector --> Turbulence3D["LES Turbulence Model\n- computeTurbulentViscosity()"]
        Turbulence3D --> Forces3D["Force Computation\n- computePressureForces()\n- computeViscousForces()"]
        Forces3D --> Output3D["Output Generation\n- saveToVTK()\n- exportSummary()"]
        
        %% Loop back
        Output3D -->|"t < t_end"| MainLoop3D
        Output3D -->|"t >= t_end"| Finish3D["Finalize Results"]
    end
    
    %% 2D Solver Branch
    subgraph "2D Solver Pipeline"
        TwoD --> Shape["Shape Generation\n- setupMask()\n- 16 predefined shapes"]
        Shape --> InitCond2D["Initialize Conditions\n- setupBoundaries()\n- initializeFields()"]
        InitCond2D --> MainLoop2D["Main Time Stepping Loop"]
        
        MainLoop2D --> AdvDiff["Advection-Diffusion\n- computeAdvection()\n- computeDiffusion()"]
        AdvDiff --> Pressure2D["Pressure Correction\n- pressurePoisson()"]
        Pressure2D --> Velocity["Velocity Update\n- updateVelocities()"]
        Velocity --> Turbulence2D["Turbulence Modeling\n- computeTurbulentViscosity()"]
        Turbulence2D --> Forces2D["Force Computation\n- computeDragCoefficient()\n- computeLift()"]
        Forces2D --> Output2D["Output Generation\n- saveToVTK()\n- plotVelocityField()"]
        
        %% Loop back
        Output2D -->|"t < t_end"| MainLoop2D
        Output2D -->|"t >= t_end"| Finish2D["Finalize Results"]
    end
    
    %% Final output
    Finish3D & Finish2D --> Results["Generate Result Files\n- Summary Statistics\n- VTK Visualization Data\n- Performance Metrics"]
    
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
    style Predictor fill:#d4f1f9,stroke:#333
    style Pressure3D fill:#d4f1f9,stroke:#333
    style Corrector fill:#d4f1f9,stroke:#333
    style Turbulence3D fill:#d4f1f9,stroke:#333
    style Forces3D fill:#d4f1f9,stroke:#333
    style Output3D fill:#d4f1f9,stroke:#333
    style Finish3D fill:#d4f1f9,stroke:#333
    style AdvDiff fill:#e8f8f5,stroke:#333
    style Pressure2D fill:#e8f8f5,stroke:#333
    style Velocity fill:#e8f8f5,stroke:#333
    style Turbulence2D fill:#e8f8f5,stroke:#333
    style Forces2D fill:#e8f8f5,stroke:#333
    style Output2D fill:#e8f8f5,stroke:#333
    style Finish2D fill:#e8f8f5,stroke:#333
    ```
