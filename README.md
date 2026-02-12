# LungCT Medical Imaging Pipeline

**Final Year Project - NUST CEME, Computer Engineering**

## Project Scope
3D lung mesh reconstruction from CT DICOM data with GPU-accelerated post-processing.

## Key Achievement: CPU-to-GPU Migration

Migrated mesh optimization pipeline from sequential CPU processing to parallel CUDA implementation.

### Performance Impact
- **Optimization Stage:** 48-67s (CPU) → 2.8-4.7s (GPU)
- **Speedup:** ~17x on mesh post-processing
- **Hardware:** NVIDIA RTX 4050 Laptop GPU
- **Dataset:** 14M vertices, 28M triangles (lung vessel mesh)

### Technical Stack
- **Languages:** C++17, CUDA
- **Libraries:** VTK 9.4, CUDA Toolkit 12.x
- **Build System:** CMake (conditional CUDA compilation)
- **Platform:** Windows 11, Visual Studio 2022

### Architecture
```
DICOM → Volume Generation → Mesh Extraction → GPU Optimization → Visualization
                                                ├─ CUDA kernels
                                                ├─ Parallel adjacency
                                                └─ Feature-aware smoothing
```

### Repository Structure
```
├── kernel/          # CUDA kernels (private)
├── src/             # C++ implementation (private)
├── include/         # Headers (private)
└── docs/            # Performance documentation
```

**Note:** Core implementation files are kept private as this is ongoing research work.

---

**Author:** Team Ryft  
**Specialization:** GPU Acceleration, Medical Imaging, CUDA Optimization  
**Institution:** NUST College of Electrical & Mechanical Engineering
