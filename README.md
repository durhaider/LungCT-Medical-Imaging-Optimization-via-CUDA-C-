# LungCT Medical Imaging Pipeline - GPU Acceleration

## Project Overview

High-performance 3D lung mesh reconstruction pipeline for CT medical imaging with GPU-accelerated processing.

### Key Achievement: CPU-to-GPU Migration

Successfully migrated computationally intensive mesh optimization pipeline from sequential CPU processing to parallel CUDA implementation.

**Performance Impact:**
- **Pipeline Time:** 27-28 minutes (CPU) → 7.5 minutes (GPU)
- **Overall Speedup:** **3.7x end-to-end acceleration**
- **Optimization Stage:** ~20 minutes (CPU) → 2.5 minutes (GPU) 
- **Speedup on Optimization:** **~8x faster**

---

## Hardware Configuration

| Component | Specification |
|-----------|--------------|
| **GPU** | NVIDIA GeForce RTX 4050 Laptop (6GB VRAM) |
| **CPU** | Intel Core i5 13th Generation |
| **RAM** | 16GB DDR5 |
| **Graphics API** | VTK 9.5 |
| **CUDA Runtime** | 12.x |
| **Platform** | Windows 11, Visual Studio 2022 |

---

## Dataset Characteristics

- **Test File:** 3GB medical CT volume (lung vessel data)
- **Volume Dimensions:** 442×442×344 voxels
- **Voxel Spacing:** 0.9mm isotropic
- **Output Mesh:** ~14M vertices, ~28M triangles
- **Format:** DICOM → MHA → OBJ/STL

---

## Technical Stack

**Languages:** C++17, CUDA, Python  
**Build System:** CMake (conditional CUDA compilation)  
**Libraries:** VTK 9.5, CUDA Toolkit 12.x, SimpleITK, CuPy

---


**Note:** Core implementation kept private as this is ongoing research work.

---

## Build Instructions

**Prerequisites:**
- CUDA Toolkit 12.0+ (for GPU acceleration)
- VTK 9.0+ 
- CMake 3.20+
- Visual Studio 2019/2022 (Windows) or GCC 9+ (Linux)

**Compile with GPU acceleration:**
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```


## Migration Impact Summary

### Before Migration (Pure CPU)
- Mesh extraction: Fast (VTK Flying Edges)
- **Optimization: 18-22 minutes** BOTTLENECK
- File I/O: Slow (OBJ text format)
- **Total: 27-28 minutes**

### After Migration (GPU-Accelerated)
- Mesh extraction: Fast (unchanged)
- **Optimization: 2.5 minutes**  CUDA kernels
- File I/O: Improved (binary caching)
- **Total: 7.5 minutes**

**Time Saved Per Subject:** ~20 minutes  
**For 376 Subjects:** ~125 hours saved (5+ days of compute time)

---

## Key Technical Contributions

1. **Parallel Adjacency Builder** - 23 seconds on GPU vs 8-10 minutes on CPU
2. **CUDA Laplacian Smoothing** - Feature-aware parallel smoothing kernels
3. **GPU Surface Projection** - BVH-accelerated nearest-point queries
4. **Fast Mesh I/O** - VTK topology bypass for crash-free large mesh handling
5. **Binary Caching System** - VTP format for 10x faster reload

---

## Validation

✅ **Geometric Accuracy:** Hausdorff distance < 0.5mm vs CPU reference  
✅ **Tested on:** 376 subjects from Lung-PET-CT-Dx dataset  
✅ **Memory Safe:** CUDA memcheck verified, no leaks  
✅ **Cross-Platform:** Windows MSVC ✓, Linux GCC ✓

---

## Future Work

- Multi-GPU support for parallel subject processing
- Real-time mesh streaming for interactive editing
- Deep learning integration for automatic vessel segmentation

---

## Author

**Dur E Haider Hussain Fayyaz**  
Computer Engineer 

**Work:** GPU Acceleration, Medical Imaging, CUDA Optimization  
**Contact:** [durhaider2@outlook.com]

---

## License

Research work - All rights reserved.  
Contact author for collaboration inquiries.
