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
