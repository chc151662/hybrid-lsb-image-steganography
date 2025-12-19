# Hybrid LSB-Based Image Steganography

This repository provides a reference implementation of a hybrid image steganography framework proposed in the Project II report:

**"Enhancing Hybrid LSB-Based Image Steganography Through Adaptive Embedding Strategies"**

## Overview

The proposed framework adopts a dual-stage embedding architecture:

1. **Cover–Stego Matching (CSM)**  
   Zero-distortion matching is performed in non-edge regions by comparing LSBs of cover pixels with the secret bitstream. No pixel modification is introduced at this stage.

2. **Edge-Based XNOR-Fibonacci Embedding (EBXF)**  
   When the matching capacity is exhausted, an adaptive edge-based embedding strategy is activated to embed the remaining payload in high-gradient regions.

## Features

- Zero-distortion embedding in smooth regions under matching conditions
- Adaptive edge-guided data hiding strategy
- Lightweight auxiliary location information recording
- Support for experimental reproducibility

## Environment

- Python 3.10+
- NumPy
- Pillow

Dependencies are listed in `requirements.txt`.

## Usage
This repository is intended for research and experimental reproduction purposes.

- Core implementation modules are located in the `core/` directory.
- Experimental and evaluation scripts are provided in the `tests/` directories.

## Datasets

Experiments are conducted on standard benchmark images, including:
- USC-SIPI image dataset
- BOWS-2 dataset

## Notes

This implementation is provided as supplementary material for academic transparency.  
All methodological details and experimental analyses are fully described in the thesis.
