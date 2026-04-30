# latpy

A high-performance Python array processing kernel designed for deterministic mathematics, bare-metal efficiency, and exact integer scaling in experimental Machine Learning environments.

## Overview

`latpy` strips away implicit floating-point coercion in favor of exact 128-bit scaling and rational arithmetic. It is built to ensure absolute reproducibility and determinism for specialized Neural Network training pipelines, particularly those utilizing ternary weights or pure integer forward passes.

## Features

- **Zero-Dependency Core**: Absolute self-containment to eliminate supply-chain bloat. Built directly against Python internals.
- **Deterministic**: Bit-exact operations across all OS architectures, crucial for reproducible ML research.
- **High-Performance Primitives**: Fixed-point math, exact rational scaling, and highly optimized CSR (Compressed Sparse Row) matrix operations tailored for sparse network topologies.
- **C-Speed Efficiency**: Bypasses standard interpreter overhead for core algebraic loops.

## Core Operations

- **Exact Integer + Rational Arithmetic**: Ensures no loss of precision from float approximations.
- **Sparse Propagation**: Specialized kernels (`propagate_sparse.py`, `propagate_b1.py`) for routing activations efficiently through massive graphs without standard matrix dense overhead.
- **Ternary Network Support**: Built-in methods for handling $-1, 0, 1$ quantization steps.

## License
Hybrid License.
