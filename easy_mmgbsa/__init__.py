"""
easy_mmgbsa - Validated MM-GBSA implementations for protein-ligand binding free energies.

This package provides:
- mmgbsa_amb: Main validated MM-GBSA implementation (AMBER/GROMACS)
- mmgbsa_gmx: GROMACS-oriented wrapper
- mmgbsa: Lightweight standalone for GROMACS workflows
- mmgbsa_docking: Physics-based docking rescoring engine
"""

__version__ = "0.2.0"

# Make main functions available at package level
from .mmgbsa_amb import main as mmgbsa_amb_main
from .mmgbsa_gmx import main as mmgbsa_gmx_main
from .mmgbsa import main as mmgbsa_main
from .mmgbsa_docking import main as mmgbsa_docking_main

__all__ = [
    "mmgbsa_amb_main",
    "mmgbsa_gmx_main", 
    "mmgbsa_main",
    "mmgbsa_docking_main",
]