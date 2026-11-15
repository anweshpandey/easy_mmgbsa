#!/usr/bin/env python3
"""
MM-GBSA v4 -- a corrected, validated MM-GBSA implementation for AMBER/GAFF
topologies, with optional per-residue decomposition.

Why v4
------
The v1-v3 scripts used several approximations that are not consistent with
AMBER's MM-GBSA implementation:

  * Born radii were computed with a crude rj^3/r^4 integral instead of the
    Hawkins-Cramer-Truhlar (HCT) pairwise descreening integral used by OBC.
  * The OBC radius offset (0.09 A for igb=2/5) was missing.
  * The GB energy used a distance-dependent dielectric instead of the Still
    formula, and omitted the self-energy terms.
  * 1-4 LJ/Coulomb scaling was hard-coded and assumed a complete 1-4 list.
  * SASA used the GB radii and a non-AMBER surface tension.

v4 fixes those issues and validates them against AMBER's mmpbsa_py_energy /
MMPBSA.py.  Topologies and trajectories are read with ParmEd, so the code works
with both GROMACS (AMBER ff + ACPYPE/GAFF ligand) and native AMBER inputs.

Details that matter for quantitative agreement with AMBER:

  * The per-atom HCT descreening factor is taken from the prmtop SCREEN array
    when present (element-based fallback otherwise).  Ignoring SCREEN inflates
    the Born radii by roughly a factor of six.
  * The pairwise descreening sum is truncated at rgbmax = 25 A, matching
    AMBER's mmpbsa_py_energy.  Set --gb-radii-cutoff 0 to sum over all pairs.
  * The Coulomb prefactor is AMBER's 332.0522173 kcal A / mol / e^2.
  * SASA burial tests use solvent-expanded spheres (r_j + probe); using bare
    van der Waals radii overestimates the area by an order of magnitude.
  * prmtop EXCLUDED_ATOMS_LIST entries that are also proper 1-4 pairs are
    treated as 1-4 interactions (SCNB/SCEE), not as full exclusions.

What is implemented
-------------------
  * AMBER igb=2 (OBC-I) and igb=5 (OBC-II) GB, including self energies,
    HCT pairwise descreening and the 0.09 A offset.
  * LJ (AMBER Rmin combination rule) + Coulomb with exact exclusions and
    1-4 scaling read from the topology (prmtop EXCLUDED_ATOMS_LIST /
    dihedral SCEE/SCNB, or GROMACS [pairs]/[defaults]).
  * Shrake-Rupley SASA with Bondi radii and AMBER surface tension (0.0072).
  * Single-trajectory MM-GBSA (bonded terms cancel exactly and are not
    recomputed).
  * Per-receptor-residue decomposition that sums exactly to Delta G:
    vdw + elec + GB(cross + self + internal desolvation) + SASA.  Pair terms
    use AMBER idecomp-style 1/2 weighting, with a separate LIGAND row.
  * GRO, AMBER rst7/inpcrd, mdcrd and NetCDF trajectories.  XTC/TRR are read
    through MDAnalysis if it is installed (otherwise a clear error is raised).

Not implemented (deliberately, to keep the physics auditable)
--------------------------------------------------------------
  * Quasi-harmonic entropy (use --entropy from v3 if needed; it is not
    validated here).
  * Multi-trajectory MM-GBSA, ionic-strength GB screening, PB calculations.
  * LCPO surface areas.  Shrake-Rupley is used instead; expect small
    differences in the nonpolar term relative to MMPBSA.py.

Typical use
-----------
  # GROMACS topology + GRO snapshots (AMBER protein ff, ACPYPE ligand)
  python mmgbsa_amber_v4.py -t topol.top -c snap_*.gro --ligand LIG

  # AMBER prmtop + trajectory, decompose per receptor residue
  python mmgbsa_amber_v4.py -t complex.prmtop -c traj.nc \
      --ligand-resid-range 167-242 --decomp

  # One frame, verbose diagnostics
  python mmgbsa_amber_v4.py -t complex.prmtop -c frame.rst7 \
      --ligand LIG --verbose

References
----------
  Still et al., J. Am. Chem. Soc. 112, 6127 (1990).
  Hawkins, Cramer & Truhlar, J. Phys. Chem. 100, 19824 (1996).
  Onufriev, Bashford & Case, Proteins 55, 383 (2004).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import parmed as pmd
except ImportError:  # pragma: no cover
    sys.exit(
        "MM-GBSA v4 requires ParmEd (pip install parmed).  ParmEd reads both "
        "GROMACS .top and AMBER .prmtop topologies."
    )

try:
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import cdist

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    cKDTree = None
    cdist = None
    _HAVE_SCIPY = False


# ============================================================================
# 1. Constants and unit conventions
# ============================================================================

COULOMB = 332.0522173          # kcal mol^-1 A e^-2 (AMBER's constant)
DEFAULT_FUDGE_LJ = 0.5         # AMBER SCNB
DEFAULT_FUDGE_QQ = 1.0 / 1.2   # AMBER SCEE
DEFAULT_SURFACE_TENSION = 0.0072  # kcal mol^-1 A^-2 (AMBER)
DEFAULT_PROBE_RADIUS = 1.4     # A
DEFAULT_EPS_SOLVENT = 78.3
DEFAULT_EPS_SOLUTE = 1.0
DEFAULT_GB_RADII_CUTOFF = 25.0  # A; AMBER mmpbsa_py_energy uses rgbmax=25
DEFAULT_LIGAND_RESNAME = "LIG"  # used only when no other ligand selector is given

# AMBER OBC parameters.  igb=2 is OBC-I, igb=5 is OBC-II.
OBC_PARAMETERS = {
    2: dict(alpha=0.8, beta=0.0, gamma=2.909125, offset=0.09),
    5: dict(alpha=1.0, beta=0.8, gamma=4.85, offset=0.09),
}

# Bondi radii for SASA (A).  These are *not* the GB radii.
BONDI_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98,
    "B": 1.92, "Si": 2.10, "Se": 1.90,
}

# Common solvent/ion residue names.  MM-GBSA must be run on a desolvated system.
DEFAULT_STRIP_RESNAMES = (
    "WAT", "HOH", "SOL", "TIP3", "TIP4", "SPC", "DOD",
    "NA", "CL", "K", "MG", "ZN", "CA", "FE", "MN", "CU",
    "NA+", "CL-", "K+", "MG2", "ZN2", "CA2",
)

ELEMENT_BY_Z = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 25: "Mn", 26: "Fe",
    27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 34: "Se", 35: "Br", 53: "I",
}


# ============================================================================
# 2. Configuration
# ============================================================================

@dataclass
class Config:
    top_file: str
    coord_files: List[str] = field(default_factory=list)

    # Ligand selection
    ligand_resnames: List[str] = field(
        default_factory=lambda: [DEFAULT_LIGAND_RESNAME]
    )
    ligand_resid_range: Optional[Tuple[int, int]] = None
    ligand_mask: Optional[str] = None

    # Solvent / ion stripping
    strip_resnames: Set[str] = field(
        default_factory=lambda: set(DEFAULT_STRIP_RESNAMES)
    )
    no_strip: bool = False

    # GB model
    igb: int = 5
    eps_solute: float = DEFAULT_EPS_SOLUTE
    eps_solvent: float = DEFAULT_EPS_SOLVENT
    radii_set: str = "auto"       # auto, mbondi, mbondi2, mbondi3
    obc_alpha: Optional[float] = None
    obc_beta: Optional[float] = None
    obc_gamma: Optional[float] = None
    obc_offset: Optional[float] = None
    gb_radii_cutoff: Optional[float] = DEFAULT_GB_RADII_CUTOFF

    # SASA
    sasa_radii_set: str = "bondi"
    probe_radius: float = DEFAULT_PROBE_RADIUS
    surface_tension: float = DEFAULT_SURFACE_TENSION
    surface_offset: float = 0.0
    sasa_points: int = 240

    # Trajectory
    stride: int = 1
    start_frame: int = 1
    end_frame: Optional[int] = None
    unwrap: bool = False

    # Output
    output_csv: str = "mmgbsa_results.csv"
    decomp: bool = False
    decomp_csv: str = "mmgbsa_decomposition.csv"
    json_output: Optional[str] = None
    verbose: bool = False

    # gromacs library directory (optional; set GMXLIB before ParmEd reads .top)
    gmx_lib: Optional[str] = None


# ============================================================================
# 3. Small helpers
# ============================================================================

def setup_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("mmgbsa")


def guess_element_from_name(name: str, atom_type: str = "") -> str:
    """Best-effort element guess for topologies without atomic numbers."""
    for text in (name, atom_type):
        s = "".join(c for c in text if c.isalpha())
        if not s:
            continue
        s = s.upper()
        two = s[:2]
        if two in ("CL", "BR", "NA", "MG", "ZN", "FE", "CA", "MN", "CU",
                   "CO", "NI", "SI", "SE", "LI", "BE", "NE", "AL", "AR"):
            # "CA" is usually carbon-alpha, not calcium, in protein atom names.
            if two == "CA" and len(s) <= 2 and text is name:
                return "C"
            return two.capitalize()
        if s[0] in "HCNOPSFIBK":
            return s[0]
    return "C"


def element_symbol(atom: Any) -> str:
    z = getattr(atom, "atomic_number", 0) or 0
    if z in ELEMENT_BY_Z:
        return ELEMENT_BY_Z[z]
    try:
        return guess_element_from_name(atom.name, str(atom.type))
    except Exception:
        return "C"


def parse_resid_range(text: str) -> Tuple[int, int]:
    """Parse 'start-end' or 'start:end' (1-based, inclusive)."""
    m = re.match(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$", text)
    if not m:
        raise argparse.ArgumentTypeError(
            f"Invalid residue range '{text}'; use START-END (1-based)"
        )
    lo, hi = int(m.group(1)), int(m.group(2))
    if lo < 1 or hi < lo:
        raise argparse.ArgumentTypeError(
            f"Invalid residue range '{text}' (must satisfy 1 <= start <= end)"
        )
    return lo, hi


def box_lengths_from_box(box: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    if box is None:
        return None
    b = np.asarray(box, dtype=float)
    if b.size >= 3:
        return b[:3].copy()
    return None


def box_to_matrix(box: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    """Convert [a,b,c] or [a,b,c,alpha,beta,gamma] to a 3x3 matrix."""
    if box is None:
        return None
    b = np.asarray(box, dtype=float).ravel()
    if b.size == 3:
        return np.diag(b)
    if b.size < 6:
        return None
    a, bb, c, alpha, beta, gamma = b[:6]
    al, be, ga = np.radians([alpha, beta, gamma])
    sg = math.sin(ga)
    if abs(sg) < 1e-12:
        sg = 1e-12
    ax = a
    bx = bb * math.cos(ga)
    by = bb * sg
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / sg
    cz2 = c * c - cx * cx - cy * cy
    cz = math.sqrt(max(cz2, 0.0))
    return np.array([[ax, 0.0, 0.0], [bx, by, 0.0], [cx, cy, cz]])


def minimum_image_vector(dx: np.ndarray, box_mat: Optional[np.ndarray]) -> np.ndarray:
    """Return dx reduced into the primary cell (triclinic-aware)."""
    if box_mat is None:
        return dx
    try:
        inv = np.linalg.inv(box_mat)
    except np.linalg.LinAlgError:
        return dx
    frac = dx @ inv
    frac -= np.round(frac)
    return frac @ box_mat


def fibonacci_sphere(n_points: int) -> np.ndarray:
    """Approximately uniform unit-sphere points (golden-angle spiral)."""
    i = np.arange(n_points, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n_points)
    theta = math.pi * (1.0 + math.sqrt(5.0)) * i
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    pts = np.column_stack((x, y, z))
    # Guard against tiny numerical deviations from unit length.
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def chunked_iter(items: Sequence[int], size: int = 512) -> Iterator[np.ndarray]:
    for start in range(0, len(items), size):
        yield np.asarray(items[start:start + size], dtype=np.int64)


# ============================================================================
# 4. Geometry and topology model
# ============================================================================

def mbondi2_radii(structure: Any) -> np.ndarray:
    """AMBER mbondi2 radii (A), following ParmEd's changeradii implementation."""
    radii = np.empty(len(structure.atoms), dtype=float)
    for i, atom in enumerate(structure.atoms):
        z = getattr(atom, "atomic_number", 0) or 0
        if z == 1:
            partners = list(getattr(atom, "bond_partners", []))
            if partners and (getattr(partners[0], "atomic_number", 0) or 0) == 7:
                radii[i] = 1.3
            else:
                radii[i] = 1.2
        elif z == 6:
            typ = str(getattr(atom, "type", ""))
            mass = float(getattr(atom, "mass", 0.0) or 0.0)
            if isinstance(getattr(atom, "type", None), int):
                radii[i] = 1.7
            elif typ.startswith("C1") and mass > 13.0:
                radii[i] = 2.2
            elif typ.startswith("C2") and mass > 14.0:
                radii[i] = 2.2
            elif typ.startswith("C3") and mass > 15.0:
                radii[i] = 2.2
            else:
                radii[i] = 1.7
        elif z == 7:
            radii[i] = 1.55
        elif z == 8:
            radii[i] = 1.5
        elif z == 9:
            radii[i] = 1.5
        elif z == 14:
            radii[i] = 2.1
        elif z == 15:
            radii[i] = 1.85
        elif z == 16:
            radii[i] = 1.8
        elif z == 17:
            radii[i] = 1.7
        elif z == 35:
            radii[i] = 1.85
        elif z == 53:
            radii[i] = 1.98
        else:
            radii[i] = 1.5
    return radii


def mbondi_radii(structure: Any) -> np.ndarray:
    """AMBER mbondi radii (A)."""
    radii = np.empty(len(structure.atoms), dtype=float)
    for i, atom in enumerate(structure.atoms):
        z = getattr(atom, "atomic_number", 0) or 0
        if z == 1:
            partners = list(getattr(atom, "bond_partners", []))
            pz = (getattr(partners[0], "atomic_number", 0) or 0) if partners else 0
            if pz in (6, 7):
                radii[i] = 1.3
            elif pz in (8, 16):
                radii[i] = 0.8
            else:
                radii[i] = 1.2
        elif z == 6:
            typ = str(getattr(atom, "type", ""))
            mass = float(getattr(atom, "mass", 0.0) or 0.0)
            if isinstance(getattr(atom, "type", None), int):
                radii[i] = 1.7
            elif typ.startswith("C1") and mass > 13.0:
                radii[i] = 2.2
            elif typ.startswith("C2") and mass > 14.0:
                radii[i] = 2.2
            elif typ.startswith("C3") and mass > 15.0:
                radii[i] = 2.2
            else:
                radii[i] = 1.7
        elif z == 7:
            radii[i] = 1.55
        elif z == 8:
            radii[i] = 1.5
        elif z == 9:
            radii[i] = 1.5
        elif z == 14:
            radii[i] = 2.1
        elif z == 15:
            radii[i] = 1.85
        elif z == 16:
            radii[i] = 1.8
        elif z == 17:
            radii[i] = 1.7
        elif z == 35:
            radii[i] = 1.85
        elif z == 53:
            radii[i] = 1.98
        else:
            radii[i] = 1.5
    return radii


def mbondi3_radii(structure: Any) -> np.ndarray:
    """AMBER mbondi3 radii (A)."""
    radii = mbondi2_radii(structure)
    for i, atom in enumerate(structure.atoms):
        res = getattr(atom.residue, "name", "")
        name = getattr(atom, "name", "")
        if res in ("GLU", "ASP", "GL4", "AS4"):
            if name.startswith("OE") or name.startswith("OD"):
                radii[i] = 1.4
        elif res == "ARG":
            if name.startswith("HH") or name.startswith("HE"):
                radii[i] = 1.17
        if name == "OXT":
            radii[i] = 1.4
            if i > 0:
                radii[i - 1] = 1.4
    return radii


def bondi_radii(structure: Any) -> np.ndarray:
    radii = np.empty(len(structure.atoms), dtype=float)
    for i, atom in enumerate(structure.atoms):
        sym = element_symbol(atom)
        radii[i] = BONDI_RADII.get(sym, 1.5)
    return radii


def screen_factors(structure: Any) -> np.ndarray:
    """AMBER OBC screening factors for the HCT pairwise descreening integral.

    AMBER's ``tleap`` assigns these by element when it builds an mbondi2
    prmtop (ParmEd's ``_screen1``).  For native AMBER topologies the prmtop
    ``SCREEN`` array is authoritative and overrides this fallback.
    """
    screen = np.empty(len(structure.atoms), dtype=float)
    for i, atom in enumerate(structure.atoms):
        z = int(getattr(atom, "atomic_number", 0) or 0)
        if z == 1:
            screen[i] = 0.85
        elif z == 6:
            screen[i] = 0.72
        elif z == 7:
            screen[i] = 0.79
        elif z == 8:
            screen[i] = 0.85
        elif z == 9:
            screen[i] = 0.88
        elif z == 15:
            screen[i] = 0.86
        elif z == 16:
            screen[i] = 0.96
        else:
            screen[i] = 0.80
    return screen


class TopologyModel:
    """Lightweight, array-oriented view of a ParmEd Structure."""

    def __init__(self, structure: Any, config: Config, logger: logging.Logger):
        self.structure = structure
        self.config = config
        self.logger = logger
        self.n_atoms = len(structure.atoms)
        if self.n_atoms == 0:
            raise ValueError("Topology contains no atoms")

        atoms = structure.atoms
        self.names = [a.name for a in atoms]
        self.types = [str(a.type) for a in atoms]
        self.resnames = [a.residue.name for a in atoms]
        self.resids = [int(getattr(a.residue, "number", a.residue.idx + 1))
                       for a in atoms]
        self.residx = np.array([a.residue.idx for a in atoms], dtype=np.int64)
        self.n_residues = len(structure.residues)

        self.charges = np.array([float(a.charge) for a in atoms], dtype=float)
        self.masses = np.array([float(a.mass) for a in atoms], dtype=float)
        self.sigmas = np.array([float(a.sigma) for a in atoms], dtype=float)
        self.epsilons = np.array([float(a.epsilon) for a in atoms], dtype=float)
        self.atomic_numbers = np.array(
            [int(getattr(a, "atomic_number", 0) or 0) for a in atoms],
            dtype=np.int64,
        )
        self.elements = [element_symbol(a) for a in atoms]

        self.bonds: List[Tuple[int, int]] = [
            (int(b.atom1.idx), int(b.atom2.idx)) for b in structure.bonds
        ]
        self.angles: List[Tuple[int, int, int]] = [
            (int(a.atom1.idx), int(a.atom2.idx), int(a.atom3.idx))
            for a in structure.angles
        ]
        self.dihedrals: List[Tuple[int, int, int, int]] = [
            (int(d.atom1.idx), int(d.atom2.idx), int(d.atom3.idx), int(d.atom4.idx))
            for d in structure.dihedrals
        ]
        self.impropers: List[Tuple[int, int, int, int]] = [
            (int(d.atom1.idx), int(d.atom2.idx), int(d.atom3.idx), int(d.atom4.idx))
            for d in structure.impropers
        ]

        self.box: Optional[np.ndarray] = None
        if getattr(structure, "box", None) is not None:
            try:
                self.box = np.asarray(structure.box, dtype=float).copy()
            except Exception:
                self.box = None

        # ---- residues -------------------------------------------------
        self.residue_atoms: List[List[int]] = [[] for _ in range(self.n_residues)]
        for idx, ri in enumerate(self.residx):
            if 0 <= ri < self.n_residues:
                self.residue_atoms[ri].append(idx)
        self.residue_labels: List[str] = []
        for i, res in enumerate(structure.residues):
            number = int(getattr(res, "number", i + 1))
            self.residue_labels.append(f"{res.name}{number}")

        # ---- combinining rule and fudges ------------------------------
        self.combining_rule = getattr(structure, "combining_rule", "lorentz")
        defaults = getattr(structure, "defaults", None)
        self.fudge_lj = float(getattr(defaults, "fudgeLJ", DEFAULT_FUDGE_LJ) or DEFAULT_FUDGE_LJ)
        self.fudge_qq = float(getattr(defaults, "fudgeQQ", DEFAULT_FUDGE_QQ) or DEFAULT_FUDGE_QQ)
        # Guard against GROMACS files that store 1.0 for both.
        if not (0.0 < self.fudge_lj <= 1.0):
            self.fudge_lj = DEFAULT_FUDGE_LJ
        if not (0.0 < self.fudge_qq <= 1.0):
            self.fudge_qq = DEFAULT_FUDGE_QQ

        # ---- GB and SASA radii ---------------------------------------
        self.radii_gb = self._build_gb_radii()
        self.screen = self._build_screen()
        self.radii_sasa = self._build_sasa_radii()

        # ---- exclusions and 1-4 pairs --------------------------------
        self._excluded_pairs: Set[Tuple[int, int]] = set()
        self._one_four: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}
        self._build_exclusions_and_pairs()

        # ---- selection ------------------------------------------------
        self.ligand_idx = np.array([], dtype=np.int64)
        self.receptor_idx = np.array([], dtype=np.int64)
        self.complex_idx = np.array([], dtype=np.int64)
        self.stripped_idx = np.array([], dtype=np.int64)
        self._select_subsystems()

        self.lig_mask = np.zeros(self.n_atoms, dtype=bool)
        self.rec_mask = np.zeros(self.n_atoms, dtype=bool)
        self.cpx_mask = np.zeros(self.n_atoms, dtype=bool)
        self.lig_mask[self.ligand_idx] = True
        self.rec_mask[self.receptor_idx] = True
        self.cpx_mask[self.complex_idx] = True

        self._pair_data_built = False

    # ------------------------------------------------------------------
    # Radii
    # ------------------------------------------------------------------
    def _build_gb_radii(self) -> np.ndarray:
        cfg = self.config
        radii_choice = cfg.radii_set.lower()
        parm_data = getattr(self.structure, "parm_data", None)

        if radii_choice == "auto":
            if parm_data is not None and "RADII" in parm_data:
                radii = np.asarray(parm_data["RADII"], dtype=float)
                if radii.size == self.n_atoms:
                    radius_set = parm_data.get("RADIUS_SET")
                    if radius_set:
                        self.logger.info("Using GB radii from prmtop: %s", radius_set[0])
                    return radii
            radii_choice = "mbondi2"

        if radii_choice == "mbondi":
            return mbondi_radii(self.structure)
        if radii_choice == "mbondi2":
            return mbondi2_radii(self.structure)
        if radii_choice == "mbondi3":
            return mbondi3_radii(self.structure)
        if radii_choice == "bondi":
            return bondi_radii(self.structure)
        raise ValueError(f"Unknown radii set '{cfg.radii_set}'")

    def _build_screen(self) -> np.ndarray:
        parm_data = getattr(self.structure, "parm_data", None)
        if parm_data is not None and "SCREEN" in parm_data:
            screen = np.asarray(parm_data["SCREEN"], dtype=float)
            if screen.size == self.n_atoms:
                return screen
        return screen_factors(self.structure)

    def _build_sasa_radii(self) -> np.ndarray:
        choice = self.config.sasa_radii_set.lower()
        if choice in ("gb", "mbondi", "mbondi2", "mbondi3", "auto"):
            return np.asarray(self.radii_gb, dtype=float).copy()
        return bondi_radii(self.structure)

    # ------------------------------------------------------------------
    # Exclusions and 1-4 terms
    # ------------------------------------------------------------------
    def _add_excluded(self, i: int, j: int) -> None:
        if i == j:
            return
        if i > j:
            i, j = j, i
        self._excluded_pairs.add((i, j))

    def _add_one_four(self, i: int, j: int, sigma: float, epsilon: float,
                      scale_lj: float, scale_elec: float) -> None:
        if i == j:
            return
        if i > j:
            i, j = j, i
        self._one_four[(i, j)] = (
            float(sigma), float(epsilon), float(scale_lj), float(scale_elec)
        )

    def _build_exclusions_and_pairs(self) -> None:
        parm_data = getattr(self.structure, "parm_data", None)
        is_amber = parm_data is not None and "EXCLUDED_ATOMS_LIST" in parm_data

        # 1-2 and 1-3 exclusions from the bond graph.  This is also the
        # fallback for topologies without an explicit exclusion list.
        for atom in self.structure.atoms:
            for partner in getattr(atom, "bond_partners", []):
                self._add_excluded(atom.idx, partner.idx)
            for partner in getattr(atom, "angle_partners", []):
                self._add_excluded(atom.idx, partner.idx)

        if is_amber:
            # The prmtop list is authoritative and also contains ring-closure
            # 1-4 pairs that are fully excluded.
            counts = parm_data.get("NUMBER_EXCLUDED_ATOMS", [])
            flat = parm_data.get("EXCLUDED_ATOMS_LIST", [])
            offset = 0
            for i, count in enumerate(counts):
                count = int(count)
                for value in flat[offset:offset + count]:
                    j = int(value) - 1  # prmtop list is 1-based
                    if 0 <= j < self.n_atoms:
                        self._add_excluded(i, j)
                offset += count

        # GROMACS [pairs] are stored by ParmEd as NonbondedException objects.
        adjusts = list(getattr(self.structure, "adjusts", []) or [])
        for exc in adjusts:
            i = int(exc.atom1.idx)
            j = int(exc.atom2.idx)
            etype = getattr(exc, "type", None)
            if etype is None:
                self._add_excluded(i, j)
                continue
            sigma = float(getattr(etype, "sigma", 0.0) or 0.0)
            epsilon = float(getattr(etype, "epsilon", 0.0) or 0.0)
            chgscale = float(getattr(etype, "chgscale", 1.0) or 1.0)
            if epsilon == 0.0 and sigma == 0.0:
                self._add_excluded(i, j)
            else:
                # The GROMACS pair parameters already include fudgeLJ.
                self._add_one_four(i, j, sigma, epsilon, 1.0, chgscale)

        # AMBER 1-4 terms come from proper dihedrals.  The prmtop exclusion
        # list contains the same 1-4 pairs (they are excluded from the normal
        # nonbonded sum and added back here with SCEE/SCNB scaling), so a 1-4
        # pair must take precedence over an exclusion.
        if not self._one_four:
            for dih in self.structure.dihedrals:
                if getattr(dih, "improper", False) or getattr(dih, "ignore_end", False):
                    continue
                i = int(dih.atom1.idx)
                j = int(dih.atom4.idx)
                if i == j:
                    continue
                dtype = getattr(dih, "type", None)
                scee = getattr(dtype, "scee", None)
                scnb = getattr(dtype, "scnb", None)
                try:
                    scale_elec = 1.0 / float(scee) if scee is not None else self.fudge_qq
                except (TypeError, ZeroDivisionError):
                    scale_elec = self.fudge_qq
                try:
                    scale_lj = 1.0 / float(scnb) if scnb is not None else self.fudge_lj
                except (TypeError, ZeroDivisionError):
                    scale_lj = self.fudge_lj
                self._add_one_four(i, j, 0.0, 0.0, scale_lj, scale_elec)

    # ------------------------------------------------------------------
    # Ligand / receptor / stripped selection
    # ------------------------------------------------------------------
    def _select_subsystems(self) -> None:
        cfg = self.config
        ligand: Set[int] = set()

        if cfg.ligand_resnames:
            wanted = {name for name in cfg.ligand_resnames if name}
            for i, rn in enumerate(self.resnames):
                if rn in wanted:
                    ligand.add(i)

        if cfg.ligand_resid_range is not None:
            lo, hi = cfg.ligand_resid_range
            for ri in range(self.n_residues):
                # 1-based residue index, consistent with AMBER :1-242 masks.
                if lo <= (ri + 1) <= hi:
                    ligand.update(self.residue_atoms[ri])

        if cfg.ligand_mask:
            try:
                from parmed.amber.mask import AmberMask
                mask = AmberMask(self.structure, cfg.ligand_mask)
                ligand.update(int(x) for x in mask.Selection())
            except Exception as exc:
                raise ValueError(
                    f"Could not interpret --ligand-mask '{cfg.ligand_mask}': {exc}"
                ) from exc

        if not ligand:
            available = sorted(set(self.resnames))
            raise ValueError(
                "No ligand atoms selected.  Use --ligand RESNAME, "
                "--ligand-resid-range START-END, or --ligand-mask.  "
                f"Residue names present: {', '.join(available[:40])}"
                + (" ..." if len(available) > 40 else "")
            )

        strip: Set[int] = set()
        if not cfg.no_strip:
            strip = {
                i for i, rn in enumerate(self.resnames)
                if rn in cfg.strip_resnames and i not in ligand
            }

        self.ligand_idx = np.array(sorted(ligand), dtype=np.int64)
        self.stripped_idx = np.array(sorted(strip), dtype=np.int64)
        all_idx = set(range(self.n_atoms))
        self.receptor_idx = np.array(
            sorted(all_idx - ligand - strip), dtype=np.int64
        )
        self.complex_idx = np.array(
            sorted(all_idx - strip), dtype=np.int64
        )
        if self.receptor_idx.size == 0:
            raise ValueError("No receptor atoms remain after ligand selection/stripping")
        if self.ligand_idx.size == 0:
            raise ValueError("No ligand atoms remain after stripping")

    # ------------------------------------------------------------------
    # Pair arrays
    # ------------------------------------------------------------------
    def build_pair_data(self) -> None:
        """Precompute the fixed i<j pair list, exclusions and 1-4 scales."""
        if self._pair_data_built:
            return
        n = self.n_atoms
        iu = np.triu_indices(n, k=1)
        self.iu = (iu[0].astype(np.int64), iu[1].astype(np.int64))
        self.n_pairs = self.iu[0].size
        lin = self.iu[0] * n + self.iu[1]
        # triu_indices is row-major, hence lin is already sorted.
        self._lin_pairs = lin

        excl_lin = np.fromiter(
            (i * n + j for (i, j) in self._excluded_pairs),
            dtype=np.int64,
            count=len(self._excluded_pairs),
        )
        if excl_lin.size:
            excl_lin = np.sort(excl_lin)
            self.excl_mask = np.isin(lin, excl_lin)
        else:
            self.excl_mask = np.zeros(self.n_pairs, dtype=bool)

        self.scale_lj = np.ones(self.n_pairs, dtype=np.float64)
        self.scale_elec = np.ones(self.n_pairs, dtype=np.float64)
        self.sigma14 = np.zeros(self.n_pairs, dtype=np.float64)
        self.eps14 = np.zeros(self.n_pairs, dtype=np.float64)
        self.one_four_mask = np.zeros(self.n_pairs, dtype=bool)

        if self._one_four:
            keys = np.fromiter(
                (i * n + j for (i, j) in self._one_four.keys()),
                dtype=np.int64,
                count=len(self._one_four),
            )
            order = np.argsort(keys)
            keys = keys[order]
            pairs = [list(self._one_four.keys())[k] for k in order]
            pos = np.searchsorted(lin, keys)
            valid = (pos < self.n_pairs) & (lin[np.clip(pos, 0, self.n_pairs - 1)] == keys)
            if not np.all(valid):
                self.logger.warning(
                    "%d 1-4 pairs could not be located in the pair list",
                    int((~valid).sum()),
                )
            pos = pos[valid]
            pairs = [pairs[k] for k in range(len(pairs)) if valid[k]]
            for p, (i, j) in zip(pos, pairs):
                sigma, epsilon, slj, selec = self._one_four[(i, j)]
                self.sigma14[p] = sigma
                self.eps14[p] = epsilon
                self.scale_lj[p] = slj
                self.scale_elec[p] = selec
                self.one_four_mask[p] = True

        # Exclusions contribute to nothing in the gas phase.  Pairs that are
        # both excluded and 1-4 keep their 1-4 scaling (AMBER convention).
        excl_only = self.excl_mask & ~self.one_four_mask
        self.scale_lj[excl_only] = 0.0
        self.scale_elec[excl_only] = 0.0
        self.sigma14[excl_only] = 0.0
        self.eps14[excl_only] = 0.0

        self._pair_data_built = True

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        return {
            "n_atoms": self.n_atoms,
            "n_residues": self.n_residues,
            "n_ligand_atoms": int(self.ligand_idx.size),
            "n_receptor_atoms": int(self.receptor_idx.size),
            "n_stripped_atoms": int(self.stripped_idx.size),
            "n_excluded_pairs": len(self._excluded_pairs),
            "n_one_four_pairs": len(self._one_four),
            "combining_rule": self.combining_rule,
            "fudge_lj": self.fudge_lj,
            "fudge_qq": self.fudge_qq,
            "ligand_resnames": sorted({self.resnames[i] for i in self.ligand_idx}),
        }


# ============================================================================
# 5. GB model (HCT pairwise descreening + OBC)
# ============================================================================

class GBModel:
    """AMBER OBC generalized Born model (igb=2 or igb=5)."""

    def __init__(self, config: Config):
        params = OBC_PARAMETERS.get(config.igb)
        if params is None:
            raise ValueError(
                f"igb={config.igb} is not supported; use 2 (OBC-I) or 5 (OBC-II)"
            )
        self.alpha = config.obc_alpha if config.obc_alpha is not None else params["alpha"]
        self.beta = config.obc_beta if config.obc_beta is not None else params["beta"]
        self.gamma = config.obc_gamma if config.obc_gamma is not None else params["gamma"]
        self.offset = config.obc_offset if config.obc_offset is not None else params["offset"]
        self.eps_solute = float(config.eps_solute)
        self.eps_solvent = float(config.eps_solvent)
        self.kfac = COULOMB * (1.0 / self.eps_solute - 1.0 / self.eps_solvent)
        self.radii_cutoff = (float(config.gb_radii_cutoff)
                             if config.gb_radii_cutoff else None)

    def born_radii(self, coords: np.ndarray, radii: np.ndarray,
                   screen: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute OBC Born radii with the HCT pairwise integral.

        This is a vectorised transcription of the OpenMM/AMBER OBC code:

          rho_i = r_i - offset
          for each j != i with rho_i < r + rho_j:
              l = 1 / max(rho_i, |r - rho_j|)
              u = 1 / (r + rho_j)
              term = l - u + 0.25*r*(u^2 - l^2)
                     + 0.5*log(u/l)/r
                     + 0.25*rho_j^2*(l^2 - u^2)/r
              if rho_i < rho_j - r:
                  term += 2*(1/rho_i - l)
          psi = 0.5*rho_i*sum(term)
          tanh_sum = tanh(alpha*psi - beta*psi^2 + gamma*psi^3)
          R_i = 1 / (1/rho_i - tanh_sum/r_i)
        """
        coords = np.asarray(coords, dtype=float)
        radii = np.asarray(radii, dtype=float)
        n = radii.size
        born = np.empty(n, dtype=float)
        rho = radii - self.offset
        # Atoms with a non-positive offset radius are pathological; keep a
        # stable value instead of dividing by zero.
        bad = rho <= 0.0
        if np.any(bad):
            rho = np.where(bad, np.maximum(radii, 1.0), rho)
        if screen is None:
            screen = np.ones(n, dtype=float)
        else:
            screen = np.asarray(screen, dtype=float)
        scaled = rho * screen

        for i in range(n):
            rho_i = rho[i]
            if radii[i] <= 0.0:
                born[i] = 1.0
                continue
            d = coords - coords[i]
            r = np.sqrt(np.einsum("ij,ij->i", d, d))
            r[i] = np.inf  # exclude self

            # HCT only contributes when the other atom's scaled offset sphere
            # reaches beyond atom i's offset sphere.
            mask = (rho_i < r + scaled) & np.isfinite(r) & (r > 1e-8)
            if self.radii_cutoff is not None:
                mask &= r <= self.radii_cutoff
            if not np.any(mask):
                born[i] = rho_i
                continue

            rr = r[mask]
            rho_j = scaled[mask]
            l = np.maximum(rho_i, np.abs(rr - rho_j))
            l = 1.0 / l
            u = 1.0 / (rr + rho_j)
            l2 = l * l
            u2 = u * u
            inv_r = 1.0 / rr

            term = (
                l - u
                + 0.25 * rr * (u2 - l2)
                + 0.5 * inv_r * np.log(u / l)
                + 0.25 * (rho_j * rho_j) * inv_r * (l2 - u2)
            )
            # Atom i entirely inside atom j (not in the original HCT paper;
            # the correction follows Tinker/OpenMM).
            inside = rho_i < (rho_j - rr)
            if np.any(inside):
                term[inside] += 2.0 * (1.0 / rho_i - l[inside])

            psi = 0.5 * rho_i * float(np.sum(term))
            psi2 = psi * psi
            tanh_sum = math.tanh(
                self.alpha * psi - self.beta * psi2 + self.gamma * psi2 * psi
            )
            denom = 1.0 / rho_i - tanh_sum / radii[i]
            if abs(denom) < 1e-12:
                born[i] = rho_i
            else:
                born[i] = 1.0 / denom
            if not np.isfinite(born[i]) or born[i] <= 0.0:
                born[i] = rho_i

        return born

    def self_energy(self, charges: np.ndarray, born: np.ndarray) -> float:
        """GB self-energy: -0.5*kfac*q_i^2/R_i."""
        q = np.asarray(charges, dtype=float)
        b = np.asarray(born, dtype=float)
        return float(-0.5 * self.kfac * np.sum(q * q / b))

    def pair_energies(self, r: np.ndarray, qi: np.ndarray, qj: np.ndarray,
                      Ri: np.ndarray, Rj: np.ndarray) -> np.ndarray:
        """Per-pair GB energy for i<j: -kfac*qi*qj/f_GB."""
        r2 = r * r
        alpha2 = Ri * Rj
        exp_term = np.exp(-r2 / (4.0 * alpha2))
        f = np.sqrt(r2 + alpha2 * exp_term)
        return -self.kfac * qi * qj / f

    def energy(self, coords: np.ndarray, charges: np.ndarray,
               born: np.ndarray) -> float:
        """Total GB energy (self + all pairs), used for subsystems."""
        n = charges.size
        if n == 0:
            return 0.0
        e = self.self_energy(charges, born)
        if n < 2:
            return e
        if _HAVE_SCIPY:
            r = cdist(coords, coords)
        else:  # pragma: no cover
            diff = coords[:, None, :] - coords[None, :, :]
            r = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        iu = np.triu_indices(n, k=1)
        e += float(np.sum(self.pair_energies(
            r[iu], charges[iu[0]], charges[iu[1]], born[iu[0]], born[iu[1]]
        )))
        return e


# ============================================================================
# 6. Nonbonded (LJ + Coulomb) and SASA
# ============================================================================

class NonbondedModel:
    """AMBER-style LJ + Coulomb with exclusions and 1-4 scaling."""

    def __init__(self, top: TopologyModel):
        self.top = top
        self.combining_rule = top.combining_rule
        i, j = top.iu
        self.qi = top.charges[i]
        self.qj = top.charges[j]
        self.charge_product = self.qi * self.qj

        if self.combining_rule == "geometric":
            sigma = np.sqrt(top.sigmas[i] * top.sigmas[j])
        else:
            sigma = 0.5 * (top.sigmas[i] + top.sigmas[j])
        epsilon = np.sqrt(np.maximum(top.epsilons[i] * top.epsilons[j], 0.0))

        # Explicit GROMACS 1-4 pair parameters override the combining rule.
        explicit = top.eps14 > 0.0
        if np.any(explicit):
            sigma[explicit] = top.sigma14[explicit]
            epsilon[explicit] = top.eps14[explicit]

        self.sigma = sigma
        self.epsilon = epsilon

    def pair_energies(self, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r = np.asarray(r, dtype=float)
        sr6 = np.zeros_like(r)
        np.divide(self.sigma, r, out=sr6, where=r > 1e-12)
        sr6 = sr6 ** 6
        e_vdw = 4.0 * self.epsilon * (sr6 * sr6 - sr6)
        e_elec = np.zeros_like(r)
        np.divide(COULOMB * self.charge_product, r, out=e_elec, where=r > 1e-12)
        e_vdw *= self.top.scale_lj
        e_elec *= self.top.scale_elec
        return e_vdw, e_elec


class SASACalculator:
    """Shrake-Rupley solvent-accessible surface areas."""

    def __init__(self, config: Config):
        self.probe = float(config.probe_radius)
        self.n_points = int(config.sasa_points)
        if self.n_points < 12:
            raise ValueError("--sasa-points must be at least 12")
        self.points = fibonacci_sphere(self.n_points)
        self.surface_tension = float(config.surface_tension)
        self.surface_offset = float(config.surface_offset)

    def atom_areas(self, coords: np.ndarray, radii: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        radii = np.asarray(radii, dtype=float)
        n = radii.size
        areas = np.zeros(n, dtype=float)
        if n == 0:
            return areas
        max_radius = float(np.max(radii)) + self.probe
        if _HAVE_SCIPY and cKDTree is not None:
            tree = cKDTree(coords)
            for i in range(n):
                ri = radii[i]
                if ri <= 0.0:
                    continue
                R = ri + self.probe
                cand = tree.query_ball_point(coords[i], R + max_radius)
                cand = [j for j in cand if j != i and radii[j] > 0.0]
                if not cand:
                    areas[i] = 4.0 * math.pi * R * R
                    continue
                cand = np.asarray(cand, dtype=np.int64)
                pts = coords[i] + R * self.points
                d = cdist(pts, coords[cand])
                # A point is buried if it lies inside another atom's
                # *solvent-expanded* sphere (r_j + probe).  Using the bare
                # vdW radius here overestimates SASA by ~10x.
                buried = np.any(d < (radii[cand] + self.probe)[None, :], axis=1)
                areas[i] = 4.0 * math.pi * R * R * (1.0 - float(np.mean(buried)))
        else:  # pragma: no cover
            diff = coords[:, None, :] - coords[None, :, :]
            dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
            for i in range(n):
                ri = radii[i]
                if ri <= 0.0:
                    continue
                R = ri + self.probe
                cand = np.where(
                    (dist[i] < R + radii + self.probe) & (np.arange(n) != i)
                )[0]
                if cand.size == 0:
                    areas[i] = 4.0 * math.pi * R * R
                    continue
                pts = coords[i] + R * self.points
                d = np.linalg.norm(pts[:, None, :] - coords[cand][None, :, :], axis=2)
                buried = np.any(d < (radii[cand] + self.probe)[None, :], axis=1)
                areas[i] = 4.0 * math.pi * R * R * (1.0 - float(np.mean(buried)))
        return areas

    def energy(self, areas: np.ndarray) -> float:
        return self.surface_tension * float(np.sum(areas)) + \
            self.surface_offset * float(areas.size)


# ============================================================================
# 7. Trajectory readers
# ============================================================================

def expand_coord_files(patterns: Sequence[str]) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            files.extend(matches)
        elif os.path.exists(pattern):
            files.append(pattern)
    # Preserve order but remove duplicates.
    seen: Set[str] = set()
    unique: List[str] = []
    for f in files:
        key = os.path.abspath(f)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def _read_gro(path: str) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """Read a GRO file (coordinates, box, atom names)."""
    struc = pmd.load_file(path)
    coords = np.asarray(struc.coordinates, dtype=float)
    box = np.asarray(struc.box, dtype=float) if struc.box is not None else None
    names = [a.name for a in struc.atoms]
    return coords, box, names


def _read_rst7(path: str) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    from parmed.amber.asciicrd import AmberAsciiRestart
    rst = AmberAsciiRestart(path, mode="r")
    coords = np.asarray(rst.coordinates[0], dtype=float)
    box = np.asarray(rst.box, dtype=float) if rst.box is not None else None
    return coords, box, []


def _mdcrd_has_box(path: str, natom: int) -> bool:
    """Detect whether an ASCII mdcrd has a 3-value box line per frame."""
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        if len(lines) < 2:
            return False
        # Number of coordinate lines per frame in the classic 10-values/line
        # format, plus the optional box line.
        per_line = 10
        ncoord = natom * 3
        ncoord_lines = (ncoord + per_line - 1) // per_line
        # Count numeric values after the title.  This is robust to wrapping.
        values = []
        for line in lines[1:]:
            values.extend(line.split())
        nvals = len(values)
        if nvals % ncoord == 0:
            return False
        # If a 3-value box is present, nvals = nframes*(ncoord+3).
        if nvals % (ncoord + 3) == 0:
            return True
        # Fall back to the line-count heuristic.
        return len(lines) > 1 and (len(lines) - 1) % (ncoord_lines + 1) == 0
    except Exception:
        return False


def _read_mdcrd(path: str, natom: int) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray]]]:
    from parmed.amber.asciicrd import AmberMdcrd
    hasbox = _mdcrd_has_box(path, natom)
    trj = AmberMdcrd(path, natom=natom, hasbox=hasbox, mode="r")
    coords = np.asarray(trj.coordinates, dtype=float)
    box = np.asarray(trj.box, dtype=float) if hasbox else None
    for k in range(coords.shape[0]):
        if box is not None and k < len(box):
            b = box[k]
            b6 = np.array([b[0], b[1], b[2], 90.0, 90.0, 90.0])
        else:
            b6 = None
        yield coords[k], b6


def _read_netcdf(path: str) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray]]]:
    from parmed.amber.netcdffiles import NetCDFTraj
    trj = NetCDFTraj(path, mode="r")
    coords = np.asarray(trj.coordinates, dtype=float)
    box = getattr(trj, "box", None)
    box = np.asarray(box, dtype=float) if box is not None else None
    for k in range(coords.shape[0]):
        b6 = box[k] if box is not None and k < len(box) else None
        yield coords[k], b6


def _read_mdanalysis(path: str, top_file: str) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray]]]:
    try:
        import MDAnalysis as mda
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            f"Reading '{path}' requires MDAnalysis (pip install MDAnalysis), or "
            "convert it first with: gmx trjconv -f traj.xtc -s topol.tpr "
            "-sep -o frame_.gro"
        ) from exc
    u = mda.Universe(top_file, path)
    for ts in u.trajectory:
        coords = ts.positions.copy()
        dims = ts.dimensions
        box = np.asarray(dims, dtype=float) if dims is not None else None
        yield coords, box


def iter_frames(files: Sequence[str], natom: int, top_file: str
                ) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray], str]]:
    """Yield (coords, box, label) for every frame in the given files."""
    for path in files:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".gro":
            coords, box, _ = _read_gro(path)
            yield coords, box, path
        elif ext in (".rst7", ".inpcrd", ".restrt"):
            coords, box, _ = _read_rst7(path)
            yield coords, box, path
        elif ext in (".mdcrd", ".crd"):
            for coords, box in _read_mdcrd(path, natom):
                yield coords, box, path
        elif ext in (".nc", ".ncdf", ".netcdf"):
            for coords, box in _read_netcdf(path):
                yield coords, box, path
        elif ext in (".xtc", ".trr"):
            for coords, box in _read_mdanalysis(path, top_file):
                yield coords, box, path
        elif ext == ".pdb":
            struc = pmd.load_file(path)
            yield np.asarray(struc.coordinates, dtype=float), (
                np.asarray(struc.box, dtype=float) if struc.box is not None else None
            ), path
        else:
            raise ValueError(
                f"Unsupported coordinate format '{ext}' for {path}.  "
                "Supported: .gro, .rst7/.inpcrd, .mdcrd, .nc, .xtc/.trr (via "
                "MDAnalysis), .pdb"
            )


# ============================================================================
# 8. Whole-molecule PBC reconstruction
# ============================================================================

def unwrap_molecules(coords: np.ndarray, bonds: Sequence[Tuple[int, int]],
                     box: Optional[Sequence[float]],
                     logger: logging.Logger) -> np.ndarray:
    """Make each bonded molecule whole and place molecules near the main one.

    This is a convenience for trajectories that were not pre-processed with
    `gmx trjconv -pbc mol`.  It is not a replacement for proper PBC treatment.
    """
    box_mat = box_to_matrix(box)
    if box_mat is None:
        logger.warning("--unwrap requested but no box information is available")
        return coords
    n = coords.shape[0]
    coords = coords.copy()

    # Connected components (molecules) from the bond graph.
    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    adjacency: List[List[int]] = [[] for _ in range(n)]
    for i, j in bonds:
        union(i, j)
        adjacency[i].append(j)
        adjacency[j].append(i)

    components: Dict[int, List[int]] = {}
    for i in range(n):
        components.setdefault(find(i), []).append(i)

    # Walk each molecule's bond tree and apply the minimum image per bond.
    for root, atoms in components.items():
        if len(atoms) < 2:
            continue
        visited = {root}
        stack = [root]
        while stack:
            a = stack.pop()
            for b in adjacency[a]:
                if b in visited:
                    continue
                visited.add(b)
                d = minimum_image_vector(coords[b] - coords[a], box_mat)
                coords[b] = coords[a] + d
                stack.append(b)

    # Place every non-main molecule in the periodic image that best contacts
    # the main molecule.  Using centre-of-mass minimum imaging is unsafe for
    # elongated receptors: it can translate a bound ligand straight through
    # the protein.  Score images by inter-molecular contacts instead.
    if len(components) > 1:
        main_root = max(components, key=lambda r: len(components[r]))
        main_atoms = np.asarray(components[main_root], dtype=np.int64)
        main_xyz = coords[main_atoms]
        shifts = [np.zeros(3)]
        if box_mat is not None:
            for na in (-1, 0, 1):
                for nb in (-1, 0, 1):
                    for nc in (-1, 0, 1):
                        if na == nb == nc == 0:
                            continue
                        shifts.append(
                            na * box_mat[0] + nb * box_mat[1] + nc * box_mat[2]
                        )
        for root, atoms in components.items():
            if root == main_root:
                continue
            idx = np.asarray(atoms, dtype=np.int64)
            lig_xyz = coords[idx]
            zero_score, zero_clashes, zero_contacts = _contact_score(
                lig_xyz, main_xyz
            )
            if zero_clashes == 0 and zero_contacts > 0:
                continue  # already in a sensible image
            best_shift = np.zeros(3)
            best_score = zero_score
            best_norm = 0.0
            for shift in shifts[1:]:
                score, _, _ = _contact_score(lig_xyz + shift, main_xyz)
                norm = float(np.linalg.norm(shift))
                if score > best_score or (score == best_score and
                                          norm < best_norm):
                    best_score = score
                    best_shift = shift
                    best_norm = norm
            if np.any(best_shift):
                coords[idx] += best_shift

    return coords


def _contact_score(molecule_xyz: np.ndarray,
                   reference_xyz: np.ndarray) -> Tuple[int, int, int]:
    """Score an image by favourable contacts minus severe clashes."""
    if _HAVE_SCIPY:
        d, _ = cKDTree(reference_xyz).query(molecule_xyz, k=1)
    else:  # pragma: no cover - SciPy is normally present
        d = cdist(molecule_xyz, reference_xyz).min(axis=1)
    contacts = int(np.count_nonzero((d > 1.2) & (d < 4.5)))
    clashes = int(np.count_nonzero(d <= 1.2))
    return contacts - 10 * clashes, clashes, contacts


def check_broken_bonds(coords: np.ndarray, bonds: Sequence[Tuple[int, int]],
                       box: Optional[Sequence[float]],
                       logger: logging.Logger) -> None:
    if not bonds or box is None:
        return
    lengths = box_lengths_from_box(box)
    if lengths is None:
        return
    half_min = 0.5 * float(np.min(lengths))
    max_bond = 0.0
    for i, j in bonds:
        d = np.linalg.norm(coords[i] - coords[j])
        if d > max_bond:
            max_bond = d
    if max_bond > max(2.5, half_min):
        logger.warning(
            "Maximum bond length is %.2f A (box half-min %.2f A).  Molecules "
            "may be split across periodic images; consider --unwrap or "
            "'gmx trjconv -pbc mol -center'.", max_bond, half_min
        )


# ============================================================================
# 9. MM-GBSA engine
# ============================================================================

class MMGBSAEngine:
    def __init__(self, top: TopologyModel, config: Config, logger: logging.Logger):
        self.top = top
        self.config = config
        self.logger = logger
        top.build_pair_data()
        self.gb = GBModel(config)
        self.nonbonded = NonbondedModel(top)
        self.sasa = SASACalculator(config)

        # Sorted subsystem indices are essential for aligning local pair arrays
        # with the corresponding subset of the global pair list.
        self.rec_idx = np.asarray(sorted(top.receptor_idx), dtype=np.int64)
        self.lig_idx = np.asarray(sorted(top.ligand_idx), dtype=np.int64)
        self.cpx_idx = np.asarray(sorted(top.complex_idx), dtype=np.int64)
        self.iu_rec = np.triu_indices(self.rec_idx.size, k=1)
        self.iu_lig = np.triu_indices(self.lig_idx.size, k=1)

        i, j = top.iu
        self.cross = (top.rec_mask[i] & top.lig_mask[j]) | \
                     (top.lig_mask[i] & top.rec_mask[j])
        self.rec_pair = top.rec_mask[i] & top.rec_mask[j]
        self.lig_pair = top.lig_mask[i] & top.lig_mask[j]

        self.rec_res_indices = np.array(
            sorted(set(top.residx[self.rec_idx].tolist())), dtype=np.int64
        )

    # ------------------------------------------------------------------
    def _subsystem(self, coords: np.ndarray, idx: np.ndarray,
                   iu: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        c = coords[idx]
        q = self.top.charges[idx]
        r_gb = self.top.radii_gb[idx]
        born = self.gb.born_radii(c, r_gb, self.top.screen[idx])
        e_self = self.gb.self_energy(q, born)
        if idx.size >= 2:
            if _HAVE_SCIPY:
                r = cdist(c, c)
            else:  # pragma: no cover
                d = c[:, None, :] - c[None, :, :]
                r = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
            e_pair = self.gb.pair_energies(
                r[iu[0], iu[1]], q[iu[0]], q[iu[1]], born[iu[0]], born[iu[1]]
            )
        else:
            r = None
            e_pair = np.zeros(0, dtype=float)
        areas = self.sasa.atom_areas(c, self.top.radii_sasa[idx])
        return {
            "born": born,
            "gb_self": e_self,
            "gb_pair": e_pair,
            "gb": e_self + float(np.sum(e_pair)),
            "areas": areas,
            "sasa": self.sasa.energy(areas),
            "r": r,
        }

    # ------------------------------------------------------------------
    def compute_frame(self, coords: np.ndarray) -> Dict[str, Any]:
        if coords.shape[0] != self.top.n_atoms:
            raise ValueError(
                f"Coordinate frame has {coords.shape[0]} atoms but the topology "
                f"has {self.top.n_atoms}"
            )
        coords = np.asarray(coords, dtype=float)
        if _HAVE_SCIPY:
            r_full = cdist(coords, coords)
        else:  # pragma: no cover
            d = coords[:, None, :] - coords[None, :, :]
            r_full = np.sqrt(np.einsum("ijk,ijk->ij", d, d))

        i, j = self.top.iu
        r_pairs = r_full[i, j]
        e_vdw, e_elec = self.nonbonded.pair_energies(r_pairs)

        # ---- absolute gas-phase terms --------------------------------
        vdw_c = float(np.sum(e_vdw))
        elec_c = float(np.sum(e_elec))
        vdw_r = float(np.sum(e_vdw[self.rec_pair]))
        elec_r = float(np.sum(e_elec[self.rec_pair]))
        vdw_l = float(np.sum(e_vdw[self.lig_pair]))
        elec_l = float(np.sum(e_elec[self.lig_pair]))

        vdw14_c = float(np.sum(e_vdw[self.top.one_four_mask]))
        elec14_c = float(np.sum(e_elec[self.top.one_four_mask]))

        # ---- complex GB (needs all atoms) ----------------------------
        born_c = self.gb.born_radii(coords, self.top.radii_gb, self.top.screen)
        gb_self_c = self.gb.self_energy(self.top.charges, born_c)
        gb_pair_c = self.gb.pair_energies(
            r_pairs, self.top.charges[i], self.top.charges[j],
            born_c[i], born_c[j]
        )
        gb_c = gb_self_c + float(np.sum(gb_pair_c))

        # ---- receptor / ligand GB ------------------------------------
        rec = self._subsystem(coords, self.rec_idx, self.iu_rec)
        lig = self._subsystem(coords, self.lig_idx, self.iu_lig)

        # ---- complex SASA --------------------------------------------
        areas_c = self.sasa.atom_areas(coords, self.top.radii_sasa)
        sasa_c = self.sasa.energy(areas_c)

        # ---- differences ---------------------------------------------
        delta = {
            "vdw": vdw_c - vdw_r - vdw_l,
            "elec": elec_c - elec_r - elec_l,
            "gb": gb_c - rec["gb"] - lig["gb"],
            "sasa": sasa_c - rec["sasa"] - lig["sasa"],
        }
        delta["total"] = delta["vdw"] + delta["elec"] + delta["gb"] + delta["sasa"]

        data: Dict[str, Any] = {
            "r_full": r_full,
            "r_pairs": r_pairs,
            "e_vdw": e_vdw,
            "e_elec": e_elec,
            "born_c": born_c,
            "gb_pair_c": gb_pair_c,
            "areas_c": areas_c,
            "rec": rec,
            "lig": lig,
            "vdw_complex": vdw_c,
            "elec_complex": elec_c,
            "gb_complex": gb_c,
            "sasa_complex": sasa_c,
            "vdw_receptor": vdw_r,
            "elec_receptor": elec_r,
            "gb_receptor": rec["gb"],
            "sasa_receptor": rec["sasa"],
            "vdw_ligand": vdw_l,
            "elec_ligand": elec_l,
            "gb_ligand": lig["gb"],
            "sasa_ligand": lig["sasa"],
            "vdw14_complex": vdw14_c,
            "elec14_complex": elec14_c,
        }
        for key, value in delta.items():
            data[f"delta_{key}"] = float(value)
        return data

    # ------------------------------------------------------------------
    def decompose(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Per-receptor-residue decomposition that sums exactly to Delta G.

        Terms assigned to each receptor residue:
          * One half of every VDW, Coulomb and GB pair term with the ligand;
            the other half is aggregated into the 'LIGAND' row.
          * GB self-energy change for atoms in the residue.
          * One half of every intra-receptor GB pair change (complex minus
            isolated receptor); the other half goes to the partner residue.
          * SASA change for atoms in the residue.

        This is the same 1/2-per-pair convention used by AMBER's idecomp
        per-residue decomposition, so per-residue values are directly
        comparable with MMPBSA.py.  Summing all residue rows and the ligand
        row gives the total Delta G exactly.
        """
        top = self.top
        i, j = top.iu
        n_res = top.n_residues
        kfac = self.gb.kfac

        vdw_res = np.zeros(n_res, dtype=float)
        elec_res = np.zeros(n_res, dtype=float)
        gb_res = np.zeros(n_res, dtype=float)
        sasa_res = np.zeros(n_res, dtype=float)

        # Cross terms: half to the receptor residue, half to the ligand row.
        cross = self.cross
        ri = np.where(top.rec_mask[i[cross]], i[cross], j[cross])
        np.add.at(vdw_res, top.residx[ri], 0.5 * data["e_vdw"][cross])
        np.add.at(elec_res, top.residx[ri], 0.5 * data["e_elec"][cross])
        np.add.at(gb_res, top.residx[ri], 0.5 * data["gb_pair_c"][cross])
        lig_vdw = 0.5 * float(np.sum(data["e_vdw"][cross]))
        lig_elec = 0.5 * float(np.sum(data["e_elec"][cross]))
        lig_gb_cross = 0.5 * float(np.sum(data["gb_pair_c"][cross]))

        # GB self-energy change for receptor atoms.
        q = top.charges
        born_c = data["born_c"]
        born_r = data["rec"]["born"]
        self_c_rec = -0.5 * kfac * q[self.rec_idx] ** 2 / born_c[self.rec_idx]
        self_r_rec = -0.5 * kfac * q[self.rec_idx] ** 2 / born_r
        np.add.at(gb_res, top.residx[self.rec_idx], self_c_rec - self_r_rec)

        # Intra-receptor GB pair change: split equally between both residues.
        d_rec = data["gb_pair_c"][self.rec_pair] - data["rec"]["gb_pair"]
        a_idx = i[self.rec_pair]
        b_idx = j[self.rec_pair]
        np.add.at(gb_res, top.residx[a_idx], 0.5 * d_rec)
        np.add.at(gb_res, top.residx[b_idx], 0.5 * d_rec)

        # SASA change for receptor atoms.
        dA_rec = data["areas_c"][self.rec_idx] - data["rec"]["areas"]
        np.add.at(sasa_res, top.residx[self.rec_idx],
                  self.sasa.surface_tension * dA_rec)

        # Ligand remainder.
        born_l = data["lig"]["born"]
        self_c_lig = -0.5 * kfac * q[self.lig_idx] ** 2 / born_c[self.lig_idx]
        self_l_lig = -0.5 * kfac * q[self.lig_idx] ** 2 / born_l
        gb_lig = lig_gb_cross
        gb_lig += float(np.sum(self_c_lig - self_l_lig))
        gb_lig += float(np.sum(
            data["gb_pair_c"][self.lig_pair] - data["lig"]["gb_pair"]
        ))
        dA_lig = data["areas_c"][self.lig_idx] - data["lig"]["areas"]
        sasa_lig = self.sasa.surface_tension * float(np.sum(dA_lig))

        return {
            "res_index": self.rec_res_indices,
            "vdw": vdw_res[self.rec_res_indices],
            "elec": elec_res[self.rec_res_indices],
            "gb": gb_res[self.rec_res_indices],
            "sasa": sasa_res[self.rec_res_indices],
            "ligand_vdw": np.array([lig_vdw]),
            "ligand_elec": np.array([lig_elec]),
            "ligand_gb": np.array([gb_lig]),
            "ligand_sasa": np.array([sasa_lig]),
        }

    # ------------------------------------------------------------------
    def check_decomposition(self, data: Dict[str, Any],
                            decomp: Dict[str, np.ndarray]) -> float:
        """Return the residual between the decomposition and total Delta G."""
        rec_total = float(
            np.sum(decomp["vdw"] + decomp["elec"] + decomp["gb"] + decomp["sasa"])
        )
        lig_total = float(
            decomp["ligand_vdw"][0] + decomp["ligand_elec"][0]
            + decomp["ligand_gb"][0] + decomp["ligand_sasa"][0]
        )
        return rec_total + lig_total - data["delta_total"]


# ============================================================================
# 10. Output helpers
# ============================================================================

DELTA_KEYS = ["vdw", "elec", "gb", "sasa", "total"]
ABS_KEYS = [
    "vdw_complex", "vdw_receptor", "vdw_ligand",
    "elec_complex", "elec_receptor", "elec_ligand",
    "gb_complex", "gb_receptor", "gb_ligand",
    "sasa_complex", "sasa_receptor", "sasa_ligand",
    "vdw14_complex", "elec14_complex",
]


def stats(values: Sequence[float]) -> Tuple[float, float, float]:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(a))
    std = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
    sem = std / math.sqrt(a.size) if a.size > 0 else float("nan")
    return mean, std, sem


def write_frame_csv(path: str, rows: Sequence[Dict[str, Any]], config: Config) -> None:
    keys = ["delta_" + k for k in DELTA_KEYS] + ABS_KEYS
    headers = [k[len("delta_"):] if k.startswith("delta_") else k for k in keys]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame"] + headers)
        for idx, row in enumerate(rows, start=1):
            writer.writerow([idx] + [row[k] for k in keys])
        means = [stats([row[k] for row in rows])[0] for k in keys]
        stds = [stats([row[k] for row in rows])[1] for k in keys]
        writer.writerow(["AVERAGE"] + means)
        writer.writerow(["STDDEV"] + stds)


def write_detailed_frame_csv(path: str, rows: Sequence[Dict[str, Any]], config: Config) -> None:
    """Write per-frame energies in MMPBSA.py style: complex, receptor, ligand, delta."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame",
            # Complex
            "vdw_complex", "elec_complex", "gb_complex", "sasa_complex",
            "vdw14_complex", "elec14_complex",
            # Receptor
            "vdw_receptor", "elec_receptor", "gb_receptor", "sasa_receptor",
            # Ligand
            "vdw_ligand", "elec_ligand", "gb_ligand", "sasa_ligand",
            # Delta
            "delta_vdw", "delta_elec", "delta_gb", "delta_sasa", "delta_total",
        ])
        for idx, row in enumerate(rows, start=1):
            writer.writerow([
                idx,
                row["vdw_complex"], row["elec_complex"], row["gb_complex"], row["sasa_complex"],
                row["vdw14_complex"], row["elec14_complex"],
                row["vdw_receptor"], row["elec_receptor"], row["gb_receptor"], row["sasa_receptor"],
                row["vdw_ligand"], row["elec_ligand"], row["gb_ligand"], row["sasa_ligand"],
                row["delta_vdw"], row["delta_elec"], row["delta_gb"], row["delta_sasa"], row["delta_total"],
            ])
        # Average row
        def avg(key):
            return float(np.mean([r[key] for r in rows]))
        writer.writerow([
            "AVERAGE",
            avg("vdw_complex"), avg("elec_complex"), avg("gb_complex"), avg("sasa_complex"),
            avg("vdw14_complex"), avg("elec14_complex"),
            avg("vdw_receptor"), avg("elec_receptor"), avg("gb_receptor"), avg("sasa_receptor"),
            avg("vdw_ligand"), avg("elec_ligand"), avg("gb_ligand"), avg("sasa_ligand"),
            avg("delta_vdw"), avg("delta_elec"), avg("delta_gb"), avg("delta_sasa"), avg("delta_total"),
        ])


def write_decomposition_csv(path: str, top: TopologyModel,
                            rows: Sequence[Dict[str, np.ndarray]],
                            logger: logging.Logger) -> None:
    if not rows:
        return
    res_indices = rows[0]["res_index"]
    n = res_indices.size
    data = {
        "vdw": np.zeros(n), "elec": np.zeros(n),
        "gb": np.zeros(n), "sasa": np.zeros(n),
    }
    for row in rows:
        for key in data:
            data[key] += row[key]
    count = float(len(rows))
    for key in data:
        data[key] /= count
    total = data["vdw"] + data["elec"] + data["gb"] + data["sasa"]

    lig_vdw = float(np.mean([r["ligand_vdw"][0] for r in rows]))
    lig_elec = float(np.mean([r["ligand_elec"][0] for r in rows]))
    lig_gb = float(np.mean([r["ligand_gb"][0] for r in rows]))
    lig_sasa = float(np.mean([r["ligand_sasa"][0] for r in rows]))

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "residue_index", "resname", "resid",
            "vdw", "elec", "gb", "sasa", "total"
        ])
        for k, ri in enumerate(res_indices):
            label = top.residue_labels[ri]
            resname = top.structure.residues[ri].name
            resid = int(getattr(top.structure.residues[ri], "number", ri + 1))
            writer.writerow([
                int(ri + 1), resname, resid,
                data["vdw"][k], data["elec"][k], data["gb"][k],
                data["sasa"][k], total[k],
            ])
        writer.writerow([
            "LIGAND", "LIG", "",
            lig_vdw, lig_elec, lig_gb, lig_sasa,
            lig_vdw + lig_elec + lig_gb + lig_sasa,
        ])
        writer.writerow([
            "TOTAL", "", "",
            float(np.sum(data["vdw"])) + lig_vdw,
            float(np.sum(data["elec"])) + lig_elec,
            float(np.sum(data["gb"])) + lig_gb,
            float(np.sum(data["sasa"])) + lig_sasa,
            float(np.sum(total)) + lig_vdw + lig_elec + lig_gb + lig_sasa,
        ])
    logger.info("Per-residue decomposition written to %s", path)


# ============================================================================
# 11. CLI
# ============================================================================

def resolve_ligand_names(ligand_arg: Optional[str],
                         resid_range: Optional[Tuple[int, int]],
                         mask: Optional[str]) -> List[str]:
    """Resolve the --ligand selection without leaking the 'LIG' default.

    An explicit --ligand always wins.  Otherwise the default 'LIG' is used
    only when no residue-range or mask selector was supplied, so an explicit
    range/mask never gets silently unioned with a residue named LIG.
    """
    if ligand_arg is not None:
        return [x.strip() for x in ligand_arg.split(",") if x.strip()]
    if resid_range is None and mask is None:
        return [DEFAULT_LIGAND_RESNAME]
    return []


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MM-GBSA v4 (corrected OBC GB, AMBER-consistent 1-4 terms)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-t", "--top", required=True,
                   help="Topology: GROMACS .top or AMBER .prmtop/.parm7")
    p.add_argument("-c", "--coords", nargs="+", required=True,
                   help="Coordinate file(s): .gro, .rst7/.inpcrd, .mdcrd, .nc, "
                        ".xtc/.trr (MDAnalysis), .pdb.  Shell globs are allowed.")
    p.add_argument("--ligand", default=None,
                   help="Comma-separated ligand residue name(s).  Defaults to "
                        f"'{DEFAULT_LIGAND_RESNAME}' only when neither "
                        "--ligand-resid-range nor --ligand-mask is given.")
    p.add_argument("--ligand-resid-range", type=parse_resid_range, default=None,
                   help="Ligand residue range, 1-based inclusive (e.g. 167-242)")
    p.add_argument("--ligand-mask", default=None,
                   help="AMBER mask for the ligand, e.g. ':LIG' or ':167-242'")
    p.add_argument("--strip", default=",".join(DEFAULT_STRIP_RESNAMES),
                   help="Comma-separated solvent/ion residue names to ignore")
    p.add_argument("--no-strip", action="store_true",
                   help="Do not automatically strip solvent/ions")

    p.add_argument("--igb", type=int, choices=[2, 5], default=5,
                   help="GB model: 2=OBC-I, 5=OBC-II")
    p.add_argument("--eps-solute", type=float, default=DEFAULT_EPS_SOLUTE)
    p.add_argument("--eps-solvent", type=float, default=DEFAULT_EPS_SOLVENT)
    p.add_argument("--radii", choices=["auto", "mbondi", "mbondi2", "mbondi3", "bondi"],
                   default="auto", help="GB radii set")
    p.add_argument("--obc-alpha", type=float, default=None)
    p.add_argument("--obc-beta", type=float, default=None)
    p.add_argument("--obc-gamma", type=float, default=None)
    p.add_argument("--obc-offset", type=float, default=None)
    p.add_argument("--gb-radii-cutoff", type=float, default=DEFAULT_GB_RADII_CUTOFF,
                   help="Cutoff (A) for the HCT Born-radii descreening sum; "
                        "AMBER's mmpbsa_py_energy uses 25.0.  Use 0 to disable.")

    p.add_argument("--probe", type=float, default=DEFAULT_PROBE_RADIUS,
                   help="Solvent probe radius (A) for SASA")
    p.add_argument("--sasa-radii", choices=["bondi", "gb"], default="bondi",
                   help="Radii used by Shrake-Rupley SASA: Bondi (standard) or "
                        "the GB radii from the topology (closer to AMBER LCPO)")
    p.add_argument("--surface-tension", type=float, default=DEFAULT_SURFACE_TENSION,
                   help="Nonpolar surface tension (kcal/mol/A^2)")
    p.add_argument("--surface-offset", type=float, default=0.0)
    p.add_argument("--sasa-points", type=int, default=240,
                   help="Shrake-Rupley sphere points per atom")

    p.add_argument("--stride", type=int, default=1,
                   help="Process every Nth frame")
    p.add_argument("--start", type=int, default=1,
                   help="First frame to process (1-based)")
    p.add_argument("--end", type=int, default=None,
                   help="Last frame to process (1-based, inclusive)")
    p.add_argument("--unwrap", action="store_true",
                   help="Reconstruct whole molecules with minimum image (experimental)")

    p.add_argument("--decomp", action="store_true",
                   help="Per-receptor-residue decomposition")
    p.add_argument("--decomp-csv", default="mmgbsa_decomposition.csv")
    p.add_argument("--output", default="mmgbsa_results.csv")
    p.add_argument("--json", dest="json_output", default=None,
                   help="Also write a JSON summary to this path")
    p.add_argument("--gmx-lib", default=None,
                   help="GROMACS topology library dir (sets GMXLIB before reading .top)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--version", action="version", version="MM-GBSA v4")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logger = setup_logging(args.verbose)

    if args.gmx_lib:
        os.environ["GMXLIB"] = args.gmx_lib

    ligand_names = resolve_ligand_names(
        args.ligand, args.ligand_resid_range, args.ligand_mask
    )
    strip_names = {x.strip() for x in args.strip.split(",") if x.strip()}
    config = Config(
        top_file=args.top,
        coord_files=list(args.coords),
        ligand_resnames=ligand_names,
        ligand_resid_range=args.ligand_resid_range,
        ligand_mask=args.ligand_mask,
        strip_resnames=strip_names,
        no_strip=args.no_strip,
        igb=args.igb,
        eps_solute=args.eps_solute,
        eps_solvent=args.eps_solvent,
        radii_set=args.radii,
        obc_alpha=args.obc_alpha,
        obc_beta=args.obc_beta,
        obc_gamma=args.obc_gamma,
        obc_offset=args.obc_offset,
        gb_radii_cutoff=args.gb_radii_cutoff,
        probe_radius=args.probe,
        sasa_radii_set=args.sasa_radii,
        surface_tension=args.surface_tension,
        surface_offset=args.surface_offset,
        sasa_points=args.sasa_points,
        stride=args.stride,
        start_frame=args.start,
        end_frame=args.end,
        unwrap=args.unwrap,
        output_csv=args.output,
        decomp=args.decomp,
        decomp_csv=args.decomp_csv,
        json_output=args.json_output,
        verbose=args.verbose,
        gmx_lib=args.gmx_lib,
    )

    t0 = time.time()
    if not os.path.exists(config.top_file):
        logger.error("Topology file not found: %s", config.top_file)
        return 1

    try:
        structure = pmd.load_file(config.top_file)
    except Exception as exc:
        logger.error("Could not read topology %s: %s", config.top_file, exc)
        return 1

    try:
        top = TopologyModel(structure, config, logger)
    except Exception as exc:
        logger.error("Topology setup failed: %s", exc)
        return 1

    summary = top.summary()
    logger.info(
        "System: %d atoms, %d residues | ligand %d atoms (%s) | receptor %d | "
        "stripped %d",
        summary["n_atoms"], summary["n_residues"],
        summary["n_ligand_atoms"], ", ".join(summary["ligand_resnames"]),
        summary["n_receptor_atoms"], summary["n_stripped_atoms"],
    )
    logger.info(
        "Pairs: %d exclusions, %d 1-4 pairs | combining=%s fudgeLJ=%.4f fudgeQQ=%.4f",
        summary["n_excluded_pairs"], summary["n_one_four_pairs"],
        summary["combining_rule"], summary["fudge_lj"], summary["fudge_qq"],
    )
    params = OBC_PARAMETERS[config.igb]
    logger.info(
        "GB: igb=%d alpha=%.4g beta=%.4g gamma=%.4g offset=%.3f A | "
        "eps_solute=%.2f eps_solvent=%.2f | radii cutoff=%s A",
        config.igb,
        config.obc_alpha if config.obc_alpha is not None else params["alpha"],
        config.obc_beta if config.obc_beta is not None else params["beta"],
        config.obc_gamma if config.obc_gamma is not None else params["gamma"],
        config.obc_offset if config.obc_offset is not None else params["offset"],
        config.eps_solute, config.eps_solvent,
        f"{config.gb_radii_cutoff:g}" if config.gb_radii_cutoff else "none",
    )
    logger.info(
        "SASA: %s radii, probe=%.2f A, gamma=%.4f kcal/mol/A^2, %d points",
        config.sasa_radii_set, config.probe_radius, config.surface_tension,
        config.sasa_points,
    )

    coord_files = expand_coord_files(config.coord_files)
    if not coord_files:
        logger.error("No coordinate files found for: %s", " ".join(config.coord_files))
        return 1

    try:
        engine = MMGBSAEngine(top, config, logger)
    except Exception as exc:
        logger.error("Engine setup failed: %s", exc)
        return 1

    rows: List[Dict[str, Any]] = []
    decomp_rows: List[Dict[str, np.ndarray]] = []
    frame_counter = 0
    used_counter = 0
    tried = 0
    t_frames = time.time()

    try:
        for coords, box, label in iter_frames(coord_files, top.n_atoms, config.top_file):
            frame_counter += 1
            if frame_counter < config.start_frame:
                continue
            if config.end_frame is not None and frame_counter > config.end_frame:
                break
            if (frame_counter - config.start_frame) % max(1, config.stride) != 0:
                continue
            tried += 1
            if coords.shape[0] != top.n_atoms:
                raise ValueError(
                    f"{label}: coordinate frame has {coords.shape[0]} atoms but "
                    f"topology has {top.n_atoms}"
                )
            if config.unwrap:
                coords = unwrap_molecules(coords, top.bonds, box, logger)
            else:
                check_broken_bonds(coords, top.bonds, box, logger)

            data = engine.compute_frame(coords)
            rows.append(data)
            used_counter += 1
            if config.decomp:
                decomp_rows.append(engine.decompose(data))

            if used_counter == 1 or used_counter % 10 == 0 or config.verbose:
                logger.info(
                    "frame %d (%s): dG = %8.3f kcal/mol  "
                    "[vdw %7.2f, elec %8.2f, gb %8.2f, sasa %6.2f]",
                    frame_counter, os.path.basename(label), data["delta_total"],
                    data["delta_vdw"], data["delta_elec"],
                    data["delta_gb"], data["delta_sasa"],
                )
    except KeyboardInterrupt:
        logger.warning("Interrupted; writing results for completed frames")
    except Exception as exc:
        logger.error("Frame processing failed: %s", exc)
        if config.verbose:
            import traceback
            traceback.print_exc()
        return 1

    if not rows:
        logger.error("No frames were processed")
        return 1

    # ---- summary -----------------------------------------------------
    logger.info("=" * 74)
    logger.info("Processed %d frame(s) in %.1f s", len(rows), time.time() - t_frames)
    logger.info("%-12s %12s %12s %12s", "component", "mean", "std", "sem")
    logger.info("-" * 74)
    for key in DELTA_KEYS:
        mean, std, sem = stats([r[f"delta_{key}"] for r in rows])
        logger.info("%-12s %12.3f %12.3f %12.3f", key, mean, std, sem)
    logger.info("-" * 74)
    mean14_vdw, _, _ = stats([r["vdw14_complex"] for r in rows])
    mean14_elec, _, _ = stats([r["elec14_complex"] for r in rows])
    logger.info("complex 1-4 VDW  = %12.3f kcal/mol", mean14_vdw)
    logger.info("complex 1-4 EEL  = %12.3f kcal/mol", mean14_elec)
    logger.info("=" * 74)

    if config.decomp:
        residuals = [engine.check_decomposition(r, d)
                     for r, d in zip(rows, decomp_rows)]
        max_resid = max(abs(x) for x in residuals)
        logger.info(
            "Decomposition check: max |sum(residues) - dG| = %.3e kcal/mol",
            max_resid,
        )
        if max_resid > 1e-6:
            logger.warning(
                "Decomposition does not sum exactly to the total; this indicates "
                "a bug rather than a physical effect"
            )
        write_decomposition_csv(config.decomp_csv, top, decomp_rows, logger)

    write_frame_csv(config.output_csv, rows, config)
    logger.info("Frame results written to %s", config.output_csv)

    # Write detailed per-frame energy CSV (MMPBSA.py style)
    if len(rows) > 0:
        detail_csv = config.output_csv.replace(".csv", "_detail.csv")
        write_detailed_frame_csv(detail_csv, rows, config)
        logger.info("Detailed frame results written to %s", detail_csv)

    if config.json_output:
        gb_params = OBC_PARAMETERS[config.igb]
        payload = {
            "system": summary,
            "config": {
                "igb": config.igb,
                "obc_alpha": (config.obc_alpha if config.obc_alpha is not None
                              else gb_params["alpha"]),
                "obc_beta": (config.obc_beta if config.obc_beta is not None
                             else gb_params["beta"]),
                "obc_gamma": (config.obc_gamma if config.obc_gamma is not None
                              else gb_params["gamma"]),
                "obc_offset": (config.obc_offset if config.obc_offset is not None
                               else gb_params["offset"]),
                "gb_radii_cutoff": config.gb_radii_cutoff,
                "radii_set": config.radii_set,
                "eps_solute": config.eps_solute,
                "eps_solvent": config.eps_solvent,
                "sasa_radii_set": config.sasa_radii_set,
                "surface_tension": config.surface_tension,
                "surface_offset": config.surface_offset,
                "probe_radius": config.probe_radius,
                "sasa_points": config.sasa_points,
                "ligand_selection": {
                    "resnames": list(config.ligand_resnames),
                    "resid_range": (list(config.ligand_resid_range)
                                    if config.ligand_resid_range else None),
                    "mask": config.ligand_mask,
                },
                "strip_resnames": sorted(config.strip_resnames),
                "no_strip": config.no_strip,
                "decomposition": bool(config.decomp),
                "frames": len(rows),
            },
            "delta": {
                key: dict(zip(("mean", "std", "sem"),
                              stats([r[f"delta_{key}"] for r in rows])))
                for key in DELTA_KEYS
            },
            "components": {
                "complex": {
                    "vdw": dict(zip(("mean", "std", "sem"),
                                    stats([r["vdw_complex"] for r in rows]))),
                    "elec": dict(zip(("mean", "std", "sem"),
                                     stats([r["elec_complex"] for r in rows]))),
                    "gb": dict(zip(("mean", "std", "sem"),
                                   stats([r["gb_complex"] for r in rows]))),
                    "sasa": dict(zip(("mean", "std", "sem"),
                                     stats([r["sasa_complex"] for r in rows]))),
                    "vdw14": dict(zip(("mean", "std", "sem"),
                                      stats([r["vdw14_complex"] for r in rows]))),
                    "elec14": dict(zip(("mean", "std", "sem"),
                                       stats([r["elec14_complex"] for r in rows]))),
                },
                "receptor": {
                    "vdw": dict(zip(("mean", "std", "sem"),
                                    stats([r["vdw_receptor"] for r in rows]))),
                    "elec": dict(zip(("mean", "std", "sem"),
                                     stats([r["elec_receptor"] for r in rows]))),
                    "gb": dict(zip(("mean", "std", "sem"),
                                   stats([r["gb_receptor"] for r in rows]))),
                    "sasa": dict(zip(("mean", "std", "sem"),
                                     stats([r["sasa_receptor"] for r in rows]))),
                },
                "ligand": {
                    "vdw": dict(zip(("mean", "std", "sem"),
                                    stats([r["vdw_ligand"] for r in rows]))),
                    "elec": dict(zip(("mean", "std", "sem"),
                                     stats([r["elec_ligand"] for r in rows]))),
                    "gb": dict(zip(("mean", "std", "sem"),
                                   stats([r["gb_ligand"] for r in rows]))),
                    "sasa": dict(zip(("mean", "std", "sem"),
                                     stats([r["sasa_ligand"] for r in rows]))),
                },
            },
        }
        if config.decomp and decomp_rows:
            res_indices = decomp_rows[0]["res_index"]
            n = res_indices.size
            decomp_mean = {
                "vdw": np.zeros(n), "elec": np.zeros(n),
                "gb": np.zeros(n), "sasa": np.zeros(n),
            }
            for row in decomp_rows:
                for key in decomp_mean:
                    decomp_mean[key] += row[key]
            count = float(len(decomp_rows))
            for key in decomp_mean:
                decomp_mean[key] /= count
            decomp_total = decomp_mean["vdw"] + decomp_mean["elec"] + decomp_mean["gb"] + decomp_mean["sasa"]
            lig_vdw = float(np.mean([r["ligand_vdw"][0] for r in decomp_rows]))
            lig_elec = float(np.mean([r["ligand_elec"][0] for r in decomp_rows]))
            lig_gb = float(np.mean([r["ligand_gb"][0] for r in decomp_rows]))
            lig_sasa = float(np.mean([r["ligand_sasa"][0] for r in decomp_rows]))
            payload["decomposition"] = {
                "residues": [
                    {
                        "residue_index": int(res_indices[k] + 1),
                        "resname": top.structure.residues[res_indices[k]].name,
                        "resid": int(getattr(top.structure.residues[res_indices[k]], "number", res_indices[k] + 1)),
                        "vdw": float(decomp_mean["vdw"][k]),
                        "elec": float(decomp_mean["elec"][k]),
                        "gb": float(decomp_mean["gb"][k]),
                        "sasa": float(decomp_mean["sasa"][k]),
                        "total": float(decomp_total[k]),
                    }
                    for k in range(n)
                ],
                "ligand": {
                    "vdw": lig_vdw,
                    "elec": lig_elec,
                    "gb": lig_gb,
                    "sasa": lig_sasa,
                    "total": lig_vdw + lig_elec + lig_gb + lig_sasa,
                },
            }
        with open(config.json_output, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("JSON summary written to %s", config.json_output)

    logger.info("Total wall time: %.1f s", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
