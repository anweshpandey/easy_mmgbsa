#!/usr/bin/env python3
"""
MM-GBSA Docking Rescoring Engine
================================

Physics-based rescoring for protein–ligand complexes from molecular docking
or virtual screening. Handles the critical topology/parameter preparation
that docked poses require before MM-GBSA calculation.

Key features:
- Protein preparation with standard AMBER force fields (ff14SB, ff19SB)
- Ligand parameterization with GAFF2 + AM1-BCC charges (via OpenFF toolkit)
- Automatic protonation state assignment (via OpenFF/OpenEye or PROPKA)
- Complex assembly with proper topology
- Batch processing of multiple ligand poses
- Ranked output with per-pose MM-GBSA energies

Usage:
    # Single protein + multiple ligand poses (SDF/MOL2/PDB)
    python3 mmgbsa_docking.py -p protein.pdb -l ligands.sdf -o results.csv

    # With explicit protein force field and ligand charge method
    python3 mmgbsa_docking.py -p protein.pdb -l poses/ --ff ff19SB --charge am1bcc \
        --decomp decomp.csv --json summary.json

    # Provide pre-parameterized ligand (MOL2 with charges) + protein PDB
    python3 mmgbsa_docking.py -p protein.pdb -l ligand.mol2 --prepared-ligand

Requirements:
    openff-toolkit (for ligand GAFF2/AM1-BCC parameterization)
    openff-recharge (optional, for AM1-BCC via OpenEye/AM1BCC)
    propka3 (optional, for protein protonation states)
    pdbfixer (optional, for protein structure fixing)

The engine uses the validated mmgbsa_amb.py MM-GBSA implementation under the hood.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import parmed as pmd
except ImportError:
    sys.exit("mmgbsa_docking requires ParmEd (pip install parmed)")

# Optional dependencies - checked at runtime
_HAVE_OPENFF = False
_HAVE_OPENFF_RECHARGE = False
_HAVE_PROPKA = False
_HAVE_PDBFIXER = False
_HAVE_RDKIT = False

try:
    from openff.toolkit import ForceField, Molecule, Topology
    from openff.toolkit.typing.engines.smirnoff import SMIRNOFFParameterHandler
    _HAVE_OPENFF = True
except ImportError:
    pass

try:
    from openff.recharge import Recharge
    _HAVE_OPENFF_RECHARGE = True
except ImportError:
    pass

try:
    from propka import run as propka_run
    _HAVE_PROPKA = True
except ImportError:
    pass

try:
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
    _HAVE_PDBFIXER = True
except ImportError:
    pass

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    _HAVE_RDKIT = True
except ImportError:
    pass

# Import the validated MM-GBSA engine
sys.path.insert(0, os.path.dirname(__file__))
import mmgbsa_amb as v4

# ============================================================================
# Constants
# ============================================================================
DEFAULT_PROTEIN_FF = "ff14SB"
DEFAULT_LIGAND_FF = "gaff-2.11"
DEFAULT_CHARGE_METHOD = "am1bcc"  # or "am1bccelf10", "gasteiger", "mmff94"
DEFAULT_PH = 7.4
DEFAULT_IGB = 5
DEFAULT_EPS_SOLVENT = 78.3
DEFAULT_GB_CUTOFF = 25.0
DEFAULT_PROBE = 1.4
DEFAULT_SURFACE_TENSION = 0.0072
DEFAULT_SASA_POINTS = 240

# Standard protein residue names (3-letter codes)
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "HID", "HIP", "HIE",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "CYX", "CYD", "HSD", "HSE", "HSP"  # common variants
}

# ============================================================================
# Configuration
# ============================================================================
@dataclass
class DockingConfig:
    # Input files
    protein_pdb: str
    ligand_files: List[str]  # SDF, MOL2, PDB files or directory
    ligand_resname: str = "LIG"
    
    # Force fields
    protein_ff: str = DEFAULT_PROTEIN_FF  # ff14SB, ff19SB, ff03, etc.
    ligand_ff: str = DEFAULT_LIGAND_FF    # gaff-2.11, gaff-1.8, openff-2.0.0
    charge_method: str = DEFAULT_CHARGE_METHOD  # am1bcc, am1bccelf10, gasteiger, mmff94
    
    # Protein preparation
    ph: float = DEFAULT_PH
    add_hydrogens: bool = True
    fix_protein: bool = True  # use pdbfixer to fix missing atoms/residues
    protonate: bool = True    # assign protonation states
    
    # Ligand handling
    prepared_ligand: bool = False  # ligand files already have charges/types
    max_ligands: Optional[int] = None  # limit for testing
    ligand_name_prefix: str = "LIG"
    
    # MM-GBSA settings
    igb: int = DEFAULT_IGB
    eps_solvent: float = DEFAULT_EPS_SOLVENT
    gb_cutoff: float = DEFAULT_GB_CUTOFF
    radii_set: str = "auto"  # auto, mbondi, mbondi2, mbondi3, bondi
    probe_radius: float = DEFAULT_PROBE
    surface_tension: float = DEFAULT_SURFACE_TENSION
    sasa_points: int = DEFAULT_SASA_POINTS
    
    # Output
    output_csv: str = "docking_mmgbsa_results.csv"
    decomp_csv: Optional[str] = None
    json_output: Optional[str] = None
    keep_intermediate: bool = False
    work_dir: Optional[str] = None
    
    # Processing
    verbose: bool = False
    n_procs: int = 1  # for future parallelization


# ============================================================================
# Logging
# ============================================================================
def setup_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("mmgbsa_docking")


# ============================================================================
# Protein Preparation
# ============================================================================
class ProteinPreparator:
    """Prepare protein structure for MM-GBSA: fix, protonate, add hydrogens, assign FF."""
    
    def __init__(self, config: DockingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.work_dir = config.work_dir or tempfile.mkdtemp(prefix="mmgbsa_docking_")
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
    
    def prepare(self, protein_pdb: str) -> Tuple[pmd.Structure, str]:
        """
        Prepare protein and return (ParmEd Structure, prepared PDB path).
        """
        self.logger.info("Preparing protein: %s", protein_pdb)
        
        # Step 1: Fix structure with pdbfixer if available
        fixed_pdb = protein_pdb
        if self.config.fix_protein and _HAVE_PDBFIXER:
            fixed_pdb = self._fix_with_pdbfixer(protein_pdb)
        elif self.config.fix_protein:
            self.logger.warning("pdbfixer not available; skipping protein structure fixing")
        
        # Step 2: Assign protonation states with PROPKA if available
        protonated_pdb = fixed_pdb
        if self.config.protonate and _HAVE_PROPKA:
            protonated_pdb = self._protonate_with_propka(fixed_pdb)
        elif self.config.protonate:
            self.logger.warning("propka3 not available; skipping protonation state assignment")
        
        # Step 3: Load with ParmEd and parametrize with AMBER FF
        structure = self._parametrize_protein(protonated_pdb)
        
        # Step 4: Save prepared protein PDB for reference
        prepared_pdb = os.path.join(self.work_dir, "protein_prepared.pdb")
        structure.save(prepared_pdb, overwrite=True)
        self.logger.info("Prepared protein saved to: %s", prepared_pdb)
        
        return structure, prepared_pdb
    
    def _fix_with_pdbfixer(self, pdb_path: str) -> str:
        """Fix missing atoms, residues, hydrogens using pdbfixer."""
        self.logger.info("Fixing protein structure with pdbfixer...")
        fixer = PDBFixer(filename=pdb_path)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH=self.config.ph)
        
        fixed_path = os.path.join(self.work_dir, "protein_fixed.pdb")
        with open(fixed_path, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        return fixed_path
    
    def _protonate_with_propka(self, pdb_path: str) -> str:
        """Assign protonation states using PROPKA."""
        self.logger.info("Assigning protonation states with PROPKA (pH=%.1f)...", self.config.ph)
        # PROPKA modifies the PDB in place or writes output
        output_path = os.path.join(self.work_dir, "protein_protonated.pdb")
        # Run propka command line
        try:
            result = subprocess.run(
                ["propka", "--pH", str(self.config.ph), "--output", output_path, pdb_path],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0 and os.path.exists(output_path):
                return output_path
            else:
                self.logger.warning("PROPKA failed: %s", result.stderr)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.logger.warning("PROPKA not available or failed: %s", e)
        return pdb_path
    
    def _parametrize_protein(self, pdb_path: str) -> pmd.Structure:
        """Load protein PDB and apply AMBER force field using ParmEd/tleap."""
        self.logger.info("Parametrizing protein with %s...", self.config.protein_ff)
        
        # Try to use tleap for proper AMBER parametrization
        structure = self._parametrize_with_tleap(pdb_path)
        if structure is not None:
            return structure
        
        # Fallback: load with ParmEd and use openff-toolkit if available
        self.logger.warning("tleap not available; using ParmEd + OpenFF fallback")
        return self._parametrize_with_openff(pdb_path)
    
    def _parametrize_with_tleap(self, pdb_path: str) -> Optional[pmd.Structure]:
        """Use tleap to properly parametrize protein with AMBER FF."""
        leap_script = f"""
source leaprc.protein.{self.config.protein_ff}
source leaprc.water.tip3p
prot = loadpdb {pdb_path}
saveamberparm prot {self.work_dir}/protein.prmtop {self.work_dir}/protein.inpcrd
savepdb prot {self.work_dir}/protein_leap.pdb
quit
"""
        leap_in = os.path.join(self.work_dir, "leap.in")
        with open(leap_in, "w") as f:
            f.write(leap_script)
        
        try:
            result = subprocess.run(
                ["tleap", "-f", leap_in],
                capture_output=True, text=True, timeout=300,
                cwd=self.work_dir
            )
            if result.returncode == 0:
                prmtop = os.path.join(self.work_dir, "protein.prmtop")
                if os.path.exists(prmtop):
                    self.logger.info("Protein parametrized successfully with tleap")
                    return pmd.load_file(prmtop)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.logger.debug("tleap not available: %s", e)
        return None
    
    def _parametrize_with_openff(self, pdb_path: str) -> pmd.Structure:
        """Fallback parametrization using OpenFF toolkit."""
        if not _HAVE_OPENFF:
            # Last resort: load with ParmEd directly (no FF parameters)
            self.logger.warning("OpenFF not available; loading PDB without force field parameters")
            return pmd.load_file(pdb_path)
        
        # Use OpenFF to assign SMIRNOFF parameters (not ideal for proteins but works)
        self.logger.info("Assigning parameters with OpenFF SMIRNOFF...")
        # This is a simplified approach - real protein FF needs proper residue templates
        # For now, load with ParmEd and we'll rely on the ligand having proper params
        structure = pmd.load_file(pdb_path)
        
        # Try to assign basic atom types from the FF
        try:
            ff = ForceField(f"{self.config.protein_ff}.offxml")
            # OpenFF protein FFs are limited; this is a placeholder
        except Exception:
            pass
        
        return structure


# ============================================================================
# Ligand Parameterization
# ============================================================================
class LigandParameterizer:
    """Generate GAFF2/AM1-BCC parameters for ligand molecules."""
    
    def __init__(self, config: DockingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.work_dir = config.work_dir or tempfile.mkdtemp(prefix="mmgbsa_docking_")
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
    
    def parameterize_ligands(self, ligand_files: List[str]) -> List[Tuple[str, pmd.Structure, str]]:
        """
        Parameterize ligand files and return list of (name, ParmEd Structure, residue_name).
        """
        self.logger.info("Parameterizing %d ligand file(s)...", len(ligand_files))
        
        # Expand input files (handle directories, globs)
        expanded_files = self._expand_ligand_files(ligand_files)
        if self.config.max_ligands:
            expanded_files = expanded_files[:self.config.max_ligands]
        
        results = []
        for i, lig_file in enumerate(expanded_files):
            try:
                name = self._get_ligand_name(lig_file, i)
                if self.config.prepared_ligand:
                    # Ligand already has charges and types (e.g., MOL2 from antechamber)
                    struct = pmd.load_file(lig_file)
                    results.append((name, struct, self.config.ligand_resname))
                else:
                    # Need to parameterize
                    struct, resname = self._parameterize_single(lig_file, name)
                    results.append((name, struct, resname))
            except Exception as e:
                self.logger.error("Failed to parameterize %s: %s", lig_file, e)
                if self.config.verbose:
                    import traceback
                    traceback.print_exc()
        
        self.logger.info("Successfully parameterized %d ligands", len(results))
        return results
    
    def _expand_ligand_files(self, files: List[str]) -> List[str]:
        """Expand directories and glob patterns."""
        expanded = []
        for f in files:
            path = Path(f)
            if path.is_dir():
                # Find all supported ligand files in directory
                for ext in (".sdf", ".mol2", ".pdb", ".mol", ".mae"):
                    expanded.extend(sorted(path.glob(f"*{ext}")))
            else:
                # Handle glob patterns
                matches = sorted(glob.glob(f))
                if matches:
                    expanded.extend(matches)
                elif path.exists():
                    expanded.append(str(path))
        return [str(p) for p in expanded]
    
    def _get_ligand_name(self, lig_file: str, index: int) -> str:
        """Generate a unique name for the ligand."""
        stem = Path(lig_file).stem
        # Sanitize name
        name = "".join(c if c.isalnum() or c in "_-" else "_" for c in stem)
        return f"{self.config.ligand_name_prefix}_{name}_{index}"
    
    def _parameterize_single(self, lig_file: str, name: str) -> Tuple[pmd.Structure, str]:
        """Parameterize a single ligand file."""
        self.logger.info("Parameterizing ligand: %s", lig_file)
        
        # Load molecule with RDKit or OpenFF
        if _HAVE_OPENFF:
            return self._parameterize_with_openff(lig_file, name)
        elif _HAVE_RDKIT:
            return self._parameterize_with_rdkit_antechamber(lig_file, name)
        else:
            raise RuntimeError(
                "No ligand parameterization backend available. "
                "Install openff-toolkit (recommended) or rdkit+AmberTools."
            )
    
    def _parameterize_with_openff(self, lig_file: str, name: str) -> Tuple[pmd.Structure, str]:
        """Parameterize using OpenFF toolkit (GAFF + AM1-BCC)."""
        # Load molecule
        ext = Path(lig_file).suffix.lower()
        if ext == ".sdf":
            molecules = list(Molecule.from_file(lig_file, file_format="sdf"))
        elif ext == ".mol2":
            molecules = list(Molecule.from_file(lig_file, file_format="mol2"))
        elif ext in (".pdb",):
            molecules = list(Molecule.from_file(lig_file, file_format="pdb"))
        elif ext in (".mol",):
            molecules = list(Molecule.from_file(lig_file, file_format="mol"))
        else:
            raise ValueError(f"Unsupported ligand format: {ext}")
        
        if not molecules:
            raise ValueError(f"No molecules found in {lig_file}")
        
        # Take first molecule (or all if multiple poses in one file)
        mol = molecules[0]
        mol.name = name
        
        # Assign charges
        self.logger.debug("Assigning %s charges...", self.config.charge_method)
        if self.config.charge_method.lower() in ("am1bcc", "am1bccelf10"):
            if _HAVE_OPENFF_RECHARGE:
                # Use OpenFF Recharge for AM1-BCC
                recharge = Recharge()
                mol = recharge.assign_charges(mol, method=self.config.charge_method)
            else:
                # Fallback: use OpenFF's built-in AM1-BCC (requires openeye)
                try:
                    mol.assign_partial_charges("am1bcc")
                except Exception as e:
                    self.logger.warning("AM1-BCC failed (%s), trying GASTEIGER", e)
                    mol.assign_partial_charges("gasteiger")
        elif self.config.charge_method.lower() == "gasteiger":
            mol.assign_partial_charges("gasteiger")
        elif self.config.charge_method.lower() == "mmff94":
            mol.assign_partial_charges("mmff94")
        else:
            self.logger.warning("Unknown charge method %s, using gasteiger", self.config.charge_method)
            mol.assign_partial_charges("gasteiger")
        
        # Create OpenFF Topology and apply GAFF force field
        topology = Topology.from_molecules([mol])
        
        # Load GAFF force field
        try:
            ff = ForceField(self.config.ligand_ff)
        except Exception:
            # Try common variants
            for ff_name in ["gaff-2.11", "gaff-1.8", "gaff"]:
                try:
                    ff = ForceField(ff_name)
                    self.logger.info("Using force field: %s", ff_name)
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError(f"Could not load GAFF force field: {self.config.ligand_ff}")
        
        # Parametrize
        openff_system = ff.create_openff_system(topology)
        
        # Convert to ParmEd Structure
        # OpenFF Interchange -> ParmEd
        from openff.interchange import Interchange
        interchange = Interchange.from_smirnoff(ff, topology)
        structure = interchange.to_parmed()
        
        # Set residue name
        for res in structure.residues:
            res.name = self.config.ligand_resname
        
        # Save parameterized ligand for reference
        lig_out = os.path.join(self.work_dir, f"{name}.mol2")
        structure.save(lig_out, overwrite=True)
        self.logger.debug("Parameterized ligand saved to: %s", lig_out)
        
        return structure, self.config.ligand_resname
    
    def _parameterize_with_rdkit_antechamber(self, lig_file: str, name: str) -> Tuple[pmd.Structure, str]:
        """Parameterize using RDKit + antechamber (AmberTools)."""
        # Convert to MOL2 if needed
        mol2_path = lig_file
        if not lig_file.lower().endswith(".mol2"):
            mol2_path = os.path.join(self.work_dir, f"{name}.mol2")
            self._convert_to_mol2(lig_file, mol2_path)
        
        # Run antechamber for GAFF2 + AM1-BCC
        ac_out = os.path.join(self.work_dir, f"{name}_ac.mol2")
        ac_log = os.path.join(self.work_dir, f"{name}_ac.log")
        
        # antechamber -i input.mol2 -fi mol2 -o output.mol2 -fo mol2 -c bcc -s 2 -nc 0 -at gaff2
        charge_flag = "bcc" if "am1bcc" in self.config.charge_method.lower() else "gas"
        cmd = [
            "antechamber",
            "-i", mol2_path, "-fi", "mol2",
            "-o", ac_out, "-fo", "mol2",
            "-c", charge_flag, "-s", "2", "-nc", "0",
            "-at", "gaff2",
            "-pf", "y"  # generate prep file
        ]
        
        self.logger.debug("Running antechamber: %s", " ".join(cmd))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"antechamber failed: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise RuntimeError(f"antechamber not available or failed: {e}")
        
        # Load parameterized ligand
        structure = pmd.load_file(ac_out)
        for res in structure.residues:
            res.name = self.config.ligand_resname
        
        return structure, self.config.ligand_resname
    
    def _convert_to_mol2(self, input_file: str, output_file: str):
        """Convert ligand file to MOL2 using RDKit."""
        ext = Path(input_file).suffix.lower()
        if ext == ".sdf":
            mols = Chem.SDMolSupplier(input_file, removeHs=False)
            mol = next((m for m in mols if m is not None), None)
        elif ext == ".pdb":
            mol = Chem.MolFromPDBFile(input_file, removeHs=False)
        elif ext == ".mol":
            mol = Chem.MolFromMolFile(input_file, removeHs=False)
        else:
            raise ValueError(f"Cannot convert {ext} to mol2")
        
        if mol is None:
            raise ValueError(f"Failed to read molecule from {input_file}")
        
        # Add hydrogens if needed
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        writer = Chem.SDWriter(output_file.replace(".mol2", ".sdf"))
        writer.write(mol)
        writer.close()
        
        # Convert SDF to MOL2 via obabel if available, else use the SDF
        try:
            subprocess.run(["obabel", "-isdf", output_file.replace(".mol2", ".sdf"), 
                          "-omol2", "-O", output_file], check=True, capture_output=True)
        except Exception:
            # Just rename
            shutil.move(output_file.replace(".mol2", ".sdf"), output_file)


# ============================================================================
# Complex Assembly
# ============================================================================
class ComplexAssembler:
    """Assemble protein + ligand into a single complex topology for MM-GBSA."""
    
    def __init__(self, config: DockingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.work_dir = config.work_dir or tempfile.mkdtemp(prefix="mmgbsa_docking_")
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
    
    def assemble(self, protein_struct: pmd.Structure, 
                 ligand_structs: List[Tuple[str, pmd.Structure, str]]) -> List[Tuple[str, pmd.Structure]]:
        """
        Assemble protein with each ligand into a complex.
        Returns list of (ligand_name, complex_structure).
        """
        self.logger.info("Assembling %d protein-ligand complexes...", len(ligand_structs))
        
        complexes = []
        for lig_name, lig_struct, lig_resname in ligand_structs:
            try:
                complex_struct = self._merge_protein_ligand(protein_struct, lig_struct, lig_resname)
                complexes.append((lig_name, complex_struct))
            except Exception as e:
                self.logger.error("Failed to assemble complex for %s: %s", lig_name, e)
                if self.config.verbose:
                    import traceback
                    traceback.print_exc()
        
        return complexes
    
    def _merge_protein_ligand(self, protein: pmd.Structure, 
                              ligand: pmd.Structure, lig_resname: str) -> pmd.Structure:
        """Merge protein and ligand structures into a single complex."""
        # Create a copy of protein to avoid modifying original
        complex_struct = protein.copy()
        
        # Get ligand atoms (assuming single residue)
        lig_residues = list(ligand.residues)
        if not lig_residues:
            raise ValueError("Ligand structure has no residues")
        lig_res = lig_residues[0]
        
        # Add ligand residue to complex
        # ParmEd doesn't have a direct "add residue" method, so we combine at atom level
        # Better approach: use ParmEd's combine or manually add atoms
        
        # Simple approach: create new structure from combined atoms
        # This preserves all parameters
        combined_atoms = list(complex_struct.atoms) + list(ligand.atoms)
        combined_bonds = list(complex_struct.bonds) + list(ligand.bonds)
        combined_angles = list(complex_struct.angles) + list(ligand.angles)
        combined_dihedrals = list(complex_struct.dihedrals) + list(ligand.dihedrals)
        
        # Create new structure
        from parmed import Structure
        new_struct = Structure()
        
        # Add atoms
        atom_map = {}
        for i, atom in enumerate(combined_atoms):
            new_atom = new_struct.add_atom(
                name=atom.name,
                type=atom.type,
                charge=atom.charge,
                mass=atom.mass,
                atomic_number=atom.atomic_number,
                residue=atom.residue  # This might need adjustment
            )
            atom_map[atom] = new_atom
        
        # Need to properly handle residues - recreate them
        # Better: use ParmEd's join method
        try:
            # ParmEd's join combines structures
            combined = complex_struct.join(ligand)
            # Ensure ligand residue name is set
            for res in combined.residues:
                if res.name != lig_resname and any(a.name in [la.name for la in lig_res.atoms] for a in res.atoms):
                    res.name = lig_resname
            return combined
        except Exception:
            # Fallback: manual combination
            pass
        
        # Manual combination fallback
        return self._manual_combine(complex_struct, ligand, lig_resname)
    
    def _manual_combine(self, protein: pmd.Structure, ligand: pmd.Structure, 
                        lig_resname: str) -> pmd.Structure:
        """Manually combine protein and ligand structures."""
        from parmed import Structure, Atom, Residue, Bond, Angle, Dihedral
        
        new_struct = Structure()
        
        # Track residue mapping
        res_map = {}
        
        # Add protein residues
        for res in protein.residues:
            new_res = Residue(res.name, res.number, res.chain)
            new_struct.add_residue(new_res, res.chain)
            res_map[res] = new_res
            for atom in res.atoms:
                new_atom = Atom(
                    name=atom.name,
                    type=atom.type,
                    charge=atom.charge,
                    mass=atom.mass,
                    atomic_number=atom.atomic_number
                )
                new_res.add_atom(new_atom)
        
        # Add ligand as new residue (use next residue number)
        next_resnum = max(r.number for r in protein.residues) + 1 if protein.residues else 1
        lig_res = list(ligand.residues)[0]
        new_lig_res = Residue(lig_resname, next_resnum, "L")
        new_struct.add_residue(new_lig_res, "L")
        
        for atom in lig_res.atoms:
            new_atom = Atom(
                name=atom.name,
                type=atom.type,
                charge=atom.charge,
                mass=atom.mass,
                atomic_number=atom.atomic_number
            )
            new_lig_res.add_atom(new_atom)
        
        # Copy bonds (adjust indices)
        # This is complex - use ParmEd's built-in if possible
        # For now, return protein+ligand joined
        return protein.join(ligand)


# ============================================================================
# MM-GBSA Runner (wraps mmgbsa_amb.py engine)
# ============================================================================
class DockingMMGBSARunner:
    """Run MM-GBSA on assembled complexes using the validated engine."""
    
    def __init__(self, config: DockingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def run_on_complexes(self, complexes: List[Tuple[str, pmd.Structure]]) -> List[Dict[str, Any]]:
        """
        Run MM-GBSA on each complex.
        Returns list of result dictionaries with binding energies.
        """
        results = []
        
        for i, (lig_name, complex_struct) in enumerate(complexes):
            self.logger.info("Running MM-GBSA on complex %d/%d: %s", 
                           i+1, len(complexes), lig_name)
            
            try:
                result = self._run_single_complex(lig_name, complex_struct)
                results.append(result)
            except Exception as e:
                self.logger.error("MM-GBSA failed for %s: %s", lig_name, e)
                if self.config.verbose:
                    import traceback
                    traceback.print_exc()
                results.append({
                    "ligand": lig_name,
                    "error": str(e),
                    "delta_total": float('nan'),
                    "delta_vdw": float('nan'),
                    "delta_elec": float('nan'),
                    "delta_gb": float('nan'),
                    "delta_sasa": float('nan'),
                })
        
        return results
    
    def _run_single_complex(self, lig_name: str, complex_struct: pmd.Structure) -> Dict[str, Any]:
        """Run MM-GBSA on a single complex using v4 engine."""
        
        # Create temporary topology and coordinate files
        with tempfile.TemporaryDirectory(dir=self.config.work_dir) as tmpdir:
            prmtop = os.path.join(tmpdir, "complex.prmtop")
            inpcrd = os.path.join(tmpdir, "complex.inpcrd")
            
            # Save complex topology
            complex_struct.save(prmtop, overwrite=True)
            complex_struct.save(inpcrd, overwrite=True)
            
            # Build v4 config
            v4_config = v4.Config(
                top_file=prmtop,
                coord_files=[inpcrd],  # single frame
                ligand_resnames=[self.config.ligand_resname],
                strip_resnames=set(v4.DEFAULT_STRIP_RESNAMES),
                no_strip=False,
                igb=self.config.igb,
                eps_solvent=self.config.eps_solvent,
                radii_set=self.config.radii_set,
                gb_radii_cutoff=self.config.gb_cutoff if self.config.gb_cutoff > 0 else None,
                probe_radius=self.config.probe_radius,
                sasa_radii_set="bondi",
                surface_tension=self.config.surface_tension,
                sasa_points=self.config.sasa_points,
                start_frame=1,
                end_frame=1,
                stride=1,
                output_csv=os.path.join(tmpdir, "results.csv"),
                decomp=self.config.decomp_csv is not None,
                decomp_csv=os.path.join(tmpdir, "decomp.csv") if self.config.decomp_csv else None,
                json_output=None,
                verbose=self.config.verbose,
            )
            
            # Run v4 engine
            logger = v4.setup_logging(self.config.verbose)
            top_model = v4.TopologyModel(complex_struct, v4_config, logger)
            engine = v4.MMGBSAEngine(top_model, v4_config, logger)
            
            # Process single frame
            coords, box, _ = next(v4.iter_frames([inpcrd], top_model.n_atoms, prmtop))
            data = engine.compute_frame(coords)
            
            # Convert to format expected by v4.write_frame_csv
            # The v4 engine returns data with keys like:
            # - delta_vdw, delta_elec, delta_gb, delta_sasa, delta_total
            # - vdw_complex, elec_complex, gb_complex, sasa_complex, vdw14_complex, elec14_complex
            # - vdw_receptor, elec_receptor, gb_receptor, sasa_receptor
            # - vdw_ligand, elec_ligand, gb_ligand, sasa_ligand
            
            result = {
                "ligand": lig_name,
            }
            
            # Add all delta keys
            for key in v4.DELTA_KEYS:
                result[f"delta_{key}"] = float(data[f"delta_{key}"])
            
            # Add all absolute keys
            for key in v4.ABS_KEYS:
                result[key] = float(data[key])
            
            # Decomposition if requested
            if self.config.decomp_csv:
                decomp = engine.decompose(data)
                result["decomposition"] = decomp
            
            return result


# ============================================================================
# Output Writers
# ============================================================================
def write_results_csv(results: List[Dict[str, Any]], output_path: str):
    """Write ranked MM-GBSA results to CSV using v4's write_frame_csv format."""
    # Sort by delta_total (most negative = best binding)
    sorted_results = sorted(results, key=lambda x: x.get("delta_total", float('inf')))
    
    # Filter out failed results for the main CSV
    valid_results = [r for r in sorted_results if "error" not in r]
    
    if not valid_results:
        # Write empty CSV with headers
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "ligand"] + 
                          [f"delta_{k}" for k in v4.DELTA_KEYS] + 
                          [k for k in v4.ABS_KEYS])
        return
    
    # Use v4's write_frame_csv for consistent format
    # But we need to add ligand name to each row
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        headers = ["rank", "ligand"] + [f"delta_{k}" for k in v4.DELTA_KEYS] + v4.ABS_KEYS
        writer.writerow(headers)
        
        for rank, r in enumerate(sorted_results, 1):
            row = [rank, r.get("ligand", "")]
            for k in v4.DELTA_KEYS:
                row.append(r.get(f"delta_{k}", ""))
            for k in v4.ABS_KEYS:
                row.append(r.get(k, ""))
            writer.writerow(row)
        
        # Average row for valid results
        if len(valid_results) > 1:
            avg_row = ["AVERAGE", ""]
            for k in v4.DELTA_KEYS:
                vals = [r.get(f"delta_{k}", float('nan')) for r in valid_results]
                avg_row.append(float(np.nanmean(vals)))
            for k in v4.ABS_KEYS:
                vals = [r.get(k, float('nan')) for r in valid_results]
                avg_row.append(float(np.nanmean(vals)))
            writer.writerow(avg_row)
            
            std_row = ["STDDEV", ""]
            for k in v4.DELTA_KEYS:
                vals = [r.get(f"delta_{k}", float('nan')) for r in valid_results]
                std_row.append(float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0)
            for k in v4.ABS_KEYS:
                vals = [r.get(k, float('nan')) for r in valid_results]
                std_row.append(float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0)
            writer.writerow(std_row)


def write_decomposition_csv(results: List[Dict[str, Any]], output_path: str, 
                            ligand_resname: str = "LIG"):
    """Write per-residue decomposition for all ligands (merged)."""
    sorted_results = sorted(results, key=lambda x: x.get("delta_total", float('inf')))
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ligand", "rank", "residue_index", "resname", "resid",
                        "vdw", "elec", "gb", "sasa", "total"])
        
        for rank, r in enumerate(sorted_results, 1):
            lig_name = r.get("ligand", "")
            decomp = r.get("decomposition")
            if decomp is None:
                continue
            
            # v4 decompose returns: res_index, vdw, elec, gb, sasa, ligand_vdw, ligand_elec, ligand_gb, ligand_sasa
            n_res = len(decomp.get("vdw", []))
            res_indices = decomp.get("res_index", list(range(n_res)))
            
            for k in range(n_res):
                ri = res_indices[k] if k < len(res_indices) else k
                total = (decomp["vdw"][k] + decomp["elec"][k] + 
                        decomp["gb"][k] + decomp["sasa"][k])
                # We don't have resnames/resids in the decomp dict, use index
                writer.writerow([lig_name, rank, int(ri), f"RES{ri}", ri,
                               decomp["vdw"][k], decomp["elec"][k],
                               decomp["gb"][k], decomp["sasa"][k], total])
            
            # Add LIGAND row (v4 puts this as ligand_vdw etc. with single element arrays)
            if "ligand_vdw" in decomp:
                lig_vdw = float(decomp["ligand_vdw"][0]) if len(decomp["ligand_vdw"]) > 0 else 0.0
                lig_elec = float(decomp["ligand_elec"][0]) if len(decomp["ligand_elec"]) > 0 else 0.0
                lig_gb = float(decomp["ligand_gb"][0]) if len(decomp["ligand_gb"]) > 0 else 0.0
                lig_sasa = float(decomp["ligand_sasa"][0]) if len(decomp["ligand_sasa"]) > 0 else 0.0
                writer.writerow([lig_name, rank, "LIGAND", ligand_resname, "",
                               lig_vdw, lig_elec, lig_gb, lig_sasa,
                               lig_vdw + lig_elec + lig_gb + lig_sasa])


def write_json_summary(results: List[Dict[str, Any]], config: DockingConfig, 
                       output_path: str, protein_pdb: str):
    """Write machine-readable JSON summary."""
    sorted_results = sorted(results, key=lambda x: x.get("delta_total", float('inf')))
    
    summary = {
        "metadata": {
            "protein_pdb": protein_pdb,
            "n_ligands": len(results),
            "protein_ff": config.protein_ff,
            "ligand_ff": config.ligand_ff,
            "charge_method": config.charge_method,
            "ph": config.ph,
            "igb": config.igb,
            "eps_solvent": config.eps_solvent,
            "gb_cutoff": config.gb_cutoff,
            "radii_set": config.radii_set,
            "surface_tension": config.surface_tension,
            "probe_radius": config.probe_radius,
        },
        "results": sorted_results,
        "statistics": {
            "best_ligand": sorted_results[0]["ligand"] if sorted_results else None,
            "best_delta_total": sorted_results[0]["delta_total"] if sorted_results else None,
            "mean_delta_total": float(np.nanmean([r["delta_total"] for r in results])),
            "std_delta_total": float(np.nanstd([r["delta_total"] for r in results])),
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)


# ============================================================================
# Main Pipeline
# ============================================================================
def run_docking_rescoring(config: DockingConfig) -> List[Dict[str, Any]]:
    """Run the complete docking rescoring pipeline."""
    logger = setup_logging(config.verbose)
    
    # Set up work directory
    if config.work_dir is None:
        config.work_dir = tempfile.mkdtemp(prefix="mmgbsa_docking_")
    work_dir = Path(config.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Working directory: %s", work_dir)
    
    try:
        # Step 1: Prepare protein
        logger.info("=" * 60)
        logger.info("STEP 1: Protein Preparation")
        logger.info("=" * 60)
        protein_prep = ProteinPreparator(config, logger)
        protein_struct, protein_pdb = protein_prep.prepare(config.protein_pdb)
        logger.info("Protein prepared: %d atoms, %d residues", 
                   len(protein_struct.atoms), len(protein_struct.residues))
        
        # Step 2: Parameterize ligands
        logger.info("=" * 60)
        logger.info("STEP 2: Ligand Parameterization")
        logger.info("=" * 60)
        lig_param = LigandParameterizer(config, logger)
        ligand_structs = lig_param.parameterize_ligands(config.ligand_files)
        logger.info("Ligands parameterized: %d", len(ligand_structs))
        
        if not ligand_structs:
            raise RuntimeError("No ligands successfully parameterized")
        
        # Step 3: Assemble complexes
        logger.info("=" * 60)
        logger.info("STEP 3: Complex Assembly")
        logger.info("=" * 60)
        assembler = ComplexAssembler(config, logger)
        complexes = assembler.assemble(protein_struct, ligand_structs)
        logger.info("Complexes assembled: %d", len(complexes))
        
        if not complexes:
            raise RuntimeError("No complexes successfully assembled")
        
        # Step 4: Run MM-GBSA
        logger.info("=" * 60)
        logger.info("STEP 4: MM-GBSA Calculation")
        logger.info("=" * 60)
        runner = DockingMMGBSARunner(config, logger)
        results = runner.run_on_complexes(complexes)
        
        # Step 5: Output results
        logger.info("=" * 60)
        logger.info("STEP 5: Writing Results")
        logger.info("=" * 60)
        
        write_results_csv(results, config.output_csv)
        logger.info("Results written to: %s", config.output_csv)
        
        if config.decomp_csv:
            write_decomposition_csv(results, config.decomp_csv, config.ligand_resname)
            logger.info("Decomposition written to: %s", config.decomp_csv)
        
        if config.json_output:
            write_json_summary(results, config, config.json_output, config.protein_pdb)
            logger.info("JSON summary written to: %s", config.json_output)
        
        # Print summary
        valid_results = [r for r in results if "error" not in r]
        if valid_results:
            best = min(valid_results, key=lambda x: x["delta_total"])
            logger.info("=" * 60)
            logger.info("SUMMARY")
            logger.info("=" * 60)
            logger.info("Best ligand: %s (ΔG = %.3f kcal/mol)", best["ligand"], best["delta_total"])
            logger.info("Processed: %d/%d ligands successful", len(valid_results), len(results))
        
        return results
        
    finally:
        # Cleanup
        if not config.keep_intermediate and config.work_dir and not config.work_dir.startswith("/tmp"):
            try:
                shutil.rmtree(config.work_dir)
                logger.debug("Cleaned up work directory: %s", config.work_dir)
            except Exception:
                pass


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> DockingConfig:
    p = argparse.ArgumentParser(
        description="MM-GBSA Docking Rescoring Engine: physics-based rescoring for docked protein-ligand complexes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Input
    p.add_argument("-p", "--protein", required=True, help="Protein PDB file")
    p.add_argument("-l", "--ligands", nargs="+", required=True, 
                   help="Ligand file(s): SDF, MOL2, PDB, or directory")
    p.add_argument("--ligand-resname", default="LIG", help="Ligand residue name")
    p.add_argument("--prepared-ligand", action="store_true",
                   help="Ligand files already parameterized (MOL2 with charges/types)")
    p.add_argument("--max-ligands", type=int, default=None, help="Limit number of ligands (for testing)")
    p.add_argument("--ligand-name-prefix", default="LIG", help="Prefix for ligand names")
    
    # Force fields
    p.add_argument("--protein-ff", default=DEFAULT_PROTEIN_FF,
                   choices=["ff14SB", "ff19SB", "ff03", "ff99SB", "ff99SBildn"],
                   help="Protein force field")
    p.add_argument("--ligand-ff", default=DEFAULT_LIGAND_FF,
                   help="Ligand force field (GAFF variant)")
    p.add_argument("--charge-method", default=DEFAULT_CHARGE_METHOD,
                   choices=["am1bcc", "am1bccelf10", "gasteiger", "mmff94"],
                   help="Partial charge method for ligands")
    
    # Protein preparation
    p.add_argument("--ph", type=float, default=DEFAULT_PH, help="pH for protonation")
    p.add_argument("--no-fix-protein", action="store_true", help="Skip pdbfixer structure fixing")
    p.add_argument("--no-protonate", action="store_true", help="Skip PROPKA protonation")
    p.add_argument("--no-add-hydrogens", action="store_true", help="Don't add hydrogens")
    
    # MM-GBSA
    p.add_argument("--igb", type=int, choices=[2, 5], default=DEFAULT_IGB, help="GB model")
    p.add_argument("--eps-solvent", type=float, default=DEFAULT_EPS_SOLVENT)
    p.add_argument("--gb-cutoff", type=float, default=DEFAULT_GB_CUTOFF, help="GB cutoff (0=disable)")
    p.add_argument("--radii", choices=["auto", "mbondi", "mbondi2", "mbondi3", "bondi"], default="auto")
    p.add_argument("--probe", type=float, default=DEFAULT_PROBE)
    p.add_argument("--surface-tension", type=float, default=DEFAULT_SURFACE_TENSION)
    p.add_argument("--sasa-points", type=int, default=DEFAULT_SASA_POINTS)
    
    # Output
    p.add_argument("-o", "--output", default="docking_mmgbsa_results.csv", help="Output CSV")
    p.add_argument("--decomp", default=None, help="Per-residue decomposition CSV")
    p.add_argument("--json", default=None, help="JSON summary output")
    p.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate files")
    p.add_argument("--work-dir", default=None, help="Working directory (default: temp)")
    
    # Processing
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--n-procs", type=int, default=1, help="Number of processes (future)")
    
    args = p.parse_args()
    
    return DockingConfig(
        protein_pdb=args.protein,
        ligand_files=args.ligands,
        ligand_resname=args.ligand_resname,
        protein_ff=args.protein_ff,
        ligand_ff=args.ligand_ff,
        charge_method=args.charge_method,
        ph=args.ph,
        add_hydrogens=not args.no_add_hydrogens,
        fix_protein=not args.no_fix_protein,
        protonate=not args.no_protonate,
        prepared_ligand=args.prepared_ligand,
        max_ligands=args.max_ligands,
        ligand_name_prefix=args.ligand_name_prefix,
        igb=args.igb,
        eps_solvent=args.eps_solvent,
        gb_cutoff=args.gb_cutoff,
        radii_set=args.radii,
        probe_radius=args.probe,
        surface_tension=args.surface_tension,
        sasa_points=args.sasa_points,
        output_csv=args.output,
        decomp_csv=args.decomp,
        json_output=args.json,
        keep_intermediate=args.keep_intermediate,
        work_dir=args.work_dir,
        verbose=args.verbose,
        n_procs=args.n_procs,
    )


def main():
    config = parse_args()
    
    # Check dependencies
    if not _HAVE_OPENFF and not _HAVE_RDKIT:
        print("ERROR: Ligand parameterization requires either:")
        print("  - openff-toolkit (recommended): pip install openff-toolkit openff-recharge")
        print("  - rdkit + AmberTools (antechamber): conda install -c conda-forge rdkit ambertools")
        sys.exit(1)
    
    if not _HAVE_PDBFIXER:
        print("WARNING: pdbfixer not available; protein structure fixing disabled")
        print("  Install: conda install -c conda-forge pdbfixer")
    
    if not _HAVE_PROPKA:
        print("WARNING: propka3 not available; protonation state assignment disabled")
        print("  Install: pip install propka3")
    
    try:
        run_docking_rescoring(config)
        return 0
    except Exception as e:
        logging.error("Pipeline failed: %s", e)
        if config.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    import glob
    sys.exit(main())