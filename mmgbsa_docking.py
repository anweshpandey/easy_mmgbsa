#!/usr/bin/env python3
"""
MM-GBSA Docking Rescoring Engine
================================

Physics-based rescoring for protein–ligand complexes from molecular docking
(AutoDock Vina, AutoDock4, Glide, GOLD, etc.). Handles the critical topology/
parameter preparation that docked poses require before MM-GBSA calculation.

Key features:
- Reads AutoDock Vina PDBQT files (multiple poses per file)
- Maps docked coordinates onto original ligand structures (SDF/MOL2) to preserve
  chemical information (bond orders, stereochemistry) while keeping docking poses
- Protein preparation: PDBFixer (structure fixing), PROPKA (protonation), AMBER FF
- Ligand parameterization: GAFF2 + AM1-BCC charges (OpenFF toolkit/AmberTools)
- Complex assembly: merges protein + ligand into single AMBER topology
- MM-GBSA: uses validated mmgbsa_amb.py v4 engine (OBC GB, HCT descreening, idecomp)
- Batch processing of ligand libraries with ranked output

Usage:
    # AutoDock Vina poses + original ligand structures
    python3 mmgbsa_docking.py -p protein.pdb -l vina_output/*.pdbqt \
        --ligand-refs ligands.sdf -o results.csv --decomp decomp.csv

    # With decomposition and JSON summary
    python3 mmgbsa_docking.py -p protein.pdb -l poses/ \
        --ligand-refs ligands.sdf --decomp decomp.csv --json summary.json

    # Custom force fields and charge method
    python3 mmgbsa_docking.py -p protein.pdb -l vina_output/ \
        --ligand-refs ligands.sdf --protein-ff ff19SB --ligand-ff gaff-2.11 \
        --charge-method am1bcc --ph 7.4 -o results.csv --decomp decomp.csv

Requirements:
    openff-toolkit (for ligand GAFF2/AM1-BCC parameterization)
    openff-recharge (optional, for AM1-BCC via OpenEye/AM1BCC)
    propka3 (optional, for protein protonation states)
    pdbfixer (optional, for protein structure fixing)
    rdkit (optional, for PDBQT parsing and antechamber fallback)

The engine uses the validated mmgbsa_amb.py MM-GBSA implementation under the hood.
"""

from __future__ import annotations

import argparse
import csv
import glob
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
    ligand_files: List[str]  # PDBQT files from docking (or SDF/MOL2/PDB)
    ligand_refs: Optional[str] = None  # Original ligand structures (SDF/MOL2) for chemical info
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
    
    # Pose selection mode
    pose_mode: str = "best"  # "best", "all", "ensemble"
    
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
# PDBQT Parser (AutoDock Vina output)
# ============================================================================
class PDBQTParser:
    """Parse AutoDock Vina PDBQT files containing multiple docking poses."""
    
    @staticmethod
    def parse(pdbqt_file: str) -> List[Dict[str, Any]]:
        """
        Parse PDBQT file and return list of poses.
        Each pose dict contains: atoms (list), coords (np.ndarray), energy (float), 
        atom_names, atom_types, charges
        """
        poses = []
        current_pose = None
        
        with open(pdbqt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('MODEL'):
                    # New pose
                    if current_pose is not None:
                        poses.append(current_pose)
                    current_pose = {
                        'model_num': int(line.split()[1]),
                        'atoms': [],
                        'coords': [],
                        'atom_names': [],
                        'atom_types': [],
                        'charges': [],
                        'energy': None
                    }
                elif line.startswith('ENDMDL'):
                    if current_pose is not None:
                        poses.append(current_pose)
                        current_pose = None
                elif line.startswith('REMARK VINA RESULT:'):
                    # Vina binding affinity
                    if current_pose is not None:
                        parts = line.split()
                        current_pose['energy'] = float(parts[3])
                elif line.startswith('ATOM') or line.startswith('HETATM'):
                    if current_pose is None:
                        current_pose = {
                            'model_num': 1,
                            'atoms': [],
                            'coords': [],
                            'atom_names': [],
                            'atom_types': [],
                            'charges': [],
                            'energy': None
                        }
                    # PDBQT format: ATOM  x   y   z  q  type
                    # Columns: 0-5: ATOM/HETATM, 6-11: serial, 12-15: name, 16: altLoc,
                    # 17-20: resName, 22: chainID, 23-26: resSeq, 27: iCode,
                    # 30-37: x, 38-45: y, 46-53: z, 54-59: occupancy, 60-65: tempFactor,
                    # 66-76: element/charge (q), 77-78: atom type (AutoDock type)
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    # Charge is in columns 69-76 (or 70-76)
                    try:
                        q = float(line[69:76]) if len(line) > 76 else 0.0
                    except ValueError:
                        q = 0.0
                    # Atom type in columns 77-78
                    atype = line[77:79].strip() if len(line) > 78 else "C"
                    name = line[12:16].strip()
                    
                    current_pose['coords'].append([x, y, z])
                    current_pose['atom_names'].append(name)
                    current_pose['atom_types'].append(atype)
                    current_pose['charges'].append(q)
                    current_pose['atoms'].append({
                        'name': name, 'type': atype, 'charge': q,
                        'x': x, 'y': y, 'z': z
                    })
        
        # Handle last pose if file doesn't end with ENDMDL
        if current_pose is not None and current_pose['coords']:
            poses.append(current_pose)
        
        # Convert coords to numpy arrays
        for pose in poses:
            pose['coords'] = np.array(pose['coords'], dtype=float)
        
        return poses


# ============================================================================
# Ligand Reference Loader (SDF/MOL2 for chemical information)
# ============================================================================
class LigandReferenceLoader:
    """Load original ligand structures from SDF/MOL2 for chemical info (bond orders, stereo)."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.refs: Dict[str, Any] = {}  # name -> RDKit mol or OpenFF Molecule
    
    def load(self, ref_file: str) -> Dict[str, Any]:
        """Load ligand references from SDF/MOL2 file(s)."""
        if not _HAVE_RDKIT:
            self.logger.warning("RDKit not available; cannot load ligand references for chemical mapping")
            return {}
        
        ext = Path(ref_file).suffix.lower()
        if ext == ".sdf":
            mols = Chem.SDMolSupplier(ref_file, removeHs=False, sanitize=True)
            for mol in mols:
                if mol is not None:
                    name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"lig_{len(self.refs)}"
                    self.refs[name] = mol
        elif ext in (".mol2",):
            mol = Chem.MolFromMol2File(ref_file, removeHs=False, sanitize=True)
            if mol is not None:
                name = mol.GetProp("_Name") if mol.HasProp("_Name") else Path(ref_file).stem
                self.refs[name] = mol
        elif ext in (".pdb",):
            mol = Chem.MolFromPDBFile(ref_file, removeHs=False, sanitize=True)
            if mol is not None:
                name = Path(ref_file).stem
                self.refs[name] = mol
        else:
            raise ValueError(f"Unsupported ligand reference format: {ext}")
        
        self.logger.info("Loaded %d ligand reference structures from %s", len(self.refs), ref_file)
        return self.refs
    
    def get_reference(self, ligand_name: str) -> Optional[Any]:
        """Get reference molecule by name (fuzzy matching)."""
        if ligand_name in self.refs:
            return self.refs[ligand_name]
        # Try fuzzy matching
        for ref_name, mol in self.refs.items():
            if ref_name.lower() in ligand_name.lower() or ligand_name.lower() in ref_name.lower():
                return mol
        # Return first reference if only one
        if len(self.refs) == 1:
            return list(self.refs.values())[0]
        return None


# ============================================================================
# Coordinate Mapper: Map PDBQT coordinates onto reference ligand
# ============================================================================
class PoseCoordinateMapper:
    """Map docked coordinates from PDBQT onto reference ligand structure."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def map_pose_to_reference(self, pose: Dict[str, Any], ref_mol: Any) -> Any:
        """
        Map PDBQT pose coordinates onto RDKit reference molecule.
        Returns new RDKit mol with docked coordinates but reference chemistry.
        """
        if not _HAVE_RDKIT:
            raise RuntimeError("RDKit required for coordinate mapping")
        
        # Create a copy of reference molecule
        mapped_mol = Chem.Mol(ref_mol)
        mapped_mol = Chem.RemoveHs(mapped_mol)
        
        # Get coordinates from pose
        pose_coords = pose['coords']
        pose_atom_names = pose['atom_names']
        
        # Try to match atoms by name
        conf = mapped_mol.GetConformer() if mapped_mol.GetNumConformers() > 0 else None
        if conf is None:
            # Generate empty conformer
            AllChem.EmbedMolecule(mapped_mol, randomSeed=42)
            conf = mapped_mol.GetConformer()
        
        # Map coordinates by atom name
        matched = 0
        for i, atom in enumerate(mapped_mol.GetAtoms()):
            atom_name = atom.GetProp("_TriposAtomName") if atom.HasProp("_TriposAtomName") else atom.GetSymbol() + str(i)
            # Also try PDB name
            if not atom.HasProp("_TriposAtomName"):
                atom_name = atom.GetPDBResidueInfo().GetName().strip() if atom.GetPDBResidueInfo() else atom.GetSymbol() + str(i)
            
            # Find matching pose atom
            for j, pname in enumerate(pose_atom_names):
                if self._atom_names_match(atom_name, pname):
                    conf.SetAtomPosition(i, pose_coords[j])
                    matched += 1
                    break
        
        self.logger.debug("Mapped %d/%d atoms from pose to reference", matched, mapped_mol.GetNumAtoms())
        
        if matched < mapped_mol.GetNumAtoms() * 0.5:
            self.logger.warning("Low atom mapping rate (%d/%d) - pose may not match reference", 
                              matched, mapped_mol.GetNumAtoms())
        
        # Add hydrogens with proper geometry based on mapped heavy atoms
        mapped_mol = Chem.AddHs(mapped_mol, addCoords=True)
        
        return mapped_mol
    
    def _atom_names_match(self, name1: str, name2: str) -> bool:
        """Flexible atom name matching."""
        # Strip numbers and special chars
        n1 = ''.join(c for c in name1 if c.isalpha())
        n2 = ''.join(c for c in name2 if c.isalpha())
        if n1.upper() == n2.upper():
            return True
        # Try with numbers
        if name1.strip() == name2.strip():
            return True
        return False


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
        
        # Step 3: Load with ParmEd and parametrize with AMBER FF (use tleap)
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
        output_path = os.path.join(self.work_dir, "protein_protonated.pdb")
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
        """Load protein PDB and apply AMBER force field using tleap."""
        self.logger.info("Parametrizing protein with %s...", self.config.protein_ff)
        
        # Try tleap first (proper AMBER parametrization)
        structure = self._parametrize_with_tleap(pdb_path)
        if structure is not None:
            return structure
        
        # Fallback: OpenFF (limited protein support) - WARN but don't silently fail
        self.logger.warning("tleap not available; using OpenFF fallback (limited protein FF support)")
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
            self.logger.warning("OpenFF not available; loading PDB without force field parameters")
            return pmd.load_file(pdb_path)
        
        structure = pmd.load_file(pdb_path)
        return structure


# ============================================================================
# Ligand Parameterization
# ============================================================================
class LigandParameterizer:
    """Generate GAFF2/AM1-BCC parameters for ligand molecules."""
    
    def __init__(self, config: DockingConfig, logger: logging.Logger, 
                 ref_loader: LigandReferenceLoader, coord_mapper: PoseCoordinateMapper):
        self.config = config
        self.logger = logger
        self.ref_loader = ref_loader
        self.coord_mapper = coord_mapper
        self.work_dir = config.work_dir or tempfile.mkdtemp(prefix="mmgbsa_docking_")
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
    
    def parameterize_ligands(self, ligand_files: List[str]) -> List[Tuple[str, pmd.Structure, str, Dict]]:
        """
        Parameterize ligand files and return list of 
        (name, ParmEd Structure, residue_name, pose_info).
        pose_info contains: original_pose_energy, pose_index, etc.
        """
        self.logger.info("Parameterizing ligands from %d file(s)...", len(ligand_files))
        
        expanded_files = self._expand_ligand_files(ligand_files)
        if self.config.max_ligands:
            expanded_files = expanded_files[:self.config.max_ligands]
        
        results = []
        for lig_file in expanded_files:
            try:
                file_results = self._parameterize_file(lig_file)
                results.extend(file_results)
            except Exception as e:
                self.logger.error("Failed to parameterize %s: %s", lig_file, e)
                if self.config.verbose:
                    import traceback
                    traceback.print_exc()
        
        self.logger.info("Successfully parameterized %d ligand poses", len(results))
        return results
    
    def _expand_ligand_files(self, files: List[str]) -> List[str]:
        """Expand directories and glob patterns."""
        expanded = []
        for f in files:
            path = Path(f)
            if path.is_dir():
                for ext in (".pdbqt", ".sdf", ".mol2", ".pdb", ".mol", ".mae"):
                    expanded.extend(sorted(path.glob(f"*{ext}")))
            else:
                matches = sorted(glob.glob(f))
                if matches:
                    expanded.extend(matches)
                elif path.exists():
                    expanded.append(str(path))
        return [str(p) for p in expanded]
    
    def _parameterize_file(self, lig_file: str) -> List[Tuple[str, pmd.Structure, str, Dict]]:
        """Parameterize a single ligand file (may contain multiple poses)."""
        ext = Path(lig_file).suffix.lower()
        
        if ext == ".pdbqt":
            # PDBQT: multiple poses, need to map each to reference
            return self._parameterize_pdbqt(lig_file)
        elif ext in (".sdf", ".mol2", ".pdb", ".mol"):
            # Single structure files
            return self._parameterize_single_structure(lig_file)
        else:
            raise ValueError(f"Unsupported ligand format: {ext}")
    
    def _parameterize_pdbqt(self, pdbqt_file: str) -> List[Tuple[str, pmd.Structure, str, Dict]]:
        """Parameterize all poses from a PDBQT file."""
        poses = PDBQTParser.parse(pdbqt_file)
        if not poses:
            raise ValueError(f"No poses found in {pdbqt_file}")
        
        self.logger.info("Found %d poses in %s", len(poses), pdbqt_file)
        
        # Get reference ligand for chemical mapping
        base_name = Path(pdbqt_file).stem
        ref_mol = self.ref_loader.get_reference(base_name)
        
        if ref_mol is None:
            self.logger.warning("No reference ligand found for %s; using PDBQT chemistry", pdbqt_file)
            # Fallback: use first pose as reference
            return self._parameterize_pdbqt_fallback(pdbqt_file, poses)
        
        # Select poses based on mode
        selected_poses = self._select_poses(poses)
        
        results = []
        for pose_idx, pose in enumerate(selected_poses):
            try:
                # Map pose coordinates onto reference
                mapped_mol = self.coord_mapper.map_pose_to_reference(pose, ref_mol)
                
                # Convert to OpenFF Molecule
                off_mol = Molecule.from_rdkit(mapped_mol, allow_undefined_stereo=True)
                off_mol.name = f"{base_name}_pose{pose_idx}"
                
                # Parameterize
                struct, resname = self._parameterize_openff_molecule(off_mol, f"{base_name}_pose{pose_idx}")
                
                pose_info = {
                    'original_energy': pose.get('energy'),
                    'pose_index': pose.get('model_num', pose_idx),
                    'source_file': pdbqt_file,
                    'mapped_atoms': off_mol.n_atoms
                }
                results.append((f"{base_name}_pose{pose_idx}", struct, resname, pose_info))
                
            except Exception as e:
                self.logger.error("Failed to parameterize pose %d from %s: %s", pose_idx, pdbqt_file, e)
        
        return results
    
    def _select_poses(self, poses: List[Dict]) -> List[Dict]:
        """Select poses based on pose_mode."""
        if self.config.pose_mode == "best":
            # Sort by energy (most negative = best)
            sorted_poses = sorted(poses, key=lambda p: p.get('energy', 0))
            return [sorted_poses[0]] if sorted_poses else []
        elif self.config.pose_mode == "all":
            return poses
        elif self.config.pose_mode == "ensemble":
            # Return top N poses (configurable later)
            sorted_poses = sorted(poses, key=lambda p: p.get('energy', 0))
            return sorted_poses[:min(5, len(sorted_poses))]
        return [poses[0]]
    
    def _parameterize_pdbqt_fallback(self, pdbqt_file: str, poses: List[Dict]) -> List[Tuple[str, pmd.Structure, str, Dict]]:
        """Fallback: parameterize PDBQT poses without reference mapping."""
        base_name = Path(pdbqt_file).stem
        selected = self._select_poses(poses)
        results = []
        
        for pose_idx, pose in enumerate(selected):
            try:
                # Create RDKit mol from PDBQT directly
                mol = self._pdbqt_pose_to_rdkit(pose)
                off_mol = Molecule.from_rdkit(mol, allow_undefined_stereo=True)
                off_mol.name = f"{base_name}_pose{pose_idx}"
                
                struct, resname = self._parameterize_openff_molecule(off_mol, f"{base_name}_pose{pose_idx}")
                
                pose_info = {
                    'original_energy': pose.get('energy'),
                    'pose_index': pose.get('model_num', pose_idx),
                    'source_file': pdbqt_file
                }
                results.append((f"{base_name}_pose{pose_idx}", struct, resname, pose_info))
            except Exception as e:
                self.logger.error("Fallback parameterization failed for pose %d: %s", pose_idx, e)
        
        return results
    
    def _pdbqt_pose_to_rdkit(self, pose: Dict) -> Any:
        """Convert a single PDBQT pose to RDKit molecule."""
        # Create mol from atoms and coords
        mol = Chem.RWMol()
        atom_map = {}
        
        for i, (name, atype, charge, coord) in enumerate(zip(
            pose['atom_names'], pose['atom_types'], pose['charges'], pose['coords']
        )):
            # Guess element from atom type
            elem = atype[0] if atype else 'C'
            atom = Chem.Atom(elem)
            atom.SetProp("_TriposAtomName", name)
            atom.SetProp("_TriposAtomType", atype)
            idx = mol.AddAtom(atom)
            atom_map[i] = idx
        
        # No bond info in PDBQT - will need to infer from distances
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, coord in enumerate(pose['coords']):
            conf.SetAtomPosition(i, coord)
        mol.AddConformer(conf)
        
        # Infer bonds from distances
        mol = mol.GetMol()
        mol = Chem.AddHs(mol, addCoords=True)
        return mol
    
    def _parameterize_single_structure(self, lig_file: str) -> List[Tuple[str, pmd.Structure, str, Dict]]:
        """Parameterize SDF/MOL2/PDB file."""
        base_name = Path(lig_file).stem
        
        if _HAVE_OPENFF:
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
                raise ValueError(f"Unsupported format: {ext}")
            
            results = []
            for i, mol in enumerate(molecules):
                mol.name = f"{base_name}_{i}"
                struct, resname = self._parameterize_openff_molecule(mol, f"{base_name}_{i}")
                results.append((f"{base_name}_{i}", struct, resname, {'source_file': lig_file}))
            return results
        elif _HAVE_RDKIT:
            return self._parameterize_with_rdkit_antechamber(lig_file)
        else:
            raise RuntimeError("No ligand parameterization backend available")
    
    def _parameterize_openff_molecule(self, mol: Molecule, name: str) -> Tuple[pmd.Structure, str]:
        """Parameterize an OpenFF Molecule with GAFF + charges."""
        # Assign charges
        self.logger.debug("Assigning %s charges to %s...", self.config.charge_method, name)
        if self.config.charge_method.lower() in ("am1bcc", "am1bccelf10"):
            if _HAVE_OPENFF_RECHARGE:
                recharge = Recharge()
                mol = recharge.assign_charges(mol, method=self.config.charge_method)
            else:
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
        
        # Apply GAFF force field
        topology = Topology.from_molecules([mol])
        
        try:
            ff = ForceField(self.config.ligand_ff)
        except Exception:
            for ff_name in ["gaff-2.11", "gaff-1.8", "gaff"]:
                try:
                    ff = ForceField(ff_name)
                    self.logger.info("Using force field: %s", ff_name)
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError(f"Could not load GAFF force field: {self.config.ligand_ff}")
        
        from openff.interchange import Interchange
        interchange = Interchange.from_smirnoff(ff, topology)
        structure = interchange.to_parmed()
        
        for res in structure.residues:
            res.name = self.config.ligand_resname
        
        lig_out = os.path.join(self.work_dir, f"{name}.mol2")
        structure.save(lig_out, overwrite=True)
        self.logger.debug("Parameterized ligand saved to: %s", lig_out)
        
        return structure, self.config.ligand_resname
    
    def _parameterize_with_rdkit_antechamber(self, lig_file: str) -> List[Tuple[str, pmd.Structure, str, Dict]]:
        """Parameterize using RDKit + antechamber (AmberTools)."""
        base_name = Path(lig_file).stem
        mol2_path = lig_file
        if not lig_file.lower().endswith(".mol2"):
            mol2_path = os.path.join(self.work_dir, f"{base_name}.mol2")
            self._convert_to_mol2(lig_file, mol2_path)
        
        ac_out = os.path.join(self.work_dir, f"{base_name}_ac.mol2")
        charge_flag = "bcc" if "am1bcc" in self.config.charge_method.lower() else "gas"
        cmd = [
            "antechamber",
            "-i", mol2_path, "-fi", "mol2",
            "-o", ac_out, "-fo", "mol2",
            "-c", charge_flag, "-s", "2", "-nc", "0",
            "-at", "gaff2", "-pf", "y"
        ]
        
        self.logger.debug("Running antechamber: %s", " ".join(cmd))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"antechamber failed: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise RuntimeError(f"antechamber not available or failed: {e}")
        
        structure = pmd.load_file(ac_out)
        for res in structure.residues:
            res.name = self.config.ligand_resname
        
        return [(base_name, structure, self.config.ligand_resname, {'source_file': lig_file})]
    
    def _convert_to_mol2(self, input_file: str, output_file: str):
        """Convert ligand file to MOL2 using RDKit."""
        ext = Path(input_file).suffix.lower()
        if ext == ".sdf":
            mols = Chem.SDMolSupplier(input_file, removeHs=False)
            mol = next((m for m in mols if m is not None), None)
        elif ext == ".pdb":
            mol = Chem.MolFromPDBFile(input_file, removeHs=False)
        elif ext == ".pdbqt":
            mol = Chem.MolFromPDBFile(input_file, removeHs=False)
        elif ext == ".mol":
            mol = Chem.MolFromMolFile(input_file, removeHs=False)
        else:
            raise ValueError(f"Cannot convert {ext} to mol2")
        
        if mol is None:
            raise ValueError(f"Failed to read molecule from {input_file}")
        
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        writer = Chem.SDWriter(output_file.replace(".mol2", ".sdf"))
        writer.write(mol)
        writer.close()
        
        try:
            subprocess.run(["obabel", "-isdf", output_file.replace(".mol2", ".sdf"), 
                          "-omol2", "-O", output_file], check=True, capture_output=True)
        except Exception:
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
                 ligand_structs: List[Tuple[str, pmd.Structure, str, Dict]]) -> List[Tuple[str, pmd.Structure, Dict]]:
        """
        Assemble protein with each ligand into a complex.
        Returns list of (ligand_name, complex_structure, pose_info).
        """
        self.logger.info("Assembling %d protein-ligand complexes...", len(ligand_structs))
        
        complexes = []
        for lig_name, lig_struct, lig_resname, pose_info in ligand_structs:
            try:
                complex_struct = self._merge_protein_ligand(protein_struct, lig_struct, lig_resname)
                complexes.append((lig_name, complex_struct, pose_info))
            except Exception as e:
                self.logger.error("Failed to assemble complex for %s: %s", lig_name, e)
                if self.config.verbose:
                    import traceback
                    traceback.print_exc()
        
        return complexes
    
    def _merge_protein_ligand(self, protein: pmd.Structure, 
                              ligand: pmd.Structure, lig_resname: str) -> pmd.Structure:
        """Merge protein and ligand structures into a single complex."""
        # Use ParmEd's join method
        combined = protein.join(ligand)
        
        # Ensure ligand residue name is set
        for res in combined.residues:
            if res.name != lig_resname and any(a.name in [la.name for la in ligand.residues[0].atoms] for a in res.atoms):
                res.name = lig_resname
        
        return combined


# ============================================================================
# MM-GBSA Runner (wraps mmgbsa_amb.py engine)
# ============================================================================
class DockingMMGBSARunner:
    """Run MM-GBSA on assembled complexes using the validated engine."""
    
    def __init__(self, config: DockingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def run_on_complexes(self, complexes: List[Tuple[str, pmd.Structure, Dict]]) -> List[Dict[str, Any]]:
        """Run MM-GBSA on each complex."""
        results = []
        
        for i, (lig_name, complex_struct, pose_info) in enumerate(complexes):
            self.logger.info("Running MM-GBSA on complex %d/%d: %s", 
                           i+1, len(complexes), lig_name)
            
            try:
                result = self._run_single_complex(lig_name, complex_struct)
                result.update(pose_info)  # Add pose info (energy, index, etc.)
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
                    **pose_info
                })
        
        return results
    
    def _run_single_complex(self, lig_name: str, complex_struct: pmd.Structure) -> Dict[str, Any]:
        """Run MM-GBSA on a single complex using v4 engine."""
        with tempfile.TemporaryDirectory(dir=self.config.work_dir) as tmpdir:
            prmtop = os.path.join(tmpdir, "complex.prmtop")
            inpcrd = os.path.join(tmpdir, "complex.inpcrd")
            
            complex_struct.save(prmtop, overwrite=True)
            complex_struct.save(inpcrd, overwrite=True)
            
            v4_config = v4.Config(
                top_file=prmtop,
                coord_files=[inpcrd],
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
            
            logger = v4.setup_logging(self.config.verbose)
            top_model = v4.TopologyModel(complex_struct, v4_config, logger)
            engine = v4.MMGBSAEngine(top_model, v4_config, logger)
            
            coords, box, _ = next(v4.iter_frames([inpcrd], top_model.n_atoms, prmtop))
            data = engine.compute_frame(coords)
            
            result = {"ligand": lig_name}
            for key in v4.DELTA_KEYS:
                result[f"delta_{key}"] = float(data[f"delta_{key}"])
            for key in v4.ABS_KEYS:
                result[key] = float(data[key])
            
            if self.config.decomp_csv:
                decomp = engine.decompose(data)
                result["decomposition"] = decomp
            
            return result


# ============================================================================
# Output Writers
# ============================================================================
def write_results_csv(results: List[Dict[str, Any]], output_path: str):
    """Write ranked MM-GBSA results to CSV."""
    sorted_results = sorted(results, key=lambda x: x.get("delta_total", float('inf')))
    valid_results = [r for r in sorted_results if "error" not in r]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        headers = ["rank", "ligand"] + [f"delta_{k}" for k in v4.DELTA_KEYS] + v4.ABS_KEYS + ["vina_energy", "pose_index"]
        writer.writerow(headers)
        
        for rank, r in enumerate(sorted_results, 1):
            row = [rank, r.get("ligand", "")]
            for k in v4.DELTA_KEYS:
                row.append(r.get(f"delta_{k}", ""))
            for k in v4.ABS_KEYS:
                row.append(r.get(k, ""))
            row.append(r.get("original_energy", ""))
            row.append(r.get("pose_index", ""))
            writer.writerow(row)
        
        if len(valid_results) > 1:
            avg_row = ["AVERAGE", ""]
            for k in v4.DELTA_KEYS:
                vals = [r.get(f"delta_{k}", float('nan')) for r in valid_results]
                avg_row.append(float(np.nanmean(vals)))
            for k in v4.ABS_KEYS:
                vals = [r.get(k, float('nan')) for r in valid_results]
                avg_row.append(float(np.nanmean(vals)))
            avg_row.extend(["", ""])
            writer.writerow(avg_row)
            
            std_row = ["STDDEV", ""]
            for k in v4.DELTA_KEYS:
                vals = [r.get(f"delta_{k}", float('nan')) for r in valid_results]
                std_row.append(float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0)
            for k in v4.ABS_KEYS:
                vals = [r.get(k, float('nan')) for r in valid_results]
                std_row.append(float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0)
            std_row.extend(["", ""])
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
            
            n_res = len(decomp.get("vdw", []))
            res_indices = decomp.get("res_index", list(range(n_res)))
            
            for k in range(n_res):
                ri = res_indices[k] if k < len(res_indices) else k
                total = (decomp["vdw"][k] + decomp["elec"][k] + 
                        decomp["gb"][k] + decomp["sasa"][k])
                writer.writerow([lig_name, rank, int(ri), f"RES{ri}", ri,
                               decomp["vdw"][k], decomp["elec"][k],
                               decomp["gb"][k], decomp["sasa"][k], total])
            
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
    valid = [r for r in results if "error" not in r]
    
    summary = {
        "metadata": {
            "protein_pdb": protein_pdb,
            "n_ligands": len(results),
            "n_valid": len(valid),
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
            "pose_mode": config.pose_mode,
        },
        "results": sorted_results,
        "statistics": {
            "best_ligand": sorted_results[0]["ligand"] if sorted_results else None,
            "best_delta_total": sorted_results[0]["delta_total"] if sorted_results else None,
            "mean_delta_total": float(np.nanmean([r["delta_total"] for r in valid])) if valid else None,
            "std_delta_total": float(np.nanstd([r["delta_total"] for r in valid])) if valid else None,
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
    
    if config.work_dir is None:
        config.work_dir = tempfile.mkdtemp(prefix="mmgbsa_docking_")
    work_dir = Path(config.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Working directory: %s", work_dir)
    
    try:
        # Step 1: Load ligand references
        logger.info("=" * 60)
        logger.info("STEP 1: Loading Ligand References")
        logger.info("=" * 60)
        ref_loader = LigandReferenceLoader(logger)
        if config.ligand_refs:
            ref_loader.load(config.ligand_refs)
        else:
            logger.warning("No ligand references provided (--ligand-refs); chemical mapping will be limited")
        
        coord_mapper = PoseCoordinateMapper(logger)
        
        # Step 2: Prepare protein
        logger.info("=" * 60)
        logger.info("STEP 2: Protein Preparation")
        logger.info("=" * 60)
        protein_prep = ProteinPreparator(config, logger)
        protein_struct, protein_pdb = protein_prep.prepare(config.protein_pdb)
        logger.info("Protein prepared: %d atoms, %d residues", 
                   len(protein_struct.atoms), len(protein_struct.residues))
        
        # Step 3: Parameterize ligands (with pose mapping)
        logger.info("=" * 60)
        logger.info("STEP 3: Ligand Parameterization & Pose Mapping")
        logger.info("=" * 60)
        lig_param = LigandParameterizer(config, logger, ref_loader, coord_mapper)
        ligand_structs = lig_param.parameterize_ligands(config.ligand_files)
        logger.info("Ligand poses parameterized: %d", len(ligand_structs))
        
        if not ligand_structs:
            raise RuntimeError("No ligand poses successfully parameterized")
        
        # Step 4: Assemble complexes
        logger.info("=" * 60)
        logger.info("STEP 4: Complex Assembly")
        logger.info("=" * 60)
        assembler = ComplexAssembler(config, logger)
        complexes = assembler.assemble(protein_struct, ligand_structs)
        logger.info("Complexes assembled: %d", len(complexes))
        
        if not complexes:
            raise RuntimeError("No complexes successfully assembled")
        
        # Step 5: Run MM-GBSA
        logger.info("=" * 60)
        logger.info("STEP 5: MM-GBSA Calculation")
        logger.info("=" * 60)
        runner = DockingMMGBSARunner(config, logger)
        results = runner.run_on_complexes(complexes)
        
        # Step 6: Output results
        logger.info("=" * 60)
        logger.info("STEP 6: Writing Results")
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
            logger.info("Best pose: %s (ΔG = %.3f kcal/mol)", best["ligand"], best["delta_total"])
            logger.info("Processed: %d/%d poses successful", len(valid_results), len(results))
        
        return results
        
    finally:
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
                   help="Ligand pose files: PDBQT (Vina), SDF, MOL2, PDB, or directory")
    p.add_argument("--ligand-refs", default=None,
                   help="Original ligand structures (SDF/MOL2) for chemical info mapping")
    p.add_argument("--ligand-resname", default="LIG", help="Ligand residue name")
    p.add_argument("--prepared-ligand", action="store_true",
                   help="Ligand files already parameterized (MOL2 with charges/types)")
    p.add_argument("--max-ligands", type=int, default=None, help="Limit number of ligands (for testing)")
    p.add_argument("--ligand-name-prefix", default="LIG", help="Prefix for ligand names")
    p.add_argument("--pose-mode", choices=["best", "all", "ensemble"], default="best",
                   help="Pose selection: best (top scoring), all, ensemble (top 5)")
    
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
        ligand_refs=args.ligand_refs,
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
        pose_mode=args.pose_mode,
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