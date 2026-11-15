# Docking Rescoring Example

This example demonstrates how to use `mmgbsa_docking.py` for physics-based rescoring of docked protein-ligand complexes from AutoDock Vina, AutoDock4, and other docking programs.

## Prerequisites

```bash
# Install docking dependencies
pip install "easy-mmgbsa[docking]"

# Or install individually:
# pip install openff-toolkit openff-recharge openff-interchange pdbfixer propka3 rdkit

# System dependencies (for AMBER force fields):
# conda install -c conda-forge ambertools  # provides tleap, antechamber
```

## Input Files

You need **THREE** inputs for proper docking rescoring:

1. **Protein PDB** - Receptor structure (from PDB, homology model, etc.)
2. **Docked poses** - PDBQT files from AutoDock Vina (or SDF/MOL2/PDB from other programs)
3. **Ligand references** - Original ligand library SDF/MOL2 with correct chemistry (bond orders, stereochemistry)

### Why ligand references?
PDBQT files from Vina **do not preserve** bond orders, formal charges, or stereochemistry. The `--ligand-refs` file provides the correct chemical graph, while the PDBQT provides the docked coordinates. The engine maps coordinates onto the reference chemistry.

### Example Directory Structure (AutoDock Vina output)
```
project/
├── protein.pdb              # Protein structure
├── ligands.sdf              # Original ligand library (for chemistry)
├── vina_output/             # AutoDock Vina output directory
│   ├── ligand_1_out.pdbqt   # Docked poses (multiple MODEL blocks)
│   ├── ligand_2_out.pdbqt
│   ├── ligand_3_out.pdbqt
│   └── ...
└── run_rescoring.sh         # Script to run rescoring
```

### Example Directory Structure (Other docking programs)
```
project/
├── protein.pdb          # Protein structure
├── ligands.sdf          # Original ligand library (for chemistry)
├── docking_poses/       # Docked poses from Glide, GOLD, etc.
│   ├── pose_1.sdf
│   ├── pose_2.sdf
│   ├── pose_3.sdf
│   └── ...
└── run_rescoring.sh     # Script to run rescoring
```

## Basic Usage

### AutoDock Vina Rescoring (Recommended Workflow)
```bash
# Rescore Vina poses with chemical mapping from original ligand library
mmgbsa-docking -p protein.pdb -l vina_output/*.pdbqt \
    --ligand-refs ligands.sdf \
    -o results.csv --decomp decomp.csv --json summary.json
```

### Pose Selection Modes
```bash
# Best pose only (default) - uses Vina's top-scoring pose per ligand
mmgbsa-docking -p protein.pdb -l vina_output/*.pdbqt \
    --ligand-refs ligands.sdf --pose-mode best -o results.csv

# All poses - rescore every pose in every PDBQT file
mmgbsa-docking -p protein.pdb -l vina_output/*.pdbqt \
    --ligand-refs ligands.sdf --pose-mode all -o results.csv

# Ensemble - top 5 poses per ligand (more robust)
mmgbsa-docking -p protein.pdb -l vina_output/*.pdbqt \
    --ligand-refs ligands.sdf --pose-mode ensemble -o results.csv
```

### Other Docking Programs (SDF/MOL2/PDB input)
```bash
# For Glide, GOLD, DOCK output (SDF with multiple poses)
mmgbsa-docking -p protein.pdb -l docking_poses/*.sdf \
    --ligand-refs ligands.sdf -o results.csv

# Pre-parameterized ligands (MOL2 with GAFF/AM1-BCC from antechamber)
mmgbsa-docking -p protein.pdb -l ligand.mol2 \
    --prepared-ligand -o results.csv
```

### Custom Force Fields and Settings
```bash
mmgbsa-docking -p protein.pdb -l vina_output/*.pdbqt \
    --ligand-refs ligands.sdf \
    --protein-ff ff19SB \
    --ligand-ff gaff-2.11 \
    --charge-method am1bcc \
    --ph 7.4 \
    --igb 5 \
    --eps-solvent 78.3 \
    --gb-cutoff 25.0 \
    -o results.csv \
    --decomp decomp.csv
```

### Testing with Subset
```bash
mmgbsa-docking -p protein.pdb -l vina_output/*.pdbqt \
    --ligand-refs ligands.sdf \
    --max-ligands 10 --keep-intermediate -v -o test_results.csv
```

## Output Files

### `results.csv` - Ranked Binding Energies (with Vina traceability)
```
rank,ligand,delta_total,delta_vdw,delta_elec,delta_gb,delta_sasa,...,vina_energy,pose_index
1,ligand_1_pose0,-45.2,-50.1,-200.3,180.5,-5.3,...,-9.2,1
2,ligand_3_pose0,-42.8,-48.2,-195.1,175.2,-4.9,...,-8.7,1
3,ligand_1_pose1,-41.5,-49.0,-198.2,178.0,-4.8,...,-8.5,2
...
AVERAGE,,-43.5,-49.2,-197.7,177.8,-5.1,...,,
STDDEV,,1.2,1.0,2.6,2.7,0.2,...,,
```

**New columns:**
- `vina_energy` — Original Vina binding affinity (kcal/mol) for comparison
- `pose_index` — MODEL number from PDBQT (1-based)

### `decomp.csv` - Per-Residue Decomposition
```
ligand,rank,residue_index,resname,resid,vdw,elec,gb,sasa,total
ligand_1_pose0,1,0,ALA,10,-0.5,-2.1,1.8,-0.1,-0.9
ligand_1_pose0,1,1,GLU,11,-1.2,-5.3,4.1,-0.2,-2.6
...
ligand_1_pose0,1,LIGAND,LIG,,-3.1,-12.5,8.2,-0.3,-7.7
ligand_3_pose0,2,0,ALA,10,-0.4,-1.9,1.7,-0.1,-0.7
...
```

### `summary.json` - Machine-Readable Summary
```json
{
  "metadata": {
    "protein_pdb": "protein.pdb",
    "n_ligands": 100,
    "n_valid": 98,
    "protein_ff": "ff14SB",
    "ligand_ff": "gaff-2.11",
    "charge_method": "am1bcc",
    "ph": 7.4,
    "igb": 5,
    "eps_solvent": 78.3,
    "gb_cutoff": 25.0,
    "radii_set": "auto",
    "surface_tension": 0.0072,
    "probe_radius": 1.4,
    "pose_mode": "best"
  },
  "results": [...],
  "statistics": {
    "best_ligand": "ligand_1_pose0",
    "best_delta_total": -45.2,
    "mean_delta_total": -43.5,
    "std_delta_total": 1.2
  }
}
```

## Advanced Options

| Option | Description | Default |
|--------|-------------|---------|
| `--protein-ff` | Protein force field (ff14SB, ff19SB, ff03, ff99SB, ff99SBildn) | ff14SB |
| `--ligand-ff` | Ligand force field (GAFF variant) | gaff-2.11 |
| `--charge-method` | Partial charge method (am1bcc, am1bccelf10, gasteiger, mmff94) | am1bcc |
| `--ph` | pH for protonation state assignment | 7.4 |
| `--no-fix-protein` | Skip PDBFixer structure fixing | false |
| `--no-protonate` | Skip PROPKA protonation | false |
| `--igb` | GB model: 2=OBC-I, 5=OBC-II | 5 |
| `--eps-solvent` | Solvent dielectric constant | 78.3 |
| `--gb-cutoff` | GB radii cutoff in Å (0=disable) | 25.0 |
| `--radii` | GB radii set (auto, mbondi, mbondi2, mbondi3, bondi) | auto |
| `--probe` | Solvent probe radius for SASA (Å) | 1.4 |
| `--surface-tension` | Nonpolar surface tension (kcal/mol/Å²) | 0.0072 |
| `--sasa-points` | Shrake-Rupley sphere points per atom | 240 |

## What Happens Under the Hood

```
1. LIGAND REFERENCE LOADING
   └── Read original ligand SDF/MOL2 for bond orders, stereochemistry

2. PDBQT PARSING (AutoDock Vina)
   ├── Parse ALL MODEL/ENDMDL blocks
   ├── Extract coordinates, atom types, charges per pose
   └── Keep Vina binding affinity (REMARK VINA RESULT)

3. POSE COORDINATE MAPPING
   ├── Match PDBQT atoms to reference by name
   ├── Transfer docked coordinates to reference (NO optimization!)
   ├── Add hydrogens AFTER coordinate transfer
   └── Convert to OpenFF Molecule with correct chemistry

4. PROTEIN PREPARATION
   ├── PDBFixer: Add missing atoms, residues, hydrogens
   ├── PROPKA: Assign protonation states at target pH
   └── tleap: Apply AMBER ff14SB/ff19SB force field

5. LIGAND PARAMETERIZATION
   ├── OpenFF toolkit: GAFF2 atom types
   ├── AM1-BCC charges (OpenFF Recharge or antechamber)
   └── Convert to ParMEd Structure

6. COMPLEX ASSEMBLY
   └── Merge protein + ligand into single AMBER topology (ParmEd join)

7. MM-GBSA CALCULATION (validated engine - mmgbsa_amb.py v4)
   ├── OBC GB (igb=2/5) with HCT pairwise descreening
   ├── AMBER-consistent 1-4 terms (SCEE/SCNB from topology)
   ├── Shrake-Rupley SASA (Bondi radii, probe=1.4Å)
   └── Per-residue decomposition (AMBER idecomp convention)

8. OUTPUT
   ├── Ranked ΔGbind with components + Vina energy + pose index
   ├── Per-residue decomposition (merged, with ligand/rank)
   └── JSON summary (config, all results, statistics)
```

## Common Issues and Solutions

### "No ligand references provided" / Poor atom mapping
- **Required**: Provide `--ligand-refs ligands.sdf` with original ligand library
- Without references, PDBQT chemistry is used (no bond orders, no stereo)
- Check mapping rate in logs: "Mapped X/Y atoms" - should be >80%

### "tleap not found" / Protein not parameterized
- **Required**: `conda install -c conda-forge ambertools` (provides tleap)
- The engine will fail explicitly if tleap unavailable (no silent fallback)
- Without proper protein FF, MM-GBSA results are meaningless

### "antechamber not found" / "OpenFF Recharge not available"
- Install AmberTools: `conda install -c conda-forge ambertools`
- Or install OpenEye toolkit (commercial) for AM1-BCC
- Fallback: use `--charge-method gasteiger` (no QM required)

### "pdbfixer not available"
- Install: `conda install -c conda-forge pdbfixer`
- Or use `--no-fix-protein` to skip structure fixing

### "propka3 not available"
- Install: `pip install propka3`
- Or use `--no-protonate` to skip protonation assignment

### Low atom mapping rate ("Mapped X/Y atoms" warning)
- PDBQT atom names may differ from reference (e.g., "C1" vs "CA")
- Check reference ligand has proper atom names (Tripos/SYBYL format)
- Use `--verbose` to see detailed mapping

### Memory Issues with Large Ligand Sets
- Use `--max-ligands` to process in batches
- Increase `--work-dir` to a location with more disk space
- Use `--keep-intermediate` to debug

## Performance Tips

1. **Use AM1-BCC charges** for best accuracy (requires OpenEye or AmberTools)
2. **Process in batches** if you have >1000 ligands (`--max-ligands`)
3. **Use SSD** for `--work-dir` for faster I/O
4. **Parallelize manually** by splitting ligand sets across multiple runs
5. **Use `--pose-mode best`** for fastest screening (only top Vina pose)

## References

- AMBER ff14SB/ff19SB: Maier et al., J. Chem. Theory Comput. 2015
- GAFF2: Wang et al., J. Phys. Chem. B 2004
- AM1-BCC: Jakalian et al., J. Chem. Inf. Model. 2002
- OBC GB: Onufriev et al., Proteins 2004
- HCT descreening: Hawkins et al., J. Phys. Chem. 1996
- Shrake-Rupley SASA: Shrake & Rupley, J. Mol. Biol. 1973