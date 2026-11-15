# Docking Rescoring Example

This example demonstrates how to use `mmgbsa_docking.py` for physics-based rescoring of docked protein-ligand complexes.

## Prerequisites

```bash
# Install docking dependencies
pip install "easy-mmgbsa[docking]"

# Or install individually:
# pip install openff-toolkit openff-recharge openff-interchange pdbfixer propka3 rdkit
```

## Input Files

You need:
1. **Protein PDB** - A protein structure file (can be from PDB, homology model, etc.)
2. **Ligand files** - One or more ligand files in SDF, MOL2, or PDB format (from docking)

### Example Directory Structure
```
project/
├── protein.pdb          # Protein structure
├── ligands/             # Directory of docked ligand poses
│   ├── pose_1.sdf
│   ├── pose_2.sdf
│   ├── pose_3.sdf
│   └── ...
└── run_rescoring.sh     # Script to run rescoring
```

## Basic Usage

### Simple Rescoring
```bash
mmgbsa-docking -p protein.pdb -l ligands/ -o results.csv
```

### With Decomposition and JSON Summary
```bash
mmgbsa-docking -p protein.pdb -l ligands/ \
    --decomp decomp.csv --json summary.json -o results.csv
```

### Custom Force Fields and Settings
```bash
mmgbsa-docking -p protein.pdb -l ligands/ \
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

### Pre-parameterized Ligands (from antechamber)
```bash
mmgbsa-docking -p protein.pdb -l ligand.mol2 \
    --prepared-ligand -o results.csv
```

### Testing with Subset
```bash
mmgbsa-docking -p protein.pdb -l ligands/ \
    --max-ligands 10 --keep-intermediate -v -o test_results.csv
```

## Output Files

### `results.csv` - Ranked Binding Energies
```
rank,ligand,delta_total,delta_vdw,delta_elec,delta_gb,delta_sasa,...
1,pose_42,-45.2,-50.1,-200.3,180.5,-5.3,...
2,pose_15,-42.8,-48.2,-195.1,175.2,-4.9,...
...
AVERAGE,,-43.5,-49.2,-197.7,177.8,-5.1,...
STDDEV,,1.2,1.0,2.6,2.7,0.2,...
```

### `decomp.csv` - Per-Residue Decomposition
```
ligand,rank,residue_index,resname,resid,vdw,elec,gb,sasa,total
pose_42,1,0,ALA,10,-0.5,-2.1,1.8,-0.1,-0.9
pose_42,1,1,GLU,11,-1.2,-5.3,4.1,-0.2,-2.6
...
pose_42,1,LIGAND,LIG,,-3.1,-12.5,8.2,-0.3,-7.7
pose_15,2,0,ALA,10,-0.4,-1.9,1.7,-0.1,-0.7
...
```

### `summary.json` - Machine-Readable Summary
```json
{
  "metadata": {
    "protein_pdb": "protein.pdb",
    "n_ligands": 100,
    "protein_ff": "ff14SB",
    "ligand_ff": "gaff-2.11",
    "charge_method": "am1bcc",
    "ph": 7.4,
    "igb": 5,
    "eps_solvent": 78.3,
    "gb_cutoff": 25.0,
    "radii_set": "auto",
    "surface_tension": 0.0072,
    "probe_radius": 1.4
  },
  "results": [...],
  "statistics": {
    "best_ligand": "pose_42",
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
1. PROTEIN PREPARATION
   ├── PDBFixer: Add missing atoms, residues, hydrogens
   ├── PROPKA: Assign protonation states at target pH
   └── tleap/AMBER: Apply protein force field (ff14SB/ff19SB)

2. LIGAND PARAMETERIZATION
   ├── OpenFF/RDKit: Parse ligand structure(s)
   ├── AM1-BCC: Compute partial charges (semi-empirical QM)
   └── GAFF2: Assign atom types and parameters

3. COMPLEX ASSEMBLY
   └── Merge protein + ligand into single AMBER topology

4. MM-GBSA CALCULATION (validated engine)
   ├── OBC GB (igb=2/5) with HCT descreening
   ├── AMBER-consistent 1-4 terms
   ├── Shrake-Rupley SASA (Bondi radii)
   └── Per-residue decomposition (AMBER idecomp convention)

5. OUTPUT
   ├── Ranked ΔGbind with components
   ├── Per-residue decomposition
   └── JSON summary
```

## Common Issues and Solutions

### "No ligand atoms selected"
- Ensure ligand files have proper residue names
- Use `--ligand-resname` if residue name is non-standard

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

### Memory Issues with Large Ligand Sets
- Use `--max-ligands` to process in batches
- Increase `--work-dir` to a location with more disk space
- Use `--keep-intermediate` to debug

## Performance Tips

1. **Use AM1-BCC charges** for best accuracy (requires OpenEye or AmberTools)
2. **Process in batches** if you have >1000 ligands
3. **Use SSD** for `--work-dir` for faster I/O
4. **Parallelize manually** by splitting ligand sets across multiple runs

## References

- AMBER ff14SB/ff19SB: Maier et al., J. Chem. Theory Comput. 2015
- GAFF2: Wang et al., J. Phys. Chem. B 2004
- AM1-BCC: Jakalian et al., J. Chem. Inf. Model. 2002
- OBC GB: Onufriev et al., Proteins 2004
- HCT descreening: Hawkins et al., J. Phys. Chem. 1996
- Shrake-Rupley SASA: Shrake & Rupley, J. Mol. Biol. 1973