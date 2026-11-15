#!/usr/bin/env python3
"""
MM-GBSA GROMACS - Fast vectorized engine with GROMACS-oriented interface.

Uses mmgbsa_amber_v4.py's validated engine under the hood.

Usage:
    python3 mmgbsa_gmx.py -p topol.top -l lig.itp -c frame_*.gro -o results.csv --decomp decomp.csv
    python3 mmgbsa_gmx.py -p topol.top -c traj.xtc --lig-resname LIG -o results.csv --decomp decomp.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Import the validated v4 engine
sys.path.insert(0, os.path.dirname(__file__))
import mmgbsa_amb as v4
import parmed as pmd

# ============================================================================
# GROMACS-specific topology loading
# ============================================================================
def load_gmx_topology(top_file: str, lig_itp: str | None, lig_resname: str | None,
                       strip: str, no_strip: bool, logger: logging.Logger
                       ) -> tuple[pmd.Structure, str, set]:
    """Load GROMACS topol.top (+ optional lig.itp), return (structure, ligand_resname, strip_set)."""
    logger.info("Loading topology: %s", top_file)
    # ParmEd auto-detects format; for GROMACS .top it resolves #includes, [pairs], [defaults]
    structure = pmd.load_file(top_file)
    
    # If it's a GROMACS topology, parametrize it
    if hasattr(structure, 'parametrize'):
        structure.parametrize()
    
    # Determine ligand residue name
    if lig_resname:
        logger.info("Using user-specified ligand residue: %s", lig_resname)
    elif lig_itp:
        logger.info("Loading ligand ITP: %s", lig_itp)
        lig_struct = pmd.load_file(lig_itp, parametrize=True)
        for res in lig_struct.residues:
            lig_resname = res.name
            break
        logger.info("Ligand residue from itp: %s", lig_resname)
        # Merge ligand parameters into main structure
        for res in structure.residues:
            if res.name == lig_resname:
                by_name = {a.name: a for a in lig_struct.atoms}
                for atom in res.atoms:
                    if atom.name in by_name:
                        src = by_name[atom.name]
                        atom.type = src.type
                        atom.charge = src.charge
                        atom.mass = src.mass
                        atom.atomic_number = src.atomic_number
                logger.info("Merged %d ligand atoms from itp", len(res.atoms))
                break
    else:
        # Auto-detect: first non-standard residue
        std_protein = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","HID","HIP",
                       "ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}
        std_solvent = {"WAT","HOH","SOL","TIP3","TIP4","SPC","DOD",
                       "NA","CL","K","MG","ZN","CA","FE","MN","CU",
                       "NA+","CL-","K+","MG2","ZN2","CA2"}
        for res in structure.residues:
            if res.name not in std_protein and res.name not in std_solvent:
                lig_resname = res.name
                logger.info("Auto-detected ligand: %s", lig_resname)
                break
    
    if not lig_resname:
        raise ValueError("Could not determine ligand residue. Use --lig-resname or provide lig.itp")
    
    strip_set = set(strip.split(",")) if strip else set()
    return structure, lig_resname, strip_set


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="MM-GBSA GROMACS: fast vectorized engine with GROMACS interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-p", "--top", required=True, help="GROMACS topology (topol.top)")
    p.add_argument("-l", "--lig-itp", default=None, help="Ligand ITP file (optional)")
    p.add_argument("--lig-resname", default=None, help="Ligand residue name (e.g., LIG, UNL)")
    p.add_argument("--lig-resid-range", default=None, help="Ligand residue range START-END (1-based)")
    p.add_argument("-c", "--coords", nargs="+", required=True,
                   help="Coordinate files: .gro, .pdb, .xtc, .trr, .nc (globs ok)")
    p.add_argument("-o", "--output", default="mmgbsa_results.csv", help="Output CSV")
    p.add_argument("--decomp", default=None, help="Per-residue decomposition CSV")
    p.add_argument("--json", default=None, help="JSON summary output")
    
    p.add_argument("--igb", type=int, choices=[2, 5], default=5, help="GB model: 2=OBC-I, 5=OBC-II")
    p.add_argument("--eps-solvent", type=float, default=78.3)
    p.add_argument("--gb-cutoff", type=float, default=25.0, help="GB radii cutoff (Å), 0=disable")
    p.add_argument("--radii", choices=["auto","mbondi","mbondi2","mbondi3","bondi"], default="auto")
    p.add_argument("--sasa-radii", choices=["bondi","gb"], default="bondi")
    p.add_argument("--probe", type=float, default=1.4)
    p.add_argument("--surface-tension", type=float, default=0.0072)
    p.add_argument("--sasa-points", type=int, default=240)
    
    p.add_argument("--strip", default="WAT,HOH,SOL,TIP3,TIP4,SPC,DOD,NA,CL,K,MG,ZN,CA,FE,MN,CU,NA+,CL-,K+,MG2,ZN2,CA2",
                   help="Residue names to strip (comma-separated)")
    p.add_argument("--no-strip", action="store_true")
    
    p.add_argument("--start", type=int, default=1, help="First frame (1-based)")
    p.add_argument("--end", type=int, default=None, help="Last frame (1-based)")
    p.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logger = v4.setup_logging(args.verbose)
    
    # Load topology
    structure, lig_resname, strip_set = load_gmx_topology(
        args.top, args.lig_itp, args.lig_resname, args.strip, args.no_strip, logger
    )
    
    # Build v4 config
    resid_range = None
    if args.lig_resid_range:
        try:
            lo, hi = map(int, args.lig_resid_range.split("-"))
            resid_range = (lo, hi)
        except Exception:
            logger.error("Invalid --lig-resid-range format. Use START-END (e.g., 167-242)")
            return 1
    
    config = v4.Config(
        top_file=args.top,
        coord_files=list(args.coords),
        ligand_resnames=[lig_resname] if lig_resname and not resid_range else [],
        ligand_resid_range=resid_range,
        ligand_mask=None,
        strip_resnames=strip_set,
        no_strip=args.no_strip,
        igb=args.igb,
        eps_solvent=args.eps_solvent,
        radii_set=args.radii,
        gb_radii_cutoff=args.gb_cutoff if args.gb_cutoff > 0 else None,
        probe_radius=args.probe,
        sasa_radii_set=args.sasa_radii,
        surface_tension=args.surface_tension,
        sasa_points=args.sasa_points,
        start_frame=args.start,
        end_frame=args.end,
        stride=args.stride,
        output_csv=args.output,
        decomp=args.decomp is not None,
        decomp_csv=args.decomp,
        json_output=args.json,
        verbose=args.verbose,
    )
    
    # Run v4 main (it handles everything)
    # We need to call v4.main with our config - but v4.main uses argparse
    # Instead, replicate the core loop here using v4 components
    
    logger.info("Building v4 engine...")
    top_model = v4.TopologyModel(structure, config, logger)
    engine = v4.MMGBSAEngine(top_model, config, logger)
    
    # Process frames
    coord_files = v4.expand_coord_files(config.coord_files)
    if not coord_files:
        logger.error("No coordinate files found")
        return 1
    
    rows = []
    decomp_rows = []
    frame_counter = 0
    used = 0
    
    for coords, box, label in v4.iter_frames(coord_files, top_model.n_atoms, config.top_file):
        frame_counter += 1
        if frame_counter < config.start_frame:
            continue
        if config.end_frame and frame_counter > config.end_frame:
            break
        if (frame_counter - config.start_frame) % max(1, config.stride) != 0:
            continue
        
        if coords.shape[0] != top_model.n_atoms:
            logger.warning("%s: %d atoms vs topology %d - skip", label, coords.shape[0], top_model.n_atoms)
            continue
        
        data = engine.compute_frame(coords)
        rows.append(data)
        used += 1
        
        if config.decomp:
            decomp_rows.append(engine.decompose(data))
        
        if used % 10 == 0 or config.verbose:
            logger.info("Frame %d (%s): dG=%.3f [vdw=%.2f elec=%.2f gb=%.2f sasa=%.2f]",
                        frame_counter, os.path.basename(label), data["delta_total"],
                        data["delta_vdw"], data["delta_elec"], data["delta_gb"], data["delta_sasa"])
    
    if not rows:
        logger.error("No frames processed")
        return 1
    
    # Write outputs using v4 helpers
    v4.write_frame_csv(config.output_csv, rows, config)
    logger.info("Results written to %s", config.output_csv)
    
    if config.decomp:
        v4.write_decomposition_csv(config.decomp_csv, top_model, decomp_rows, logger)
    
    if config.json_output:
        # Build JSON payload (similar to v4.main)
        summary = top_model.summary()
        gb_params = v4.OBC_PARAMETERS[config.igb]
        payload = {
            "system": summary,
            "config": {
                "igb": config.igb,
                "obc_alpha": config.obc_alpha or gb_params["alpha"],
                "obc_beta": config.obc_beta or gb_params["beta"],
                "obc_gamma": config.obc_gamma or gb_params["gamma"],
                "obc_offset": config.obc_offset or gb_params["offset"],
                "gb_radii_cutoff": config.gb_radii_cutoff,
                "radii_set": config.radii_set,
                "eps_solute": config.eps_solute,
                "eps_solvent": config.eps_solvent,
                "sasa_radii_set": config.sasa_radii_set,
                "surface_tension": config.surface_tension,
                "probe_radius": config.probe_radius,
                "sasa_points": config.sasa_points,
                "ligand_selection": {"resnames": list(config.ligand_resnames)},
                "frames": len(rows),
            },
            "delta": {
                k: dict(zip(("mean","std","sem"),
                           v4.stats([r[f"delta_{k}"] for r in rows])))
                for k in v4.DELTA_KEYS
            },
        }
        import json
        with open(config.json_output, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("JSON written to %s", config.json_output)
    
    return 0


if __name__ == "__main__":
    import os
    sys.exit(main())