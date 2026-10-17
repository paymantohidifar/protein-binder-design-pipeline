"""In-notebook 3D rendering and PyMOL script generation.

Two outputs for every checkpoint: a py3Dmol view that renders in the notebook
cell, and a .pml script for opening the same scene in a real PyMOL session.

Colour conventions: target light grey, hotspots dark red, design pose cyan, prediction magenta.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from . import pdb_io

TARGET_COLOUR = "lightgrey"
HOTSPOT_COLOUR = "darkred"
DESIGN_COLOUR = "cyan"
PREDICTION_COLOUR = "magenta"

PYMOL_TARGET_COLOUR = "grey80"
PYMOL_HOTSPOT_COLOUR = "firebrick" # PyMOL's darkred
PYMOL_DESIGN_COLOUR = "cyan"
PYMOL_PREDICTION_COLOUR = "magenta"

# Distinct colours for overlaying multiple backbones to judge binding-mode
# diversity. Chosen to stay distinguishable against grey and in both themes.
BACKBONE_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _read(path: str | Path) -> str:
    return Path(path).read_text()


def _view(width: int = 800, height: int = 520):
    import py3Dmol

    return py3Dmol.view(width=width, height=height)


def _resi_selector(labels: list[str]) -> list[str]:
    """'A59','A83' -> ['59','83'] for py3Dmol's resi selector."""
    return [str(pdb_io.parse_label(lab)[1]) for lab in labels]


def _pymol_resi_selector(labels: list[str]) -> str:
    """'A59','A83' -> '59+83' for PyMOL's resi selector."""
    return "+".join(str(pdb_io.parse_label(lab)[1]) for lab in labels)


# --- Checkpoint 1: hotspot selection ---------------------------------------


def show_target_hotspots(
    target_pdb: str | Path,
    labels: list[str],
    chain: str = "A",
    surface: bool = True,
    width: int = 800,
    height: int = 520,
):
    """Target surface in grey with hotspots as dark red sticks."""
    view = _view(width, height)
    view.addModel(_read(target_pdb), "pdb")
    view.setStyle({"cartoon": {"color": TARGET_COLOUR}})

    selection = {"chain": chain, "resi": _resi_selector(labels)}
    view.setStyle(selection, {"stick": {"color": HOTSPOT_COLOUR, "radius": 0.3},
                              "cartoon": {"color": HOTSPOT_COLOUR}})
    
    for label in labels:
        _, resnum = pdb_io.parse_label(label)
        view.addLabel(
            label,
            {"fontSize": 11, "fontColor": "white", "backgroundColor": HOTSPOT_COLOUR,
             "backgroundOpacity": 0.85},
            {"chain": chain, "resi": str(resnum), "atom": "CA"},
        )
    if surface:
        view.addSurface("VDW", {"opacity": 0.65, "color": TARGET_COLOUR}, {"chain": chain})
    view.zoomTo()
    return view


# --- Checkpoint 2: backbone diversity --------------------------------------


def _superpose_onto_design(
    target_pdb: str | Path,
    design_pdb: str | Path,
    design_target_chain: str = "B",
) -> str:
    """Input target PDB text, rotated into a design's coordinate frame.

    Needed because RFdiffusion recentres: the designs put their target copy at
    the origin, while the input target sits wherever the crystal left it.
    """
    from . import metrics

    source = pdb_io.read_pdb(target_pdb)
    chain = source.only_chain() if len(source.chains) == 1 else source[
        next(iter(source.chains))
    ]
    design_target = pdb_io.read_pdb(design_pdb)[design_target_chain]

    mobile = np.array([r.ca for r in chain.residues if r.ca is not None])
    fixed = np.array([r.ca for r in design_target.residues if r.ca is not None])
    if mobile.shape != fixed.shape:
        raise ValueError(
            f"cannot superpose {Path(target_pdb).name} onto chain "
            f"{design_target_chain} of {Path(design_pdb).name}: "
            f"{mobile.shape[0]} CA atoms vs {fixed.shape[0]}. The design's "
            "target copy should match the input target residue for residue."
        )

    _, rotation, translation = metrics.kabsch_rmsd(mobile, fixed)
    return _as_pdb_text(_transform_chain(chain, rotation, translation))


def _transform_chain(chain, rotation: np.ndarray, translation: np.ndarray):
    """Copy of `chain` with every atom moved into the target frame.

    All atoms, not just the ones that drove the fit, and B-factors are carried
    over -- they hold pLDDT, which the interface view colours by.
    """
    from . import metrics

    return pdb_io.Chain(
        chain_id=chain.chain_id,
        residues=[
            pdb_io.Residue(
                seq_id=r.seq_id,
                name=r.name,
                insertion=r.insertion,
                atoms={
                    a: metrics.apply_transform(c[None, :], rotation, translation)[0]
                    for a, c in r.atoms.items()
                },
                bfactors=dict(r.bfactors),
            )
            for r in chain.residues
        ],
    )


def _as_pdb_text(*chains) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        out = pdb_io.write_pdb(list(chains), Path(tmp) / "superposed.pdb")
        return out.read_text()


def show_backbone_ensemble(
    design_pdbs: list[str | Path],
    target_pdb: str | Path | None = None,
    hotspots: list[str] | None = None,
    target_chain: str = "B",
    binder_chain: str = "A",
    width: int = 800,
    height: int = 560,
):
    """Target cartoon plus every generated binder backbone in its own colour.

    Reveals whether RFdiffusion found one binding mode or several. A single
    tight cluster across all designs usually means the hotspots over-constrained
    the site.

    One target copy is drawn, with `hotspots` highlighted on it in dark red so
    binding modes can be read against the epitope they were conditioned on.

    `target_chain` / `binder_chain` are the chains *inside each design*.

    `hotspots` are labels in the ORIGINAL target numbering when `target_pdb` is given, 
    since that file keeps the original numbers. Without `target_pdb` the first design's 
    own target copy is used as the reference, and the labels are renumbered through its 
    .trb -- so pass original numbering either way.
    """
    if not design_pdbs:
        raise ValueError("no design PDBs to render")

    view = _view(width, height)
    designs = [Path(p) for p in design_pdbs]

    # One reference target copy. Either the input file superposed into the
    # design frame, or the first design's own copy -- which is already in the
    # right frame and needs no transform.
    if target_pdb:
        view.addModel(_superpose_onto_design(target_pdb, designs[0], target_chain), "pdb")
        reference_chain = pdb_io.read_pdb(target_pdb).chain_ids[0]
        hotspot_labels = list(hotspots or [])
    else:
        from . import trb

        view.addModel(_read(designs[0]), "pdb")
        reference_chain = target_chain
        mapping = trb.hotspots_on_design(designs[0])
        hotspot_labels = (
            [mapping.get(h, h) for h in hotspots] if hotspots
            else list(mapping.values())
        )

    view.setStyle({"model": 0}, {})
    view.setStyle({"model": 0, "chain": reference_chain},
                  {"cartoon": {"color": TARGET_COLOUR}})

    if hotspot_labels:
        resi = _resi_selector(hotspot_labels)
        view.setStyle({"model": 0, "chain": reference_chain, "resi": resi},
                      {"stick": {"color": HOTSPOT_COLOUR, "radius": 0.3},
                       "cartoon": {"color": HOTSPOT_COLOUR}})
        for label in hotspot_labels:
            _, resnum = pdb_io.parse_label(label)
            view.addLabel(
                label,
                {"fontSize": 10, "fontColor": "white",
                 "backgroundColor": HOTSPOT_COLOUR, "backgroundOpacity": 0.85},
                {"model": 0, "chain": reference_chain,
                 "resi": str(resnum), "atom": "CA"},
            )

    for i, pdb in enumerate(designs):
        view.addModel(_read(pdb), "pdb")
        model = {"model": i + 1}
        # The design file carries the target too; hide it so the single
        # reference copy is not drawn N times on top of itself.
        view.setStyle(model, {})
        view.setStyle({**model, "chain": binder_chain},
                      {"cartoon": {"color": BACKBONE_PALETTE[i % len(BACKBONE_PALETTE)]}})
    view.zoomTo()
    return view


# --- Checkpoint 3: monomer self-consistency --------------------------------


def _superpose_prediction(
    predicted_pdb: str | Path,
    design_pdb: str | Path,
    design_binder_chain: str = "A",
    predicted_chain: str | None = None,
) -> str:
    """Predicted monomer PDB text, rotated onto the design's binder chain.

    Fitted on the same paired backbone atoms `metrics.scrmsd` uses -- positional
    pairing, since a prediction is numbered from 1 while the design pose keeps
    its own numbering -- so the residual left on screen IS the reported scRMSD.
    """
    from . import metrics

    structure = pdb_io.read_pdb(predicted_pdb)
    predicted = (structure[predicted_chain] if predicted_chain
                 else structure.only_chain() if len(structure.chains) == 1
                 else structure[structure.chain_ids[0]])
    design = pdb_io.read_pdb(design_pdb)[design_binder_chain]

    _, rotation, translation = metrics.backbone_superposition(predicted, design)
    return _as_pdb_text(_transform_chain(predicted, rotation, translation))


def show_monomer_superposition(
    design_pdb: str | Path,
    predicted_pdb: str | Path,
    scrmsd: float | None = None,
    binder_chain: str = "A",
    predicted_chain: str | None = None,
    align: bool = True,
    width: int = 800,
    height: int = 520,
):
    """Design pose (cyan) against predicted monomer (magenta), superposed.

    The prediction is moved onto the design's binder chain before rendering.
    py3Dmol does not superpose, and the two files are in unrelated frames --
    RFdiffusion recentres its output while a predictor emits its own arbitrary
    frame -- so drawn as loaded they sit apart by whatever that offset happens
    to be and nothing about the fold can be read off the view.

    Superposing hides nothing: scRMSD is itself defined after optimal
    superposition, so the separation left on screen is the deviation the gate
    measures, residue by residue. The rigid-body offset it replaces is not a
    design observable at all.

    `binder_chain` is the binder INSIDE the design PDB, which also carries the
    target; only that chain is drawn, since this checkpoint is about the monomer.
    Pass `align=False` for the raw frames.
    """
    view = _view(width, height)
    view.addModel(_read(design_pdb), "pdb")
    # The design file holds the target too. Drawing it would dominate zoomTo
    # and bury the 87-residue binder next to a 150-residue target.
    view.setStyle({"model": 0}, {})
    view.setStyle({"model": 0, "chain": binder_chain},
                  {"cartoon": {"color": DESIGN_COLOUR}})

    caption = "design=cyan  predicted=magenta"
    predicted_text = _read(predicted_pdb)
    if align:
        try:
            predicted_text = _superpose_prediction(
                predicted_pdb, design_pdb, binder_chain, predicted_chain
            )
        except (ValueError, KeyError) as exc:
            # Loud, and said on the view as well: an unaligned pair reads as a
            # catastrophic prediction rather than as a failed fit.
            print(f"  note: could not superpose, view is UNALIGNED -- {exc}")
            caption = f"UNALIGNED (fit failed)   {caption}"

    view.addModel(predicted_text, "pdb")
    view.setStyle({"model": 1}, {"cartoon": {"color": PREDICTION_COLOUR}})

    if scrmsd is not None:
        caption = f"scRMSD {scrmsd:.2f} A   {caption}"
    view.addLabel(
        caption,
        {"fontSize": 12, "backgroundColor": "black", "backgroundOpacity": 0.6,
         "useScreen": True, "screenOffset": {"x": 10, "y": 10}},
    )
    view.zoomTo({"model": 1})
    return view


# --- Checkpoint 4: complex interface ---------------------------------------


def show_complex_interface(
    complex_pdb: str | Path,
    binder_chain: str = "B",
    target_chain: str = "A",
    colour_by_plddt: bool = True,
    width: int = 820,
    height: int = 560,
):
    """Binder-target complex, binder coloured by pLDDT from the B-factor column.

    pLDDT is read from the file rather than recomputed; py3Dmol's `cartoon:
    {colorscheme: ...}` maps the B-factor directly.
    """
    view = _view(width, height)
    view.addModel(_read(complex_pdb), "pdb")
    view.setStyle({"chain": target_chain}, {"cartoon": {"color": TARGET_COLOUR}})
    if colour_by_plddt:
        # Low pLDDT red, high blue -- the AlphaFold convention inverted onto
        # a continuous ramp over the 0-100 B-factor range.
        view.setStyle(
            {"chain": binder_chain},
            {"cartoon": {"colorscheme": {"prop": "b", "gradient": "roygb",
                                         "min": 50, "max": 90}}},
        )
    else:
        view.setStyle({"chain": binder_chain}, {"cartoon": {"color": PREDICTION_COLOUR}})
    view.addSurface("VDW", {"opacity": 0.5, "color": TARGET_COLOUR}, {"chain": target_chain})
    view.zoomTo()
    return view


# --- PyMOL script generation ----------------------------------------------


def write_pml_hotspots(
    target_pdb: str | Path, labels: list[str], out_path: str | Path, chain: str = "A"
) -> Path:
    """PyMOL scene for the hotspot checkpoint."""
    resi = _pymol_resi_selector(labels)
    script = f"""\
# Hotspot selection checkpoint
load {Path(target_pdb).resolve()}, target
hide everything
show surface, target
show cartoon, target
color {PYMOL_TARGET_COLOUR}, target
set transparency, 0.65, target
select hotspots, target and chain {chain} and resi {resi}
show sticks, hotspots
color {PYMOL_HOTSPOT_COLOUR}, hotspots
set label_size, 16
set label_color, white
label hotspots and name CA, "%s%s" % (resi, resn)
orient hotspots
zoom hotspots, 8
set ray_shadows, 0
"""
    return _write(script, out_path)


def write_pml_superposition(
    design_pdb: str | Path,
    predicted_pdb: str | Path,
    out_path: str | Path,
    binder_chain: str = "A",
) -> Path:
    """PyMOL scene for the monomer self-consistency checkpoint.

    `binder_chain` is the binder inside the DESIGN pdb, chain A per the .trb
    classification -- not the Step 4 complex layout `show_complex_interface`
    takes.

    Uses `align` so PyMOL reports its own RMSD in the log. That number is an
    independent check on pipeline.metrics -- it will differ slightly, because
    PyMOL's align does outlier rejection cycles by default while scRMSD uses
    every paired backbone atom.
    """
    script = f"""\
# Monomer self-consistency checkpoint (MANIFEST Section 4)
load {Path(design_pdb).resolve()}, design
load {Path(predicted_pdb).resolve()}, predicted
hide everything
show cartoon, design or predicted
color {PYMOL_DESIGN_COLOUR}, design
color {PYMOL_PREDICTION_COLOUR}, predicted
# cycles=0 disables outlier rejection, for comparability with scRMSD
align predicted, design and chain {binder_chain}, cycles=0
orient design
set ray_shadows, 0
print "PyMOL align RMSD above is an independent check on metrics.scrmsd"
"""
    return _write(script, out_path)


def write_pml_interface(
    complex_pdb: str | Path,
    out_path: str | Path,
    binder_chain: str = "B",
    target_chain: str = "A",
) -> Path:
    """PyMOL scene for the complex interface checkpoint.

    Shows polar contacts across the interface and colours the binder by the
    B-factor column (pLDDT), per MANIFEST Section 4.
    """
    script = f"""\
# Complex interface checkpoint (MANIFEST Section 4)
load {Path(complex_pdb).resolve()}, complex
hide everything
show cartoon, complex
color {PYMOL_TARGET_COLOUR}, chain {target_chain}
spectrum b, red_yellow_green_blue, chain {binder_chain}, minimum=50, maximum=90
select interface, byres (chain {binder_chain} within 5 of chain {target_chain})
select interface_target, byres (chain {target_chain} within 5 of chain {binder_chain})
show sticks, interface or interface_target
distance polar_contacts, interface, interface_target, mode=2
show surface, chain {target_chain}
set transparency, 0.5
orient interface
set ray_shadows, 0
print "Binder coloured by pLDDT (B-factor): red low, blue high"
"""
    return _write(script, out_path)


def _write(script: str, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(script)
    return out_path


def chain_summary(pdb: str | Path) -> str:
    """Chain composition, for captioning a view."""
    return pdb_io.describe(pdb_io.read_pdb(pdb))
