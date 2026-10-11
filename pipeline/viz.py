"""In-notebook 3D rendering and PyMOL script generation (MANIFEST Section 4).

Two outputs for every checkpoint: a py3Dmol view that renders in the notebook
cell, and a .pml script for opening the same scene in a real PyMOL session.

Colour conventions: target light grey, hotspots dark red, design pose cyan, prediction magenta.
"""

from __future__ import annotations

from pathlib import Path

from . import pdb_io

TARGET_COLOUR = "lightgrey"
HOTSPOT_COLOUR = "darkred"
DESIGN_COLOUR = "cyan"
PREDICTION_COLOUR = "magenta"
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


def _resi_selector(labels: list[str]) -> str:
    """'A59','A83' -> '59,83' for py3Dmol's resi selector."""
    return ",".join(str(pdb_io.parse_label(lab)[1]) for lab in labels)


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
        # Opacity below ~0.7 makes the hotspot sticks hard to read through.
        view.addSurface("VDW", {"opacity": 0.65, "color": TARGET_COLOUR}, {"chain": chain})
    view.zoomTo()
    return view


# --- Checkpoint 2: backbone diversity --------------------------------------


def show_backbone_ensemble(
    design_pdbs: list[str | Path],
    target_pdb: str | Path | None = None,
    target_chain: str = "A",
    binder_chain: str = "B",
    width: int = 800,
    height: int = 560,
):
    """Target cartoon plus every generated binder backbone in its own colour.

    Reveals whether RFdiffusion found one binding mode or several. A single
    tight cluster across all designs usually means the hotspots over-constrained
    the site.
    """
    view = _view(width, height)
    if target_pdb:
        view.addModel(_read(target_pdb), "pdb")
        view.setStyle({"model": 0}, {"cartoon": {"color": TARGET_COLOUR}})
        offset = 1
    else:
        offset = 0

    for i, pdb in enumerate(design_pdbs):
        view.addModel(_read(pdb), "pdb")
        colour = BACKBONE_PALETTE[i % len(BACKBONE_PALETTE)]
        model = {"model": i + offset}
        if target_pdb:
            # The design file contains the target too; hide it so the single
            # reference copy is not drawn N times.
            view.setStyle({**model, "chain": target_chain}, {})
            view.setStyle({**model, "chain": binder_chain},
                          {"cartoon": {"color": colour}})
        else:
            view.setStyle(model, {"cartoon": {"color": colour}})
    view.zoomTo()
    return view


# --- Checkpoint 3: monomer self-consistency --------------------------------


def show_monomer_superposition(
    design_pdb: str | Path,
    predicted_pdb: str | Path,
    scrmsd: float | None = None,
    width: int = 800,
    height: int = 520,
):
    """Design pose (cyan) against predicted monomer (magenta).

    Both are drawn in their own frames: py3Dmol does not superpose, and
    pre-superposing here would hide exactly the disagreement scRMSD measures.
    The caption carries the number.
    """
    view = _view(width, height)
    view.addModel(_read(design_pdb), "pdb")
    view.setStyle({"model": 0}, {"cartoon": {"color": DESIGN_COLOUR}})
    view.addModel(_read(predicted_pdb), "pdb")
    view.setStyle({"model": 1}, {"cartoon": {"color": PREDICTION_COLOUR}})
    if scrmsd is not None:
        view.addLabel(
            f"scRMSD {scrmsd:.2f} A   design=cyan  predicted=magenta",
            {"fontSize": 12, "backgroundColor": "black", "backgroundOpacity": 0.6,
             "position": {"x": 0, "y": 0, "z": 0}},
        )
    view.zoomTo()
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
    resi = "+".join(str(pdb_io.parse_label(lab)[1]) for lab in labels)
    script = f"""\
# Hotspot selection checkpoint (MANIFEST Section 4)
load {Path(target_pdb).resolve()}, target
hide everything
show surface, target
color grey80, target
set transparency, 0.35, target
select hotspots, target and chain {chain} and resi {resi}
show sticks, hotspots
color darkred, hotspots
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
    binder_chain: str = "B",
) -> Path:
    """PyMOL scene for the monomer self-consistency checkpoint.

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
color cyan, design
color magenta, predicted
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
color grey80, chain {target_chain}
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
