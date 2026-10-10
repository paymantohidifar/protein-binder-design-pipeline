#!/usr/bin/env python
"""ESMFold batch driver for the Step 3A monomer gate.

fair-esm ships no batch CLI, so this is the thin wrapper the container needs.
It writes one PDB per sequence with pLDDT in the B-factor column -- the
convention pipeline.parsers reads -- plus a sidecar JSON carrying pTM, which
the B-factor column cannot hold.

Deliberately minimal: all filtering, metric computation and bookkeeping happen
on the host in pipeline/, so this does nothing but fold and record.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def read_fasta(path):
    records, name, parts = [], None, []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(parts)))
            name, parts = line[1:].split()[0], []
        else:
            parts.append(line)
    if name is not None:
        records.append((name, "".join(parts)))
    return records


def main():
    ap = argparse.ArgumentParser(description="Fold sequences with ESMFold")
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Axial attention chunk. Lower trades speed for VRAM; 128 keeps a "
             "100-residue binder well inside 16 GB.",
    )
    ap.add_argument("--max-tokens-per-batch", type=int, default=1024)
    args = ap.parse_args()

    import torch
    import esm

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_fasta(args.fasta)
    if not records:
        sys.exit(f"no sequences in {args.fasta}")
    print(f"[esmfold] {len(records)} sequence(s) from {args.fasta}", flush=True)

    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()
    model.set_chunk_size(args.chunk_size)
    print(f"[esmfold] model ready on {torch.cuda.get_device_name(0)}", flush=True)

    failures = 0
    for name, sequence in records:
        started = time.time()
        try:
            with torch.no_grad():
                output = model.infer(sequence)
            pdb_text = model.output_to_pdb(output)[0]
        except RuntimeError as exc:
            # OOM on one long sequence must not abort the batch; the host
            # treats a missing output as a parser-visible gap.
            failures += 1
            print(f"[esmfold] FAILED {name}: {exc}", file=sys.stderr, flush=True)
            torch.cuda.empty_cache()
            continue

        (out_dir / f"{name}.pdb").write_text(pdb_text)
        # mean_plddt is on 0-100 here; the host normalises either scale.
        metrics = {
            "name": name,
            "length": len(sequence),
            "ptm": float(output["ptm"]),
            "mean_plddt": float(output["mean_plddt"]),
            "seconds": round(time.time() - started, 2),
        }
        (out_dir / f"{name}.json").write_text(json.dumps(metrics, indent=2))
        print(
            f"[esmfold] {name} len={metrics['length']} "
            f"pLDDT={metrics['mean_plddt']:.1f} pTM={metrics['ptm']:.3f} "
            f"{metrics['seconds']}s",
            flush=True,
        )

    print(f"[esmfold] done: {len(records) - failures} ok, {failures} failed", flush=True)
    # Non-zero only if nothing succeeded -- a partial batch is still useful.
    sys.exit(1 if failures == len(records) else 0)


if __name__ == "__main__":
    main()
