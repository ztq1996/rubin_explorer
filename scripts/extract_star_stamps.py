#!/usr/bin/env python
"""
Extract 32x32 PSF-star image stamps from Butler for shapelet decomposition.

For each visit, iterates over science rafts.  For each raft:
  1. Randomly select one detector within that raft.
  2. Load the full preliminary_visit_image (PVI) for that detector **once**.
  3. Randomly select up to n_per_raft stars whose centroids lie on that
     detector, and cut their stamps directly from the loaded array (numpy
     slice – no extra Butler calls).
  4. Normalise each stamp and mean-stack them.

This means exactly one Butler get() per raft per visit (~21 calls total),
regardless of n_per_raft, making large stacks essentially free vs. the old
per-star bbox approach.

Memory design
-------------
- spawn context: workers start fresh (~400 MB each), no fork COW blowup
- Butler deleted in main before pool starts (saves ~400 MB in parent)
- POLARS_MAX_THREADS=1: prevents per-worker thread-pool bloat
- maxtasksperchild=100: recycles workers to prevent slow heap growth
- Full detector array freed immediately after stamps are cut

Output per visit: data/stamps/stamps_{visit}.npz  containing:
  - stamps   : float32 array (N_rafts, 32, 32), mean-stacked, flux-normalised
  - cx, cy   : float32 arrays (N_rafts,), mean centroid of contributing stars
  - detector : int32 array (N_rafts,), detector used for the raft
  - raft     : object array (N_rafts,), e.g. 'R22'
  - visit    : int64 array (N_rafts,)
  - band     : str scalar (same for all stars in a visit)
  - n_stack  : int32 array (N_rafts,), number of stars averaged per raft

Usage
-----
  python extract_star_stamps.py                          # all visits, 1/raft
  python extract_star_stamps.py --n-per-raft 10         # stack 10/raft
  python extract_star_stamps.py --visit 2024021200001   # single visit
  python extract_star_stamps.py --visit-list visits.txt
  python extract_star_stamps.py --ncpu 4                # fewer workers
"""

# Must be set before polars is imported anywhere (including in workers).
import os
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import argparse
import gc
import multiprocessing
import multiprocessing.pool
import time
import numpy as np
import polars as pl
from tqdm import tqdm

import lsst.geom as geom
from lsst.daf.butler import Butler
from lsst.obs.lsst import LsstCam

REPO        = "dp2_prep"
COLLECTION  = "LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/stage2"
STAMP_SIZE  = 32
NCPU        = 8
N_PER_RAFT  = 1    # stars to sample and stack per raft

SCIENCE_RAFTS = [
    "R01", "R02", "R03",
    "R10", "R11", "R12", "R13", "R14",
    "R20", "R21", "R22", "R23", "R24",
    "R30", "R31", "R32", "R33", "R34",
    "R41", "R42", "R43",
]

# ── Worker-process globals (set once by _worker_init) ────────────────────────
_W_BUTLER      = None
_W_DET2RAFT    = None
_W_SCI_RAFTS   = None
_W_STAMP_SIZE  = None
_W_OUT_DIR     = None
_W_N_PER_RAFT  = None


def _worker_init(repo_, collection_, det2raft_, sci_rafts_, stamp_size_, out_dir_, n_per_raft_):
    """Runs once per worker process after spawn. Imports are fresh here."""
    global _W_BUTLER, _W_DET2RAFT, _W_SCI_RAFTS, _W_STAMP_SIZE, _W_OUT_DIR, _W_N_PER_RAFT
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    from lsst.daf.butler import Butler as _B
    _W_BUTLER      = _B(repo_, collections=collection_)
    _W_DET2RAFT    = det2raft_
    _W_SCI_RAFTS   = sci_rafts_
    _W_STAMP_SIZE  = stamp_size_
    _W_OUT_DIR     = out_dir_
    _W_N_PER_RAFT  = n_per_raft_


# ── Top-level worker (must be importable at module level for spawn) ──────────
def _worker(visit_band):
    """Process one visit: one full PVI load per raft, numpy-slice stamps."""
    import gc as _gc
    import numpy as _np
    import polars as _pl

    visit, band = visit_band
    out_path = os.path.join(_W_OUT_DIR, f"stamps_{visit}.npz")
    if os.path.exists(out_path):
        return visit, -1, None   # -1 = already done

    t_visit = time.perf_counter()
    timings = {}

    # 1. Load centroid parquet
    t0 = time.perf_counter()
    try:
        uri = _W_BUTLER.getURI("refit_psf_star", instrument="LSSTCam", visit=visit)
        psf_table = (
            _pl.scan_parquet(uri.geturl())
            .select(["slot_Centroid_x", "slot_Centroid_y", "detector"])
            .collect()
        )
    except Exception as exc:
        return visit, 0, f"parquet load failed: {exc}"
    timings["parquet"] = time.perf_counter() - t0

    # 2. Assign raft labels and filter finite centroids
    t0 = time.perf_counter()
    det_ids = psf_table["detector"].to_numpy()
    rafts   = _np.array([_W_DET2RAFT.get(int(d), "unknown") for d in det_ids])
    psf_table = psf_table.with_columns(_pl.Series("raft", rafts)).filter(
        _pl.col("slot_Centroid_x").is_finite() & _pl.col("slot_Centroid_y").is_finite()
    )
    del det_ids, rafts
    timings["sampling"] = time.perf_counter() - t0

    # 3. Per-raft: pick one detector, load full PVI once, slice N stamps
    half = _W_STAMP_SIZE // 2
    stamps_list  = []
    cx_list      = []
    cy_list      = []
    det_list     = []
    raft_list    = []
    n_stack_list = []
    pvi_times    = []

    rng = _np.random.default_rng(visit)   # reproducible per visit

    for raft in _W_SCI_RAFTS:
        raft_stars = psf_table.filter(_pl.col("raft") == raft)
        if len(raft_stars) == 0:
            continue

        # Randomly pick one detector within this raft
        dets_in_raft = raft_stars["detector"].unique().to_numpy()
        chosen_det   = int(rng.choice(dets_in_raft))

        # Load full PVI for that detector (one Butler call per raft)
        t0 = time.perf_counter()
        try:
            exp  = _W_BUTLER.get(
                "preliminary_visit_image",
                instrument="LSSTCam", visit=visit, detector=chosen_det,
            )
            arr  = exp.getMaskedImage().image.array   # (ny, nx) float32-ish
            xy0x = exp.getXY0().getX()
            xy0y = exp.getXY0().getY()
            del exp   # free exposure, keep only the array
        except Exception as exc:
            pvi_times.append(time.perf_counter() - t0)
            continue
        pvi_times.append(time.perf_counter() - t0)

        ny, nx = arr.shape

        # Stars on the chosen detector
        det_stars = raft_stars.filter(_pl.col("detector") == chosen_det)
        if len(det_stars) == 0:
            del arr
            continue

        # Randomly select up to N_PER_RAFT stars
        n_sample = min(_W_N_PER_RAFT, len(det_stars))
        idx      = rng.choice(len(det_stars), size=n_sample, replace=False)
        selected = det_stars[list(idx)]

        # Cut stamps from the already-loaded array
        raft_stamps = []
        raft_cx     = []
        raft_cy     = []
        for row in selected.iter_rows(named=True):
            cx = float(row["slot_Centroid_x"])
            cy = float(row["slot_Centroid_y"])
            ix = int(round(cx))
            iy = int(round(cy))

            # Convert to array indices (account for detector xy0 offset)
            row0 = iy - xy0y - half
            col0 = ix - xy0x - half

            if row0 < 0 or col0 < 0 or row0 + _W_STAMP_SIZE > ny or col0 + _W_STAMP_SIZE > nx:
                continue   # star too close to detector edge

            stamp = arr[row0:row0 + _W_STAMP_SIZE, col0:col0 + _W_STAMP_SIZE].copy()
            total = stamp.sum()
            if not _np.isfinite(total) or total <= 0:
                continue

            raft_stamps.append(stamp / total)
            raft_cx.append(cx)
            raft_cy.append(cy)

        del arr   # free detector array

        if not raft_stamps:
            continue

        # Mean-stack all stamps for this raft
        stacked = _np.mean(raft_stamps, axis=0).astype(_np.float32)
        stamps_list.append(stacked)
        cx_list.append(float(_np.mean(raft_cx)))
        cy_list.append(float(_np.mean(raft_cy)))
        det_list.append(chosen_det)
        raft_list.append(raft)
        n_stack_list.append(len(raft_stamps))

    timings["pvi_total"] = sum(pvi_times)
    timings["pvi_mean"]  = _np.mean(pvi_times) * 1e3 if pvi_times else 0.0  # ms

    if not stamps_list:
        return visit, 0, "all rafts failed"

    # 4. Save
    t0 = time.perf_counter()
    _np.savez_compressed(
        out_path,
        stamps   = _np.stack(stamps_list),
        cx       = _np.array(cx_list,      dtype=_np.float32),
        cy       = _np.array(cy_list,      dtype=_np.float32),
        detector = _np.array(det_list,     dtype=_np.int32),
        raft     = _np.array(raft_list),
        visit    = _np.full(len(cx_list), visit, dtype=_np.int64),
        band     = _np.array(band),
        n_stack  = _np.array(n_stack_list, dtype=_np.int32),
    )
    timings["save"] = time.perf_counter() - t0

    timings["total"] = time.perf_counter() - t_visit
    _gc.collect()
    return visit, len(stamps_list), timings


# ── Helpers ──────────────────────────────────────────────────────────────────

def build_visit_band_map(butler):
    dsrs = list(butler.registry.queryDatasets("refit_psf_star"))
    return {dsr.dataId["visit"]: dsr.dataId["band"] for dsr in dsrs}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract 32x32 PSF-star stamps from Butler")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--visit",      type=int, help="Single visit ID")
    grp.add_argument("--visit-list", type=str, help="File with one visit ID per line")
    parser.add_argument("--repo",        default=REPO)
    parser.add_argument("--collection",  default=COLLECTION)
    parser.add_argument("--stamp-size",  type=int, default=STAMP_SIZE)
    parser.add_argument("--n-per-raft",  type=int, default=N_PER_RAFT,
                        help="Stars to sample and stack per raft (default: %(default)s)")
    parser.add_argument("--out-dir",     default="data/stamps")
    parser.add_argument("--ncpu",        type=int, default=NCPU)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Build visit→band map, then DELETE the butler before spawning workers.
    print("Querying all refit_psf_star datasets…")
    t0 = time.perf_counter()
    butler = Butler(args.repo, collections=args.collection)
    visit_band_map = build_visit_band_map(butler)
    del butler
    gc.collect()
    print(f"Found {len(visit_band_map)} visits  ({time.perf_counter()-t0:.1f}s)")

    if args.visit:
        visits = [args.visit]
    elif args.visit_list:
        with open(args.visit_list) as f:
            visits = [int(l.strip()) for l in f if l.strip()]
    else:
        visits = sorted(visit_band_map.keys())

    tasks = [(v, visit_band_map.get(v, "unknown")) for v in visits]

    # Build det→raft in main (avoids re-importing LsstCam in every worker).
    _camera  = LsstCam.getCamera()
    det2raft = {det.getId(): det.getName().split("_")[0] for det in _camera}

    n_done = n_skip = n_fail = 0
    timing_acc = {"parquet": 0, "pvi_total": 0, "save": 0, "total": 0}

    print(
        f"Processing {len(tasks)} visit(s) with {args.ncpu} workers (spawn), "
        f"n_per_raft={args.n_per_raft}…"
    )

    ctx = multiprocessing.get_context("spawn")
    with multiprocessing.pool.Pool(
        processes=args.ncpu,
        context=ctx,
        initializer=_worker_init,
        initargs=(args.repo, args.collection, det2raft,
                  list(SCIENCE_RAFTS), args.stamp_size, args.out_dir, args.n_per_raft),
        maxtasksperchild=100,
    ) as pool:
        pbar = tqdm(total=len(tasks), desc="visits")
        for visit, result, info in pool.imap_unordered(_worker, tasks):
            if result == -1:
                n_skip += 1
            elif isinstance(info, str):
                n_fail += 1
                tqdm.write(f"  [error] visit {visit}: {info}")
            else:
                n_done += 1
                for k in timing_acc:
                    timing_acc[k] += info.get(k, 0)
                if n_done % 50 == 0:
                    nd = n_done
                    tqdm.write(
                        f"  [avg over {nd} visits]"
                        f"  total={timing_acc['total']/nd:.1f}s"
                        f"  parquet={timing_acc['parquet']/nd:.2f}s"
                        f"  pvi={timing_acc['pvi_total']/nd:.2f}s"
                        f"  save={timing_acc['save']/nd:.2f}s"
                    )
            pbar.set_postfix(done=n_done, skip=n_skip, fail=n_fail)
            pbar.update(1)
        pbar.close()

    gc.collect()
    if n_done:
        nd = n_done
        print(
            f"\nAverage timings over {nd} visits:"
            f"  total={timing_acc['total']/nd:.1f}s"
            f"  parquet={timing_acc['parquet']/nd:.2f}s"
            f"  pvi={timing_acc['pvi_total']/nd:.2f}s"
            f"  save={timing_acc['save']/nd:.2f}s"
        )
    print(f"Done.  extracted={n_done}  skipped={n_skip}  failed={n_fail}")


if __name__ == "__main__":
    main()
