#!/usr/bin/env python
"""
Extract native-size Piff PSF model images per raft per visit from Butler.

For each visit, samples up to N_PER_RAFT stars per raft from `refit_psf_star`
and evaluates the `piff_psf` model at each star's centroid.  When N_PER_RAFT > 1
the rendered PSF images from each raft are mean-stacked so the saved output
always has one image per raft (same file format as the single-star case).

Differences from extract_star_stamps.py
----------------------------------------
- Reads `refit_psf_models_detector` (ExposureCatalog, one per detector) instead of
  bbox-cutting `preliminary_visit_image`; extracts the PiffPsf via rec.getPsf().
- Calls psf.computeKernelImage(position) to render the PSF kernel at a star's
  centroid, then crops / zero-pads to STAMP_SIZE × STAMP_SIZE.
- No image flux normalisation needed (PSF models are already normalised to unit sum);
  any PSF image with non-positive sum is discarded.

Memory design
-------------
- spawn context: workers start fresh (~400 MB each), no fork COW blowup
- Butler deleted in main before pool starts (saves ~400 MB in parent)
- POLARS_MAX_THREADS=1: prevents per-worker thread-pool bloat
- maxtasksperchild=100: recycles workers to prevent slow heap growth

Output per visit: data/piff_stamps/piff_stamps_{visit}.npz  containing:
  - stamps   : float32 array (N_rafts, H, W), mean-stacked at native PSF size (typically 27x27)
  - cx, cy   : float32 arrays (N_rafts,), mean centroid of contributing stars
  - detector : int32 array (N_rafts,), detector of the first contributing star
  - raft     : object array (N_rafts,), e.g. 'R22'
  - visit    : int64 array (N_rafts,)
  - band     : str scalar (same for all stars in a visit)
  - n_stack  : int32 array (N_rafts,), number of PSF evals averaged per raft

Usage
-----
  python extract_piff_stamps.py                          # all visits, 1/raft
  python extract_piff_stamps.py --n-per-raft 10         # stack 10/raft
  python extract_piff_stamps.py --visit 2024021200001   # single visit
  python extract_piff_stamps.py --visit-list visits.txt
  python extract_piff_stamps.py --ncpu 4                # fewer workers
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
N_PER_RAFT  = 1    # stars to evaluate and stack per raft (1 = original behaviour)

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


def _crop_or_pad(arr, size):
    """Centre-crop or zero-pad a 2-D array to (size, size)."""
    h, w = arr.shape
    if h == size and w == size:
        return arr
    out = np.zeros((size, size), dtype=arr.dtype)
    # crop extents in source
    sy = max(0, (h - size) // 2)
    sx = max(0, (w - size) // 2)
    ch = min(h - sy, size)
    cw = min(w - sx, size)
    # paste extents in destination
    dy = (size - ch) // 2
    dx = (size - cw) // 2
    out[dy:dy + ch, dx:dx + cw] = arr[sy:sy + ch, sx:sx + cw]
    return out


# ── Top-level worker (must be importable at module level for spawn) ──────────
def _worker(visit_band):
    """Process one visit: sample N stars/raft, render Piff PSF, save .npz."""
    import gc as _gc
    import numpy as _np
    import polars as _pl
    import lsst.geom as _geom

    visit, band = visit_band
    out_path = os.path.join(_W_OUT_DIR, f"piff_stamps_{visit}.npz")
    if os.path.exists(out_path):
        return visit, -1, None   # -1 = already done

    t_visit = time.perf_counter()
    timings = {}

    # 1. load centroid parquet (same as extract_star_stamps.py)
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

    # 2. sample up to N_PER_RAFT stars per raft
    t0 = time.perf_counter()
    det_ids = psf_table["detector"].to_numpy()
    rafts   = _np.array([_W_DET2RAFT.get(int(d), "unknown") for d in det_ids])
    psf_table = psf_table.with_columns(_pl.Series("raft", rafts))

    rng  = _np.random.default_rng(visit)   # reproducible per visit
    sampled_rows = []   # list of (raft, rows_df)
    for raft in _W_SCI_RAFTS:
        sub = psf_table.filter(_pl.col("raft") == raft)
        sub = sub.filter(
            _pl.col("slot_Centroid_x").is_finite() & _pl.col("slot_Centroid_y").is_finite()
        )
        if len(sub) == 0:
            continue
        n_draw = min(_W_N_PER_RAFT, len(sub))
        idx    = rng.choice(len(sub), size=n_draw, replace=False)
        sampled_rows.append((raft, sub[idx.tolist()]))
    del psf_table, det_ids, rafts
    timings["sampling"] = time.perf_counter() - t0

    if not sampled_rows:
        return visit, 0, "no stars after raft sampling"

    # 3. render Piff PSF at each star's centroid, stack per raft
    half      = _W_STAMP_SIZE // 2
    psf_times = []

    stamps_list  = []
    cx_list      = []
    cy_list      = []
    det_list     = []
    raft_list    = []
    nstack_list  = []

    first_exc = None   # capture the first failure for diagnostics

    for raft, sub_df in sampled_rows:
        raft_imgs  = []
        raft_cx    = []
        raft_cy    = []
        first_det  = None

        for row in sub_df.iter_rows(named=True):
            detector = int(row["detector"])
            cx       = float(row["slot_Centroid_x"])
            cy       = float(row["slot_Centroid_y"])

            t0 = time.perf_counter()
            try:
                psf_cat = _W_BUTLER.get(
                    "refit_psf_models_detector",
                    instrument="LSSTCam", visit=visit, detector=detector,
                )
                piff_psf = psf_cat[0].getPsf()
                position = _geom.Point2D(cx, cy)
                psf_img  = piff_psf.computeKernelImage(position)
                arr      = psf_img.array.copy()
                del psf_cat, piff_psf, psf_img
            except Exception as exc:
                if first_exc is None:
                    first_exc = f"det={detector}: {type(exc).__name__}: {exc}"
                psf_times.append(time.perf_counter() - t0)
                continue
            psf_times.append(time.perf_counter() - t0)

            arr = arr.astype(_np.float32)
            total = arr.sum()
            if not _np.isfinite(total) or total <= 0:
                continue

            raft_imgs.append(arr)
            raft_cx.append(cx)
            raft_cy.append(cy)
            if first_det is None:
                first_det = detector

        if not raft_imgs:
            continue

        stacked = _np.mean(raft_imgs, axis=0).astype(_np.float32)
        stamps_list.append(stacked)
        cx_list.append(float(_np.mean(raft_cx)))
        cy_list.append(float(_np.mean(raft_cy)))
        det_list.append(first_det)
        raft_list.append(raft)
        nstack_list.append(len(raft_imgs))

    timings["psf_total"] = sum(psf_times)
    timings["psf_mean"]  = _np.mean(psf_times) * 1e3 if psf_times else 0.0  # ms

    if not stamps_list:
        reason = f"all PSF renders failed; first error: {first_exc}"
        return visit, 0, reason

    # 4. save
    t0 = time.perf_counter()
    _np.savez_compressed(
        out_path,
        stamps   = _np.stack(stamps_list),
        cx       = _np.array(cx_list,     dtype=_np.float32),
        cy       = _np.array(cy_list,     dtype=_np.float32),
        detector = _np.array(det_list,    dtype=_np.int32),
        raft     = _np.array(raft_list),
        visit    = _np.full(len(cx_list), visit, dtype=_np.int64),
        band     = _np.array(band),
        n_stack  = _np.array(nstack_list, dtype=_np.int32),
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
    parser = argparse.ArgumentParser(
        description="Extract 32x32 Piff PSF model images per raft per visit"
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--visit",      type=int, help="Single visit ID")
    grp.add_argument("--visit-list", type=str, help="File with one visit ID per line")
    parser.add_argument("--repo",        default=REPO)
    parser.add_argument("--collection",  default=COLLECTION)
    parser.add_argument("--stamp-size",  type=int, default=STAMP_SIZE)
    parser.add_argument("--n-per-raft",  type=int, default=N_PER_RAFT,
                        help="Stars to render & stack per raft (default: %(default)s)")
    parser.add_argument("--out-dir",     default="data/piff_stamps")
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
    timing_acc = {"parquet": 0, "sampling": 0, "psf_total": 0, "save": 0, "total": 0}

    print(
        f"Processing {len(tasks)} visit(s) with {args.ncpu} workers (spawn),"
        f" n_per_raft={args.n_per_raft}…"
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
            elif isinstance(info, str):   # error string
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
                        f"  psf={timing_acc['psf_total']/nd:.2f}s"
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
            f"  psf={timing_acc['psf_total']/nd:.2f}s"
            f"  save={timing_acc['save']/nd:.2f}s"
        )
    print(f"Done.  extracted={n_done}  skipped={n_skip}  failed={n_fail}")


if __name__ == "__main__":
    main()
