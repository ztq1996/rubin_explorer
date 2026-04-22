"""
Build per-tract summary table for DP2 object catalog.

For each tract, saves:
  - Vertex coordinates (4 corners) for plotting
  - Center RA/Dec
  - Area in arcmin^2
  - Star and galaxy counts and densities
  - Magnitude limit per band (mag where galaxy median SNR ~ 5) using gaap1p0Flux
  - Median PSF size, e1, e2 per band from PSF moments

Uses column-level loading for memory efficiency.
Saves output to data/tract_summary.parquet
"""

import os
import time
import numpy as np
import lsst.daf.butler as dafButler
from astropy.table import Table

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
REPO = "dp2_prep"
COLLECTION = "LSSTCam/runs/DRP/DP2/v30_0_6/DM-53881/stage4"
SKYMAP = "lsst_cells_v2"
BANDS = ["u", "g", "r", "i", "z", "y"]
PIXEL_SCALE = 0.2  # arcsec/pixel for LSSTCam
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTFILE = os.path.join(OUTDIR, "tract_summary.parquet")

# Columns to load from the object table
LOAD_COLS = ["refExtendedness", "detect_isIsolated", "detect_isDeblendedModelSource"]
for b in BANDS:
    LOAD_COLS += [
        f"{b}_gaap1p0Flux", f"{b}_gaap1p0FluxErr",
        f"{b}_ixxPSF", f"{b}_iyyPSF", f"{b}_ixyPSF",
    ]


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def flux_to_mag(flux_nJy):
    with np.errstate(divide="ignore", invalid="ignore"):
        return -2.5 * np.log10(flux_nJy) + 31.4


def compute_tract_area_arcmin2(skymap, tract_id):
    """Compute approximate tract area from vertices."""
    verts = skymap[tract_id].getVertexList()
    ras = np.array([v.getRa().asDegrees() for v in verts])
    decs = np.array([v.getDec().asDegrees() for v in verts])

    # Handle RA wrapping
    if np.ptp(ras) > 180:
        ras = np.where(ras > 180, ras - 360, ras)

    ra_span = np.ptp(ras)
    dec_span = np.ptp(decs)
    mid_dec_rad = np.radians(np.mean(decs))
    area_deg2 = ra_span * np.cos(mid_dec_rad) * dec_span
    return area_deg2 * 3600.0


def compute_maglim(flux, flux_err, min_galaxies=50):
    """Find the magnitude at which median galaxy SNR crosses 5.

    Uses fine bins and linear interpolation between the last bin with
    median SNR >= 5 and the first bin with median SNR < 5.
    Returns NaN if data is insufficient.
    """
    ok = np.isfinite(flux) & np.isfinite(flux_err) & (flux > 0) & (flux_err > 0)
    if ok.sum() < min_galaxies:
        return np.nan

    mag = flux_to_mag(flux[ok])
    snr = flux[ok] / flux_err[ok]

    bin_edges = np.arange(18, 30.01, 0.25)
    bin_ctrs = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.digitize(mag, bin_edges)

    # Compute median SNR per bin
    med_snrs = np.full(len(bin_ctrs), np.nan)
    for i in range(1, len(bin_edges)):
        in_bin = bin_idx == i
        if in_bin.sum() > 10:
            med_snrs[i - 1] = np.median(snr[in_bin])

    # Find where median SNR crosses 5 (going faint)
    valid = np.isfinite(med_snrs)
    if valid.sum() < 2:
        return np.nan

    for i in range(len(med_snrs) - 1):
        if np.isfinite(med_snrs[i]) and med_snrs[i] >= 5.0:
            if np.isfinite(med_snrs[i + 1]) and med_snrs[i + 1] < 5.0:
                # Linear interpolation between bin centers
                frac = (5.0 - med_snrs[i]) / (med_snrs[i + 1] - med_snrs[i])
                return bin_ctrs[i] + frac * (bin_ctrs[i + 1] - bin_ctrs[i])
    return np.nan


def compute_psf_stats(ixx, iyy, ixy):
    """Compute median PSF trace radius (arcsec), e1, e2 from moments.

    Moments are in pixel^2. Converts to arcsec using PIXEL_SCALE.
    """
    ok = (
        np.isfinite(ixx) & np.isfinite(iyy) & np.isfinite(ixy)
        & (ixx > 0) & (iyy > 0)
    )
    if ok.sum() < 10:
        return np.nan, np.nan, np.nan

    ixx_ok = ixx[ok]
    iyy_ok = iyy[ok]
    ixy_ok = ixy[ok]
    trace = ixx_ok + iyy_ok
    # Trace radius in pixels, converted to arcsec
    size_arcsec = np.sqrt(trace / 2.0) * PIXEL_SCALE
    e1 = (ixx_ok - iyy_ok) / trace
    e2 = 2.0 * ixy_ok / trace
    return np.median(size_arcsec), np.median(e1), np.median(e2)


def process_tract(butler, skymap, tract_id):
    """Process one tract, return a dict of summary values."""
    row = {"tract": tract_id}

    # Vertices and center
    ti = skymap[tract_id]
    verts = ti.getVertexList()
    for j, v in enumerate(verts):
        row[f"ra_v{j}"] = v.getRa().asDegrees()
        row[f"dec_v{j}"] = v.getDec().asDegrees()
    ctr = ti.getCtrCoord()
    row["ra_ctr"] = ctr.getRa().asDegrees()
    row["dec_ctr"] = ctr.getDec().asDegrees()

    # Area
    area = compute_tract_area_arcmin2(skymap, tract_id)
    row["area_arcmin2"] = area

    # Load data (column-selected)
    tab = butler.get(
        "object", skymap=SKYMAP, tract=tract_id,
        parameters={"columns": LOAD_COLS},
    )

    # Quality cut
    good = np.array(tab["detect_isIsolated"]) | np.array(tab["detect_isDeblendedModelSource"])
    ext = np.array(tab["refExtendedness"], dtype=float)

    is_star = good & (ext == 0)
    is_gal = good & (ext == 1)

    row["n_objects"] = int(len(tab))
    row["n_stars"] = int(is_star.sum())
    row["n_galaxies"] = int(is_gal.sum())
    row["star_density"] = is_star.sum() / area if area > 0 else np.nan
    row["galaxy_density"] = is_gal.sum() / area if area > 0 else np.nan

    # Per-band: magnitude limit and PSF stats
    for b in BANDS:
        # Magnitude limit from galaxy gaap1p0 flux
        flux = np.array(tab[f"{b}_gaap1p0Flux"], dtype=float)
        flux_err = np.array(tab[f"{b}_gaap1p0FluxErr"], dtype=float)
        row[f"{b}_maglim"] = compute_maglim(flux[is_gal], flux_err[is_gal])

        # PSF stats (use all good objects, not just galaxies)
        ixx = np.array(tab[f"{b}_ixxPSF"], dtype=float)
        iyy = np.array(tab[f"{b}_iyyPSF"], dtype=float)
        ixy = np.array(tab[f"{b}_ixyPSF"], dtype=float)
        psf_size, psf_e1, psf_e2 = compute_psf_stats(ixx[good], iyy[good], ixy[good])
        row[f"{b}_psf_size"] = psf_size
        row[f"{b}_psf_e1"] = psf_e1
        row[f"{b}_psf_e2"] = psf_e2

    # Free memory
    del tab
    return row


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("Connecting to Butler...")
    butler = dafButler.Butler(REPO, collections=[COLLECTION])
    skymap = butler.get("skyMap", skymap=SKYMAP)

    # Get all tracts
    refs = list(butler.registry.queryDatasets("object"))
    tracts = sorted(set(r.dataId["tract"] for r in refs))
    print(f"Processing {len(tracts)} tracts")

    rows = []
    t0 = time.time()
    for i, tract_id in enumerate(tracts):
        try:
            row = process_tract(butler, skymap, tract_id)
            rows.append(row)
        except Exception as e:
            print(f"  [ERROR] tract {tract_id}: {e}")

        if (i + 1) % 50 == 0 or (i + 1) == len(tracts):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(tracts) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1}/{len(tracts)}] "
                f"{elapsed:.0f}s elapsed, {rate:.1f} tracts/s, ETA {eta:.0f}s"
            )

    # Build and save table
    result = Table(rows)
    result.write(OUTFILE, overwrite=True)
    print(f"\nSaved {len(result)} tracts to {OUTFILE}")
    print(f"Columns: {result.colnames}")


if __name__ == "__main__":
    main()
