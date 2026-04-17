"""
ShapeletFitter and ShapeletPSFLibrary for PSF shapelet decomposition.

ShapeletFitter  — reads visit/band metadata, loads stamp files, fits each stamp,
                  and saves all bvec + sigma arrays to a single .npz.
ShapeletPSFLibrary — loads a compiled .npz, reconstructs galsim.Shapelet objects,
                     and computes a shapelet_score (non-Gaussian, non-atmospheric
                     fractional power).
"""

import numpy as np
import galsim
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
import pandas as pd

# Shapelet coefficient indices for non-Gaussian, non-atmospheric power.
# These exclude the Gaussian core (orders 0-1) and the atmospheric/optical
# radial terms, leaving coma, trefoil, and higher-order aberration modes.
NON_GAUSS_NON_ATMOSPHERE = list(range(6, 10)) + list(range(15, 24))


class ShapeletFitter:
    """Fit Shapelet PSFs for all stamps in a given band and save to a .npz library.

    Parameters
    ----------
    stamps_dir : str | Path
        Directory containing ``stamps_{visit}.npz`` files.
    moments_pq : str | Path
        Path to ``psf_moments_allbands.pq`` — used only for visit_id / band lookup.
    bmax : int
        Maximum shapelet order (default 6).
    pixel_scale : float
        Pixel scale in arcsec/pixel (default 0.2).
    stamp_type : str
        Either "image" or "psf" (for PIFF stamps).
    """

    def __init__(
        self,
        stamps_dir: str = 'data/stamps',
        moments_pq: str = 'data/psf_moments_allbands.pq',
        bmax: int = 6,
        pixel_scale: float = 0.2,
        stamp_type: str = "image",
    ):
        self.stamps_dir = Path(stamps_dir)
        self.bmax = bmax
        self.pixel_scale = pixel_scale
        self.stamp_type = stamp_type

        df = pd.read_parquet(moments_pq, columns=['visit_id', 'band'])
        self._visit_df = df.drop_duplicates('visit_id').set_index('visit_id')

    def get_visits_for_band(
        self,
        band: str,
        date_after: datetime | None = None,
        date_before: datetime | None = None,
    ) -> list[int]:
        """Return visit IDs for *band*, optionally restricted to a date window.

        The first 8 digits of a visit_id encode the observation date as YYYYMMDD.
        """
        df = self._visit_df[self._visit_df['band'] == band]
        visit_ids = df.index.tolist()

        if date_after is None and date_before is None:
            return visit_ids

        filtered = []
        for vid in visit_ids:
            vdate = datetime.strptime(str(vid)[:8], '%Y%m%d')
            if date_after is not None and vdate < date_after:
                continue
            if date_before is not None and vdate >= date_before:
                continue
            filtered.append(vid)
        return filtered

    def _fit_stamp(self, img_arr: np.ndarray):
        """Fit one 2-D stamp. Returns (bvec, sigma) or (None, None) on failure."""
        img = galsim.Image(img_arr.astype(np.float64), scale=self.pixel_scale)
        try:
            hsm = galsim.hsm.FindAdaptiveMom(img)
            shp = galsim.Shapelet.fit(
                hsm.moments_sigma * self.pixel_scale, self.bmax, img, normalization='sb'
            )
            return shp.bvec.copy(), float(hsm.moments_sigma * self.pixel_scale)
        except Exception:
            return None, None

    def fit_and_save(
        self,
        band: str,
        output_path: str | Path,
        date_after: datetime | None = None,
        date_before: datetime | None = None,
    ) -> Path:
        """Fit shapelets for every PSF stamp in *band* and write a .npz library.

        Saved arrays
        ------------
        bvec      : float64 (N, n_coeffs)  -- shapelet coefficient vectors
        sigma     : float64 (N,)           -- adaptive-moment sigma [pixels]
        visit     : int64   (N,)           -- visit ID for each PSF
        detector  : int32   (N,)           -- detector ID for each PSF
        bmax      : scalar int             -- max shapelet order used
        band      : scalar str             -- photometric band
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        visit_ids = self.get_visits_for_band(band, date_after, date_before)
        print(f"Band {band!r}: {len(visit_ids)} visits selected")

        bvec_list, sigma_list, visit_list, det_list = [], [], [], []
        n_failed = 0

        for vid in tqdm(visit_ids, desc=f'band={band}', unit='visit'):
            if self.stamp_type == "image":
                stamp_file = self.stamps_dir / f'stamps_{vid}.npz'
            elif self.stamp_type == "psf":
                stamp_file = self.stamps_dir / f'piff_stamps_{vid}.npz'

            if not stamp_file.exists():
                continue
            data = np.load(stamp_file)
            stamps = data['stamps']
            detectors = data['detector']

            for i in range(len(stamps)):
                bvec, sigma = self._fit_stamp(stamps[i])
                if bvec is None:
                    n_failed += 1
                    continue
                bvec_list.append(bvec)
                sigma_list.append(sigma)
                visit_list.append(vid)
                det_list.append(int(detectors[i]))

        print(f"  fitted={len(bvec_list)}  failed={n_failed}")

        np.savez_compressed(
            output_path,
            bvec=np.array(bvec_list, dtype=np.float64),
            sigma=np.array(sigma_list, dtype=np.float64),
            visit=np.array(visit_list, dtype=np.int64),
            detector=np.array(det_list, dtype=np.int32),
            bmax=np.array(self.bmax),
            band=np.array(band),
        )
        print(f"  saved -> {output_path}")
        return output_path


class ShapeletPSFLibrary:
    """Load a compiled shapelet bvec library and reconstruct GalSim PSF objects.

    Parameters
    ----------
    npz_path : str | Path
        Path to a ``.npz`` file produced by :class:`ShapeletFitter`.
    non_negative : bool
        If True, ``sample_psf`` resamples to avoid images with negative pixels.
    threshold : float
        Minimum pixel value threshold for ``non_negative`` rejection.

    Attributes
    ----------
    shapelet_score : ndarray, shape (N,)
        Non-Gaussian, non-atmospheric fractional power for each PSF:
        ``sum(bvec[non_atm_indices]**2) / sum(bvec**2)``.
    """

    def __init__(self, npz_path: str | Path, non_negative: bool = False, threshold=-1e-10):
        data = np.load(npz_path)
        self.bvec_all = data['bvec']
        self.sigma_all = data['sigma']
        self.bmax = int(data['bmax'])
        self.band = str(data['band'])
        self.visit = data['visit']
        self.detector = data['detector']
        self.npz_path = Path(npz_path)
        self.non_negative = non_negative
        self._n = len(self.sigma_all)
        self.threshold = threshold

        # Compute shapelet_score: non-Gaussian, non-atmospheric fractional power
        total_power = np.sum(self.bvec_all ** 2, axis=1)
        non_atm_power = np.sum(self.bvec_all[:, NON_GAUSS_NON_ATMOSPHERE] ** 2, axis=1)
        self.shapelet_score = np.where(total_power > 0, non_atm_power / total_power, 0.0)

        print(self._n)
        print(f"Loaded {self._n} PSFs  |  band={self.band}  bmax={self.bmax}  "
              f"n_coeffs={self.bvec_all.shape[1]}  from {self.npz_path.name}"
              f"  non_negative={self.non_negative}")

    def get_psf(self, idx: int) -> galsim.Shapelet:
        """Return the shapelet PSF at position *idx* as a ``galsim.Shapelet``."""
        return galsim.Shapelet(
            float(self.sigma_all[idx]),
            self.bmax,
            self.bvec_all[idx],
        )

    def sample_psf(self, rng: np.random.Generator | None = None) -> galsim.Shapelet:
        """Return a randomly sampled ``galsim.Shapelet`` from the library."""
        if rng is None:
            rng = np.random.default_rng()
        idx = int(rng.integers(0, self._n))
        psf = self.get_psf(idx)

        if self.non_negative:
            counter = 0
            psf_image = self.draw_psf(psf, n=32)
            while np.min(psf_image) < self.threshold and counter < 99:
                psf = self.get_psf(idx + counter)
                psf_image = self.draw_psf(psf, n=32)
                counter += 1

        return psf

    def draw_psf(
        self,
        psf: galsim.Shapelet,
        n: int = 64,
        pixel_scale: float = 0.2,
    ) -> np.ndarray | None:
        """Draw *psf* onto an ``n x n`` image and return the pixel array."""
        img = galsim.Image(n, n, scale=pixel_scale)
        psf.drawImage(image=img, method='sb')
        return img.array

    def __len__(self) -> int:
        return self._n

    def __repr__(self) -> str:
        return (f"ShapeletPSFLibrary(band={self.band!r}, n={self._n}, "
                f"bmax={self.bmax}, non_negative={self.non_negative}, "
                f"source={self.npz_path.name!r})")
