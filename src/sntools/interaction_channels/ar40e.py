"""
Charged-current electron neutrino interaction on Argon-40:
    nu_e + 40Ar -> e- + 40K*
Implemented using a continuous, normalization-preserving PCHIP reconstruction
of MARLEY Monte Carlo event distributions and total cross sections.
"""

from importlib.resources import files
import random
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from sntools.event import Event
from sntools.interaction_channels import BaseChannel, emass_threshold

# List of neutrino flavors ("e", "eb", "x", "xb") that interact in this channel.
possible_flavors = ['e']

# Conversion factor from 10^-42 cm^2 (MARLEY standard) to MeV^-2 (sntools standard)
# 1 cm = 5.067730716e10 MeV^-1  =>  10^-42 cm^2 = 1e-42 * (5.067730716e10)^2 MeV^-2
_CM2_1E42_TO_MEVM2 = 1e-42 * (5.067731e10 ** 2)

# Minimum physical invisible energy loss (Q-value ~ 1.505 MeV for 40Ar -> 40K g.s. + e-)
# Used to tighten the upper kinematic bound and guide adaptive quadrature around the peak.
_EINVIS_MIN_CUT = 1.40  # MeV
_EINVIS_PEAK_MAX = 1.50  # MeV
_EINVIS_NEUTRON_THRESHOLD = 9.30 # MeV
_EINVIS_NEUTRON_PEAK = 15.0 # MeV

class _MarleyEvaluator:
    """
    Loads the edge-aligned MARLEY ASCII tables and provides evaluation of
    dSigma/dE(eNu, eE), total_sigma(eNu), and exact inverse-CDF sampling of eE.
    """
    def __init__(self, cdf_filename: str, xsec_filename: str):
        data_dir = files("sntools.interaction_channels.data")
        raw_table = np.loadtxt(data_dir.joinpath(cdf_filename))
        xsec_data = np.loadtxt(data_dir.joinpath(xsec_filename))

        self.Einvis_edges = raw_table[0, 1:]
        self.Enu_edges = raw_table[1:, 0]
        self.cdf_grid = raw_table[1:, 1:]

        # Strict fail-fast validation of the CDF table
        assert np.all(np.diff(self.Einvis_edges) > 0.0), "Einvis_edges must be strictly increasing."
        assert np.all(np.diff(self.Enu_edges) > 0.0), "Enu_edges must be strictly increasing."
        assert np.allclose(self.cdf_grid[:, 0], 0.0, atol=1e-12), "CDF must start at 0.0."
        assert np.allclose(self.cdf_grid[:, -1], 1.0, atol=1e-7), "CDF must end at 1.0."
        assert np.all(np.diff(self.cdf_grid, axis=1) >= -1e-12), "CDF must be monotonically non-decreasing."

        xsec_Enu = xsec_data[:, 0]
        xsec_val = xsec_data[:, 1] * _CM2_1E42_TO_MEVM2
        assert np.all(np.diff(xsec_Enu) > 0.0), "xsec_Enu must be strictly increasing."

        positive_mask = xsec_val > 0.0
        assert np.any(positive_mask), "Cross section table contains no positive values."
        xsec_thr = float(xsec_Enu[positive_mask][0])

        self.E_thr = max(float(self.Enu_edges[0]), xsec_thr, _EINVIS_MIN_CUT)
        self.E_max = float(min(self.Enu_edges[-1], xsec_Enu[-1]))

        # Verify compatibility with sntools global upper energy limit
        assert self.E_max >= BaseChannel.E_NU_MAX, (
            f"Evaluator E_max ({self.E_max} MeV) is below BaseChannel.E_NU_MAX ({BaseChannel.E_NU_MAX} MeV)."
        )

        self.sigma_interp = PchipInterpolator(xsec_Enu, xsec_val, extrapolate=False)

        # Build both the 1D PCHIP CDF splines (for exact inverse-CDF sampling in get_eE)
        # and their analytical derivatives (for dSigma_dE) at each E_nu edge.
        self.cdf_splines = []
        self.pdf_splines = []
        for i in range(len(self.Enu_edges)):
            # Clean tiny numerical rounding before building the monotone spline
            row_cdf = np.maximum.accumulate(np.clip(self.cdf_grid[i, :], 0.0, 1.0))
            row_cdf[0] = 0.0
            row_cdf[-1] = 1.0
            c_spline = PchipInterpolator(self.Einvis_edges, row_cdf, extrapolate=False)
            self.cdf_splines.append(c_spline)
            self.pdf_splines.append(c_spline.derivative())

    def _bracketing_Enu(self, eNu: float):
        """Return adjacent E_nu edge index k and linear weight alpha in [0, 1]."""
        if eNu < self.E_thr or eNu > self.E_max:
            raise ValueError(f"Neutrino energy eNu={eNu} MeV is outside valid range [{self.E_thr}, {self.E_max}].")
        idx = int(np.searchsorted(self.Enu_edges, eNu, side="right") - 1)
        if idx == len(self.Enu_edges) - 1:
            idx -= 1
        assert 0 <= idx < len(self.Enu_edges) - 1
        E0 = self.Enu_edges[idx]
        E1 = self.Enu_edges[idx + 1]
        alpha = float((eNu - E0) / (E1 - E0))
        return idx, alpha

    def total_sigma(self, eNu: float) -> float:
        """Return total cross section in MeV^-2 at neutrino energy eNu."""
        if eNu < self.E_thr:
            return 0.0
        if eNu > self.E_max:
            raise ValueError(f"eNu={eNu} MeV exceeds maximum tabulated energy E_max={self.E_max} MeV.")
        sig = float(self.sigma_interp(eNu))
        if not np.isfinite(sig) or sig < 0.0:
            raise RuntimeError(f"Invalid interpolated total cross section {sig} at eNu={eNu} MeV.")
        return sig

    def dSigma_dE(self, eNu: float, eE):
        """
        Evaluate differential cross section dsigma/deE(eNu, eE) in MeV^-3,
        where eE = E_vis + emass_threshold is the total energy of the outgoing electron.
        """
        if eNu < self.E_thr:
            return np.zeros_like(eE, dtype=float) if np.ndim(eE) > 0 else 0.0

        sig = self.total_sigma(eNu)
        idx, alpha = self._bracketing_Enu(eNu)

        eE_arr = np.asarray(eE, dtype=float)
        E_vis = eE_arr - emass_threshold
        E_invis = eNu - E_vis

        in_domain = (E_vis >= 0) & (E_invis >= _EINVIS_MIN_CUT) & (E_invis <= self.Einvis_edges[-1])
        safe_Einvis = np.where(in_domain, E_invis, _EINVIS_MIN_CUT)

        p0 = self.pdf_splines[idx](safe_Einvis)
        p1 = self.pdf_splines[idx + 1](safe_Einvis)
        pdf_vals = (1.0 - alpha) * p0 + alpha * p1

        # Clean floating-point roundoff (~ -1e-18) while failing on real negative values
        if np.any(pdf_vals < -1e-10) or not np.all(np.isfinite(pdf_vals)):
            raise RuntimeError(f"Invalid PDF evaluation at eNu={eNu}: min={np.min(pdf_vals)}")
        pdf_vals = np.maximum(0.0, pdf_vals)

        res = np.where(in_domain, sig * pdf_vals, 0.0)
        return res if np.ndim(eE) > 0 else float(res)

    def sample_eE(self, eNu: float) -> float:
        """
        Sample outgoing total electron energy eE from the conditional distribution
        P(eE | eNu) using exact inverse transform sampling on the PCHIP CDF.
        """
        idx, alpha = self._bracketing_Enu(eNu)

        # Maximum allowed E_invis such that eE = emass_threshold + eNu - E_invis >= emass_threshold
        Einvis_max = eNu
        Einvis_min = _EINVIS_MIN_CUT
        if Einvis_max <= Einvis_min:
            raise ValueError(f"No kinematic phase space available at eNu={eNu} MeV.")

        # Evaluate CDF at the kinematic bounds
        def eval_cdf(x: float) -> float:
            if x <= self.Einvis_edges[0]:
                return 0.0
            if x >= self.Einvis_edges[-1]:
                return 1.0
            return float((1.0 - alpha) * self.cdf_splines[idx](x) + alpha * self.cdf_splines[idx + 1](x))

        cdf_min = eval_cdf(Einvis_min)
        cdf_max = eval_cdf(Einvis_max)
        if cdf_max <= cdf_min:
            raise RuntimeError(f"Zero probability in kinematic range [{Einvis_min}, {Einvis_max}] at eNu={eNu} MeV.")

        # Draw uniform random target CDF value in [cdf_min, cdf_max]
        u = random.uniform(cdf_min, cdf_max)

        # Fast binary search on the discrete grid to bracket the 0.2 MeV bin containing u
        cdf_row_discrete = (1.0 - alpha) * self.cdf_grid[idx, :] + alpha * self.cdf_grid[idx + 1, :]
        j = int(np.searchsorted(cdf_row_discrete, u, side="right") - 1)
        j = int(np.clip(j, 0, len(self.Einvis_edges) - 2))
    
        x_left = float(self.Einvis_edges[j])
        x_right = float(self.Einvis_edges[j + 1])
        c_left = float(cdf_row_discrete[j])
        c_right = float(cdf_row_discrete[j + 1])

        assert c_left <= u <= c_right, (
            f"Binary search invariant violated: u={u} not in [{c_left}, {c_right}] at bin j={j}."
        )
        assert c_right > c_left, (
            f"Zero-probability plateau bin selected at j={j} ([{x_left}, {x_right}] MeV) for u={u}."
        )
        
        Einvis_sampled = float(brentq(lambda x: eval_cdf(x) - u, x_left, x_right, xtol=1e-3))
        
        if not (x_left <= Einvis_sampled <= x_right):
            raise RuntimeError(
                f"Newton step left bin [{x_left}, {x_right}]: got {Einvis_sampled} MeV."
            )
        if not (Einvis_min <= Einvis_sampled <= Einvis_max):
            raise RuntimeError(
                f"Sampled Einvis={Einvis_sampled} MeV outside kinematic bounds [{Einvis_min}, {Einvis_max}]."
            )
            
        eE_sampled = emass_threshold + eNu - Einvis_sampled
        return eE_sampled


# Instantiate once when the module is imported
_EVALUATOR = _MarleyEvaluator("ar40e_marley_cdf.dat", "ar40e_marley_xsec.txt")
e_thr = _EVALUATOR.E_thr

class Channel(BaseChannel):
    """Charged-current nu_e + 40Ar -> e- + 40K* interaction channel."""
    
    def generate_event(self, eNu, dirx, diry, dirz):
        """Return an event with the appropriate incoming/outgoing particles.

        Input:
            eNu: neutrino energy
            dirx, diry, dirz: direction of outgoing particle (normalized to 1)
        """
        eE = self.get_eE(eNu, dirz)
        evt = Event(1018040)
        evt.incoming_particles.append([12, eNu, 0, 0, 1])  # incoming nu_e
        evt.incoming_particles.append((18040, 37225, 0, 0, 1))  # argon nucleus at rest
        evt.outgoing_particles.append([11, eE, dirx, diry, dirz])  # outgoing electron
        return evt
    
    # List with minimum & maximum energy of incoming neutrino.
    bounds_eNu = [e_thr, BaseChannel.E_NU_MAX]

    def dSigma_dE(self, eNu, eE):
        """Return differential cross-section in MeV^-3."""
        return _EVALUATOR.dSigma_dE(eNu, eE)
    
    def total_sigma(self, eNu):
        """Return total cross-section in MeV^-2."""
        return _EVALUATOR.total_sigma(eNu)

    def bounds_eE(self, eNu, *args):
        """
        Return kinematic bounds [eE_min, eE_max] for the total energy
        eE = E_vis + emass_threshold of the effective outgoing electron (in MeV).
        """
        eMin = emass_threshold
        eMax = max(eMin, emass_threshold + eNu - _EINVIS_MIN_CUT)
        return [eMin, eMax]

    def get_eE(self, eNu, cosT=0):
        """
        Return total energy eE (in MeV) of the outgoing effective electron.
        Samples eE directly from the continuous MARLEY conditional distribution
        P(eE | eNu) via inverse transform sampling (generalizing o16e.py to the continuum).
        """
        return _EVALUATOR.sample_eE(eNu)

    def _opts(self, eNu, *args):
        """
        Provide break points around the narrow ground/bound-state absorption peak
        (where E_invis ~ 1.5 MeV) so scipy.integrate.quad resolves it with machine precision.
        """
        eMin, eMax = self.bounds_eE(eNu)
        p_peak = emass_threshold + eNu - _EINVIS_PEAK_MAX
        p_neutron_thr = emass_threshold + eNu - _EINVIS_NEUTRON_THRESHOLD
        p_neutron_peak = emass_threshold + eNu - _EINVIS_NEUTRON_PEAK
        pts = [p for p in (p_peak, p_neutron_thr, p_neutron_peak) if eMin < p < eMax]
        return {"points": pts}
    
    def dSigma_dCosT(self, eNu, cosT):
        """Return differential cross section in MeV^-2 as a function of the emission angle of the outgoing (detected) particle.

        Input:
            eNu:  neutrino energy (MeV)
            cosT: cosine of the angle between neutrino and outgoing (detected) particle
        """
        # Energy dependence is unclear, so we use a constant value for now.
        if abs(cosT) > 1:
            return 0
        # isotropic in cos(theta)
        return 0.5
        #Should be: return 0.5 * _EVALUATOR.total_sigma(eNu)
        #But it doesn't make an actual difference, so we omit it for efficiency.
    isotropic = True # The channel is approximated as isotropic.
    
    def _bounds_eNu(self, eE):
        """Min/max neutrino energy that can produce a given positron energy."""
        return (eE + _EINVIS_MIN_CUT - emass_threshold, self.bounds_eNu[1])