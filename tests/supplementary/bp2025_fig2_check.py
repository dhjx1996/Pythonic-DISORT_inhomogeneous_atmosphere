"""Fact-check BP2025 Fig. 2: is the median VOCALS-REx profile non-adiabatic near
cloud top, contrary to the paper's "closely resemble the adiabatic profile"?

Reproduces their construction (over 100 non-precipitating profiles, vertical
dimension normalized, 30 bins, median per bin) from OUR CDP-derived profiles, and
quantifies the deviation from an adiabatic reference in the top decile z in [0.91, 1].

Adiabatic references (N_c const, LWC linear in height, r_e^3 linear in height):
  - PLOT (BP2025-style): anchored at the boundary VALUES of the median profile.
  - DEVIATION TEST: adiabatic SHAPE fit to the lower/mid cloud z in [0.10, 0.75]
    (where adiabatic holds) and EXTRAPOLATED to the top -> tests a *systematic*
    near-top departure, not a boundary-anchoring artifact.

    /tmp/jaxve/bin/python tests/supplementary/bp2025_fig2_check.py
"""
import sys
from pathlib import Path

import numpy as np

_src = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(_src))
import vocals_io as vio  # noqa: E402

DATA = ('/home/jovyan/cloud_profile_retrieval/'
        'multispectral-retrieval-using-MODIS/VOCALS_REx_data')
NBIN = 30
zc = (np.arange(NBIN) + 0.5) / NBIN                       # bin centers in (0,1)
TOP = zc >= 0.91                                          # top decile (user's region)
FIT = (zc >= 0.10) & (zc <= 0.75)                         # adiabatic-holds region


def main():
    profiles = vio.load_all_profiles(DATA)
    # Per-profile: normalize altitude (0=base, 1=top) and interpolate each quantity
    # onto the 30 bin centers, so every profile gets equal weight (BP2025 normalize-
    # then-median). Profiles are monotone traverses ordered top->base.
    R, L, NC = [], [], []
    used = 0
    for p in profiles:
        a = np.asarray(p.altitude, float)
        if a.max() - a.min() < 30.0:
            continue
        z = (a - a.min()) / (a.max() - a.min())            # 0 base ->1 top
        o = np.argsort(z)
        z = z[o]
        R.append(np.interp(zc, z, np.asarray(p.r_e)[o]))
        L.append(np.interp(zc, z, np.asarray(p.lwc)[o]))
        NC.append(np.interp(zc, z, np.asarray(p.total_Nc)[o]))
        used += 1
    R, L, NC = map(np.array, (R, L, NC))
    print(f"profiles used: {used} of {len(profiles)} "
          f"(flights {sorted({p.flight for p in profiles})})")

    med = {'r_e': np.median(R, 0), 'LWC': np.median(L, 0), 'N_c': np.median(NC, 0)}

    # --- adiabatic SHAPE fit to z in [0.10,0.75], extrapolated to top -----------
    def lin_fit(y):                                         # y(z) = m z + b
        return np.polyfit(zc[FIT], y[FIT], 1)

    ad_extrap = {}
    m, b = lin_fit(med['LWC']);            ad_extrap['LWC'] = m * zc + b          # linear
    ncbar = med['N_c'][FIT].mean();        ad_extrap['N_c'] = np.full(NBIN, ncbar)  # const
    m3, b3 = lin_fit(med['r_e'] ** 3);     ad_extrap['r_e'] = np.maximum(m3 * zc + b3, 0) ** (1 / 3.)  # r_e^3 linear

    print("\nNear-top deviation of the MEDIAN from the adiabatic shape "
          "(fit to z in [0.10,0.75], extrapolated), averaged over z in [0.91,1]:")
    print(f"  {'quantity':6} {'median_top':>11} {'adiab_top':>10} "
          f"{'abs_dev':>9} {'rel_dev':>8}")
    rel = {}
    for k in ('r_e', 'LWC', 'N_c'):
        mt = med[k][TOP].mean(); at = ad_extrap[k][TOP].mean()
        rel[k] = (mt - at) / at
        unit = 'um' if k == 'r_e' else ('g/m3' if k == 'LWC' else '1/cm3')
        print(f"  {k:6} {mt:11.3f} {at:10.3f} {mt-at:+9.3f} "
              f"{100*rel[k]:+7.1f}%   ({unit})")

    print("\nUser's claim — near-top departure more prominent in LWC & N_c than r_e:")
    print(f"  |rel_dev|:  r_e {abs(rel['r_e'])*100:.1f}%   "
          f"LWC {abs(rel['LWC'])*100:.1f}%   N_c {abs(rel['N_c'])*100:.1f}%")
    verdict = (abs(rel['LWC']) > abs(rel['r_e'])) and (abs(rel['N_c']) > abs(rel['r_e']))
    print(f"  -> LWC & N_c deviate MORE than r_e: {verdict}")

    # --- BP2025-style boundary-anchored adiabatic (for the plot) ----------------
    z = zc
    ad_anchor = {
        'LWC': med['LWC'][0] + (med['LWC'][-1] - med['LWC'][0]) * z,
        'N_c': np.full(NBIN, np.median(med['N_c'])),
        'r_e': (med['r_e'][0] ** 3 + (med['r_e'][-1] ** 3 - med['r_e'][0] ** 3) * z) ** (1 / 3.),
    }

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        mad = {k: np.median(np.abs(v - np.median(v, 0)), 0)
               for k, v in (('r_e', R), ('LWC', L), ('N_c', NC))}
        fig, ax = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)
        for j, (k, xl) in enumerate((('r_e', r'$r_e$ [$\mu$m]'),
                                     ('LWC', 'LWC [g m$^{-3}$]'),
                                     ('N_c', r'$N_c$ [cm$^{-3}$]'))):
            ax[j].fill_betweenx(zc, med[k]-mad[k], med[k]+mad[k], color='g', alpha=.18)
            ax[j].plot(med[k], zc, 'g-', lw=2, label='median')
            ax[j].plot(ad_anchor[k], zc, 'k--', lw=1.3, label='adiabatic (anchored)')
            ax[j].plot(ad_extrap[k], zc, 'r:', lw=1.5, label='adiabatic (lower-fit extrap.)')
            ax[j].axhspan(0.91, 1.0, color='orange', alpha=.12)
            ax[j].set_xlabel(xl); ax[j].set_ylim(0, 1)
        ax[0].set_ylabel('normalized altitude (0=base, 1=top)')
        ax[0].legend(fontsize=7, loc='lower right')
        fig.suptitle(f'Our VOCALS-REx median profiles vs adiabatic (n={used}); '
                     'orange = top decile z in [0.91,1]')
        fig.tight_layout()
        out = Path(__file__).resolve().parents[2] / 'docs' / 'cached_results' / 'our_bp2025_fig2.png'
        fig.savefig(out, dpi=130)
        print(f"\nwrote {out}")
    except Exception as e:                                  # noqa: BLE001
        print(f"\n(plot skipped: {e})")


if __name__ == '__main__':
    main()
