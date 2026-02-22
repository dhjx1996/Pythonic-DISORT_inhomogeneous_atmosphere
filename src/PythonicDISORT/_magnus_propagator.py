import numpy as np
import scipy.linalg


def _compute_magnus_propagator(A_func, S_func, tau_bot, N_steps, NQuad):
    """
    Accumulates the full-domain Magnus propagator (Phi_hom, phi_part) for the ODE

        dI/dtau = A(tau) I + S(tau),    I in R^{NQuad},    tau in [0, tau_bot].

    Uses first-order Magnus (midpoint rule): one (NQuad+1) x (NQuad+1) matrix
    exponential per step via the extended-state-vector identity

        exp([[h*A, h*S], [0, 0]]) = [[exp(h*A), A^{-1}(exp(h*A)-I)h*S], [0, 1]]

    which gives the homogeneous step propagator and the particular-solution
    increment without explicit matrix inversion.

    Arguments
    ---------
    A_func   : callable, tau (float) -> (NQuad, NQuad) ndarray
    S_func   : callable, tau (float) -> (NQuad,) ndarray  (zeros if no beam)
    tau_bot  : float > 0, optical depth of the bottom boundary
    N_steps  : int >= 1, number of equidistant Magnus steps
    NQuad    : int, 2N (total number of quadrature streams)

    Returns
    -------
    Phi_hom  : (NQuad, NQuad) ndarray
        Full-domain homogeneous propagator; satisfies
        I_hom(tau_bot) = Phi_hom @ I_hom(0).
    phi_part : (NQuad,) ndarray
        Full-domain particular-solution increment; satisfies
        I(tau_bot) = Phi_hom @ I(0) + phi_part.
    """
    h = tau_bot / N_steps
    Phi_hom = np.eye(NQuad)
    phi_part = np.zeros(NQuad)

    ext = NQuad + 1
    M_ext = np.zeros((ext, ext))  # reused each step; last row stays zero

    for k in range(N_steps):
        tau_mid = (k + 0.5) * h

        A_k = A_func(tau_mid)  # (NQuad, NQuad)
        S_k = S_func(tau_mid)  # (NQuad,)

        M_ext[:NQuad, :NQuad] = h * A_k
        M_ext[:NQuad, NQuad] = h * S_k
        # M_ext[NQuad, :] remains all zeros

        Phi_ext = scipy.linalg.expm(M_ext)       # (ext, ext)
        Phi_k = Phi_ext[:NQuad, :NQuad]           # homogeneous step propagator
        delta_p_k = Phi_ext[:NQuad, NQuad]        # particular-solution increment

        phi_part = Phi_k @ phi_part + delta_p_k
        Phi_hom = Phi_k @ Phi_hom

    return Phi_hom, phi_part
