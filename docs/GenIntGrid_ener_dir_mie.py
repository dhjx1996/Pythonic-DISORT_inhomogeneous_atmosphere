import numpy as np
import PythonicDISORT
from numpy.polynomial.legendre import Legendre
from math import pi
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
# Gen data code

####################### Setup #######################

log_offset_tau0 = 1e-2
#log_offset_omega = 1e-4


# Parameter space
expected_Leg_coeffs = np.load("Leg_coeffs_050_36.npy")
Nr = 36 # Number of data points
r_eff_lin = np.linspace(4, 21, Nr) # Effective radius; units: microns

NQuad = 22

omega = 1-1e-6
Nt = 15
#No = 25
Nm = 20

tau0_data_lin = np.linspace(np.log2(50 + log_offset_tau0), np.log2(1e-6 + log_offset_tau0), Nt)
#omega_data_lin = np.linspace(np.log2(1e-6 + log_offset_omega), np.log2(0.5 + log_offset_omega), No)
mu0_data_lin = np.linspace(0.01, 1, Nm)

dt = tau0_data_lin[2] - tau0_data_lin[1]
#do = omega_data_lin[2] - omega_data_lin[1]
dm = mu0_data_lin[2] - mu0_data_lin[1]

#################### LOOP #############################
def loop(i):
    ir = i // (Nt * Nm)
    rem = i % (Nt * Nm)

    it = rem // Nm

    im = rem % Nm
    
    if i % 10 == 1:
        print("ir =", ir, "it =", it, flush=True)
    
    '''
    ############### Perturb points ###############
    tau0 = np.exp(tau0_data_lin[it] + (np.random.random() - 0.5) * dt).clip(min=1e-6, max=50)
    omega = (1 - np.exp(omega_data_lin[io] + (np.random.random() - 0.5) * do)).clip(min=0.5, max=1-1e-6)
    mu0 = (mu0_data_lin[im] + (np.random.random() - 0.5) * dm).clip(min=0.1, max=1)
    ##############################################
    '''
    ##############################################
    reff = r_eff_lin[ir]
    tau0 = 2 ** (tau0_data_lin[it]) - log_offset_tau0
    #omega = 1 - (2 ** (omega_data_lin[io]) - log_offset_omega)
    mu0 = mu0_data_lin[im]
    ##############################################

    NLeg = NQuad
    N = NQuad // 2
    
    Leg_coeffs_all = expected_Leg_coeffs[ir, :(NQuad + 1)]
    
    I0 = 1 / mu0
    flux_up, flux_down, u0 = PythonicDISORT.pydisort(
        tau0,
        omega,
        NQuad,
        Leg_coeffs_all,
        mu0,
        I0,
        0,
        only_flux=True,
        f_arr=Leg_coeffs_all[-1]
        #autograd_compatible=True,
    )[1:4]
    
    #mu_arr_pos, W = PythonicDISORT.subroutines.Gauss_Legendre_quad(N)
    R_true = flux_up(0)
    #T_true = 2 * pi * (mu_arr_pos * W) @ u0(tau0)[N:]
    T_true = flux_down(tau0)[0]
        
    return (  
        # Inputs
        reff,
        tau0,
        #omega,
        mu0,
        # Transmittance
        T_true,
        # Reflectance
        R_true,
    )

results = Parallel(n_jobs=-2)(delayed(loop)(i) for i in range(Nr * Nt * Nm))
(   
    # Inputs
    r_eff_data,
    tau0_data,
    #omega_data,
    mu0_data,
    # Transmittance
    T_true_data,
    # Reflectance
    R_true_data,
) = map(np.array, zip(*results))

np.savez_compressed(
    "RTg_SQDirectData_28July2025_DYAMOND2_Mie",
    # Inputs
    r_eff_data=r_eff_data.reshape(Nr, Nt, Nm),
    tau0_data=tau0_data.reshape(Nr, Nt, Nm),
    #omega_data=omega_data.reshape(Nr, Nt, No, Nm),
    mu0_data=mu0_data.reshape(Nr, Nt, Nm),
    # Transmittance
    T_true_data=T_true_data.reshape(Nr, Nt, Nm),
    # Reflectance
    R_true_data=R_true_data.reshape(Nr, Nt, Nm),
)