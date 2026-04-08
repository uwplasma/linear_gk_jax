
#This file contains the physics of the gyrokinetic equation,    
# including the RHS function for time-stepping 
# and the computation of the electrostatic potential phi. 



import jax.numpy as jnp
import jax

###TODO: Look into field, grids and species classes to see how to set up the functions below



def velocity_integral(f,species,grid):
    return jnp.sum(f * velocity_measure, axis=(1, 2))

def compute_phi(h):
    # Numerator: q ∫ d^3v J0 f
    num = species.q * velocity_integral(J0_krho[None, None, :] * h)

    # Denominator term
    denom_integral = velocity_integral(
        fM_2D * J0_krho[None, None, :]**2
    )

    denom = m * (q**2 / T) * (1.0 - denom_integral / m)

    return num / denom


# J0 factor
    k_perp = 0.5
    J0_krho = jsp.special.i0(k_perp * v_perp / v_th)


# flatten/unflatten + spectral derivative


##Example, here z comes from Grids or Field
def spectral_dz(field):
    fk = jnp.fft.fft(field.z_grid)
    dfk = ikz * fk
    return jnp.fft.ifft(dfk)

def flatten_state(h):
    re = jnp.real(h)
    im = jnp.imag(h)
    return jnp.concatenate([re.ravel(), im.ravel()])

def unflatten_state(y):
    N = Nz * Nv_par * Nv_perp
    re = y[:N].reshape((Nz, Nv_par, Nv_perp))
    im = y[N:].reshape((Nz, Nv_par, Nv_perp))
    return re + 1j * im


# RHS

def rhs_fun(t, y_real, args):
    #we unpack the field, grid and species from the args
    #TODO: Look into the field, grid and species classes to see what goes into each fucntion
    field, grid, species = args
    h = unflatten_state(y_real)

    # ∂h/∂z using FFT for each (v_par,v_perp)
    def dz_col(col):
        return spectral_dz(col)

    dz_vmap = jax.vmap(
        jax.vmap(dz_col, in_axes=1, out_axes=1),
        in_axes=2, out_axes=2
    )
    dh_dz = dz_vmap(h)

    # Streaming term: -v_par ∂h/∂z
    streaming = - (v_par[None, :, None] * dh_dz)

    # Magnetic drift term
    drift = -1j * omega_d * h

    # Drive term using omega_star (from grad_psi)
    phi_z = compute_phi(h)
    drive = 1j * omega_star * fM_2D * phi_z[:, None, None] * J0_krho[None, None, :]
    dhdt = streaming + drift + drive
    return flatten_state(dhdt)


