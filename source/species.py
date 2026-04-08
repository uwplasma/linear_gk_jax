
import jax.numpy as jnp
import jax
from jax import jit
from netCDF4 import Dataset
import h5py as h5 
import interpax 
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
import equinox as eqx
from _field import Field

class species():
    def velocity_integral(f):
        return jnp.sum(f * velocity_measure, axis=(1, 2))

    def compute_phi(h):
        # Numerator: q ∫ d^3v J0 f
        num = q * velocity_integral(J0_krho[None, None, :] * h)

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

    def spectral_dz(field_z):
        fk = jnp.fft.fft(field_z)
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
