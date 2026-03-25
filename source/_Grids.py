
import jax.numpy as jnp
import jax
from jax import jit
from netCDF4 import Dataset
import h5py as h5 
import interpax 
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
import equinox as eqx
from _field import Field



class Grid(eqx.Module):
    
    @classmethod
    def fromfield(field, V, cls, q=1, T=1, m=1, eta=1):
        
        data = {}
        B = field.B

        Omega = q * B[:, None, None] / m
        Kalpha = field.kalpha
        df_dpsi = field.grad_psi[:, None, None]
        epsilon = 0.5 * (v_par[None, :, None]**2 + v_perp[None, None, :]**2)

        v_max = V[0]
        nv_par = V[1]
        nv_perp = V[2]
        v_par = jnp.linspace(-v_max, v_max, nv_par)
        v_perp = jnp.linspace(0, v_max, nv_perp)
        v_th = jnp.sqrt(2.0 * T / m)
        data["Vel"] = jnp.array([v_par, v_perp, v_th])

        vd_prefactor = (v_par[None, :, None]**2 +
                0.5 * v_perp[None, None, :]**2) / Omega

        omega_d = field.Bcross_gradB_grad_alpha[:, None, None] * vd_prefactor
        omega_star = (Kalpha * T / q) * df_dpsi * (
        1.0 + eta * ((epsilon / v_th**2) - 1.5))
        data["Freq"] = jnp.array([Omega, omega_d, omega_star])

        nz = Field.ntheta
        
        data["Nz"] = nz
        
        return cls(**data)