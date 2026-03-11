import jax.numpy as jnp
import _fields

class Grid(Field):

    nz: int
    
    @classmethod
    def vel_Grid(v_max, T, m, Nv_par, Nv_perp):
        v_par = jnp.linspace(-v_max, v_max, Nv_par)
        v_perp = jnp.linspace(0.0, v_max, Nv_perp)
        v_th = jnp.sqrt(2.0 * T / m)
        return v_par, v_perp, v_th
    