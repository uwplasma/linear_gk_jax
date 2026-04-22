#This module contains the diffrax solver part as well as the input, field and species generation.


import jax
import jax.numpy as jnp
from ._Grids import Grid
from ._Field import Field
from ._Species import Species
from ._physics import rhs_fun, compute_phi, spectral_dz, flatten_state, unflatten_state
import diffrax


def integrate(rhs):
  key = jax.random.PRNGKey(0)
  h0 = 1e-6 * (jax.random.normal(key, (Nz, Nv_par, Nv_perp))
             + 1j * jax.random.normal(key+1, (Nz, Nv_par, Nv_perp)))
  y0 = flatten_state(h0)


  # Integrate

  term = diffrax.ODETerm(rhs_fun)
  solver = diffrax.Dopri5()
  t0, t1, dt0 = 0.0, 1.0, 1e-3
  saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 21))
  controller = diffrax.PIDController(rtol=1e-6, atol=1e-9)

  sol = diffrax.diffeqsolve(term, solver,
                          t0=t0, t1=t1,
                          dt0=dt0,
                          y0=y0,
                          saveat=saveat,
                          stepsize_controller=controller)


  # Results

  def L2_norm(y_real):
      h = unflatten_state(y_real)
      return jnp.sqrt(jnp.sum(jnp.abs(h)**2) * dz *
                    (2*v_max/Nv_par) * (v_max/Nv_perp))

  norms = jax.vmap(L2_norm)(sol.ys)
  h_final = unflatten_state(sol.ys[-1])
  v_perp_indices = [0, Nv_perp//2, Nv_perp-1]
  v_par_index = Nv_par // 2
  return(norms, h_final, v_perp_indices, v_par_index)
###TODO: Complete with intialization of the field, grid and species, and the integration of the ODE using diffrax.
