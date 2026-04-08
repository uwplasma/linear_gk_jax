#This module contains the diffrax solver part as well as the input, field and species generation.


import jax
import jax.numpy as jnp
from ._Grids import Grid
from ._Field import Field
from ._Species import Species
from ._physics import rhs_fun, compute_phi, spectral_dz, flatten_state, unflatten_state
import diffrax

###TODO: Complete with intialization of the field, grid and species, and the integration of the ODE using diffrax.