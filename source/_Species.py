
import jax.numpy as jnp
import jax
from jax import jit

from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
import equinox as eqx


###TODO: ADD Temperature and density, Ln and LT for species

@jit
class Species(eqx.Module):
    number_species: int
    species_indeces: int
    mass_mp: Float[Array, "number_species"]  
    charge_qp: Float[Array, "number_species"]  

    def __init__(
        self,
        number_species: int,
        species_indeces: int,
        mass_mp: Float[Array, "number_species"],
        charge_qp: Float[Array, "number_species"]
    ):

        self.number_species = number_species
        self.species_indeces = species_indeces
        self.mass_mp = mass_mp
        self.charge_qp = charge_qp