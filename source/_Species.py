
import jax.numpy as jnp
import jax
from jax import jit
from netCDF4 import Dataset
import h5py as h5 
import interpax 
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
import equinox as eqx
from _field import Field



@jit
class Species(eqx.Module):
    number_species: int
    radial_points: int
    species_indeces: int
    mass_mp: Float[Array, "number_species"]  
    charge_qp: Float[Array, "number_species"]  
    temperature: Float[Array, "..."]#Float[Array, "number_species x radial_points"]    # in units of eV
    density: Float[Array, "..."]#Float[Array, "species x radial_points"]    # in units of particles/m^3
    Er: Float[Array, "radial_points"] 
    r_grid: Float[Array, "radial_points"] 
    r_grid_half: Float[Array, "radial_points"] 
    dr: float
    Vprime_half: Float[Array, "radial_points"] 
    overVprime: Float[Array, "radial_points"] 
    n_edge: float
    T_edge: float

    #v_thermal: Float[Array, "species x radial_points"]    # in units of particles/m^3
    @property
    def v_thermal(self):
        return jax.vmap(jax.vmap(get_v_thermal,in_axes=(None,0)),in_axes=(0,0))(self.mass,self.temperature)
    @property
    def charge(self):
        return self.charge_qp*elementary_charge
    @property
    def mass(self):
        return self.mass_mp*proton_mass
    @property
    def dTdr(self): 
        return jax.vmap(get_gradient_temperature,in_axes=(0,None,None,None,0))(self.temperature,self.r_grid,self.r_grid_half,self.dr,self.T_edge)
    @property
    def dndr(self): 
        return jax.vmap(get_gradient_density,in_axes=(0,None,None,None,0))(self.density,self.r_grid,self.r_grid_half,self.dr,self.n_edge)
    @property
    def dErdr(self): 
        return get_gradient_Er(self.Er) 
    @property
    def A1(self):
        return jax.vmap(get_Thermodynamical_Forces_A1,in_axes=(0,0,0,0,0,None))(self.charge,self.density,self.temperature,self.dndr,self.dTdr,self.Er)
    @property
    def A2(self):
        return jax.vmap(get_Thermodynamical_Forces_A2,in_axes=(0,0))(self.temperature,self.dTdr)
    @property
    def A3(self):
        return get_Thermodynamical_Forces_A3(self.Er)
    @property
    def diffusion_Er(self):
        return get_diffusion_Er(self.Er,self.r_grid,self.r_grid_half,self.Vprime_half,self.overVprime,self.dr)