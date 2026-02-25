
import jax.numpy as jnp
import jax
from jax import jit
from netCDF4 import Dataset
import h5py as h5 
import interpax 
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
import equinox as eqx




class Grid(eqx.Module):
    """Magnetic field configuration terms

    """

  
    nz: int  #int = eqx.field(static=True)
    z_grid: Float[Array, "nz"]
    
    def __init__(
        self,
        nz: int,
        z_grid : Float[Array,'...'],

    ):


        self.nz=nz
        self.z_grid=z_grid



    @classmethod
    def construct_from_field(cls,
        field, Nz
    ):
        """Construct Grid

        Parameters
        ----------
        """

        abs_gp = jnp.abs(field.gradpar)
        theta_sorted = field.theta_grid
        dtheta = theta_sorted[1] - theta_sorted[0]
        z_grid = jnp.zeros(Nz)
        for i in range(1, Nz):
            z_grid = z_grid.at[i].set(z_grid[i-1] + dtheta / abs_gp[i-1])


        data = {}
        data["nz"] = Nz
        data["z_grid"] = z_grid


        return cls(**data)


    # @property
    # def any_function(self):
    #     return self.z_grid**2