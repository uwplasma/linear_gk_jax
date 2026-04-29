
import jax
import jax.numpy as jnp
import diffrax

from _field import Field
from _Grids import Grid
from _Species import Species
from _physics import rhs_fun, flatten_state, unflatten_state

#Load field

EIK_FILE = "w7x_adiabatic_electrons_eik.nc"
field = Field.read_from_eik(EIK_FILE)
print(f"Field loaded: ntheta = {field.ntheta}")

#Species parameters

q   = 1.0   # charge
T   = 1.0   # temperature
m   = 1.0   # mass
eta = 1.0   # L_n / L_T

species = Species(
    number_species=1,
    species_indeces=0,
    mass_mp=jnp.array([m]),
    charge_qp=jnp.array([q]),
)

#Build Grid via Grid.fromfield
#
#  V = (v_max, nv_par, nv_perp) is the only extra input needed.
#  Grid.fromfield constructs:
#    grid.Vel  = [v_par, v_perp, v_th]   (velocity axes)
#    grid.Freq = [Omega, omega_d, omega_star]
#    grid.z_grid / grid.nz               (parallel coordinate from field)

v_max   = 3.0
nv_par  = 16
nv_perp = 8
V = (v_max, nv_par, nv_perp)

grid = Grid.fromfield(field, V, q=q, T=T, m=m, eta=eta)
print(f"Grid built: nz={grid.nz}, nv_par={nv_par}, nv_perp={nv_perp}")

# Unpack velocity axes from grid for use in IC and norms
v_par  = grid.Vel[0]   # shape (nv_par,)
v_perp = grid.Vel[1]   # shape (nv_perp,)

Nz      = grid.nz
Nv_par  = nv_par
Nv_perp = nv_perp
dz      = (grid.z_grid[-1] - grid.z_grid[0]) / Nz

#Initial condition

key = jax.random.PRNGKey(0)
h0  = 1e-6 * (
    jax.random.normal(key,          (Nz, Nv_par, Nv_perp))
  + 1j * jax.random.normal(key + 1, (Nz, Nv_par, Nv_perp))
)
y0 = flatten_state(h0)

#Integrate

t0, t1, dt0 = 0.0, 1.0, 1e-3
saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 21))

sol = diffrax.diffeqsolve(
    diffrax.ODETerm(rhs_fun),
    diffrax.Dopri5(),
    t0=t0,
    t1=t1,
    dt0=dt0,
    y0=y0,
    args=(field, grid, species),   # rhs_fun unpacks these
    saveat=saveat,
    stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
)


def L2_norm(y_real):
    h = unflatten_state(y_real)
    return jnp.sqrt(
        jnp.sum(jnp.abs(h) ** 2)
        * dz
        * (2 * v_max / Nv_par)
        * (v_max / Nv_perp)
    )

norms   = jax.vmap(L2_norm)(sol.ys)
h_final = unflatten_state(sol.ys[-1])

print("\nTime | L2 norm")
for t, n in zip(jnp.linspace(t0, t1, 21), norms):
    print(f"{t:.2f} | {n:.6e}")

print(f"\nFinal |h| max: {jnp.max(jnp.abs(h_final)):.6e}")
