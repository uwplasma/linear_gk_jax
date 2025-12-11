import os
number_of_processors_to_use = 1
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={number_of_processors_to_use}'


import jax
import jax.numpy as jnp
import jax.scipy as jsp
import diffrax
import matplotlib.pyplot as plt

from _field import Field

input_file='w7x_adiabatic_electrons.eik.nc'


path = os.path.join(os.path.dirname(__name__), '../input_files', input_file)# Enter path here, unsure what this should be
field = Field.read_from_eik(path)

# Geometry
theta = field.theta_grid              # (Nz,)
Nz = field.ntheta
gradpar = field.gradpar               # ∂/∂z coefficient
B = field.B                           
  
# Drift
omega_d = field.Bcross_gradB_grad_alpha[:, None, None]  # shape (Nz,1,1)
omega_star = field.grad_psi                              # (Nz,)

#Velocity space grids and parameters

Nv_par = 100
Nv_perp = 50
v_max = 3.0
q, T, m = 1.0, 1.0, 1.0

v_par = jnp.linspace(-v_max, v_max, Nv_par)
v_perp = jnp.linspace(0.0, v_max, Nv_perp)

# Maxwellian
v_th = jnp.sqrt(2.0 * T / m)
fM_par = (1.0 / jnp.sqrt(jnp.pi * v_th**2)) * jnp.exp(-(v_par**2) / (v_th**2))
fM_perp = (v_perp / (v_th**2)) * jnp.exp(-(v_perp**2) / (v_th**2))
fM_2D = fM_par[None, :, None] * fM_perp[None, None, :]

# 3. Parallel coordinate and spectral operator from Field
#    The solver expects a uniform z-grid; Field gives theta.
#    We map theta → z via a cumulative integral using gradpar.

# z' = ∫ dθ / |gradpar|
abs_gp = jnp.abs(gradpar)
theta_sorted = theta
dtheta = theta_sorted[1] - theta_sorted[0]

#z(theta)
z_raw = jnp.zeros(Nz)
for i in range(1, Nz):
    z_raw = z_raw.at[i].set(z_raw[i-1] + dtheta / abs_gp[i-1])

#Set domain = 2*pi
Lz = 2 * jnp.pi
scale = Lz / z_raw[-1]
z = scale * z_raw

dz = z[1] - z[0]

# spectral wavenumbers for ∂/∂z
kz = jnp.fft.fftfreq(Nz, d=dz) * 2.0 * jnp.pi
ikz = 1j * kz

#phi(z,t): simple decaying mode (can be replaced w/ GK quasineutrality)

def phi_of_tz(t):
    amp = 1e-3 * jnp.exp(-0.1 * t)
    return amp * jnp.sin(2.0 * jnp.pi * z / Lz)

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
    phi_z = phi_of_tz(t)
    drive = 1j * (q/T) * (omega_star[:, None, None] *
                           fM_2D *
                           phi_z[:, None, None] *
                           J0_krho[None, None, :])

    dhdt = streaming + drift + drive
    return flatten_state(dhdt)


#Initial condition

key = jax.random.PRNGKey(0)
h0 = 1e-6 * (jax.random.normal(key, (Nz, Nv_par, Nv_perp))
             + 1j * jax.random.normal(key+1, (Nz, Nv_par, Nv_perp)))
y0 = flatten_state(h0)


# Integrate

term = diffrax.ODETerm(rhs_fun)
solver = diffrax.Dopri5()
t0, t1, dt0 = 0.0, 1.0, 1e-3
saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 100))
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
print("Times:", sol.ts)
print("Final norm:", norms[-1])


# Plotting

h_final = unflatten_state(sol.ys[-1])
v_perp_indices = [0, Nv_perp//2, Nv_perp-1]
v_par_index = Nv_par // 2

plt.figure(figsize=(8,5))
for i in v_perp_indices:
    plt.plot(z, jnp.real(h_final[:, v_par_index, i]),
             label=f"v_perp={v_perp[i]:.2f}")
plt.xlabel("z")
plt.ylabel("Re[h]")
plt.title("Final Re[h(z)]")
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig('Real.png')

plt.figure(figsize=(8,5))
for i in v_perp_indices:
    plt.plot(z, jnp.imag(h_final[:, v_par_index, i]),
             label=f"v_perp={v_perp[i]:.2f}")
plt.xlabel("z")
plt.ylabel("Im[h]")
plt.title("Final Im[h(z)]")
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig('Img.png')

# plt.figure(figsize=(8,5))
# for i in v_perp_indices:
#     plt.plot(sol.ts, jnp.square(jnp.imag(sol.ys[:,10, v_par_index, i])),
#              label=f"v_perp={v_perp[i]:.2f}")
# plt.xlabel("z")
# plt.ylabel("Im[h]")
# plt.title("Final Im[h(z)]")
# plt.legend()
# plt.tight_layout()
# #plt.show()
# plt.savefig('Norm_Img_local.png')

plt.figure(figsize=(6,4))
plt.plot(sol.ts, norms)
plt.xlabel("t")
plt.ylabel("||h||₂")
plt.title("Time evolution")
plt.tight_layout()
#plt.show()
plt.savefig('norms.png')