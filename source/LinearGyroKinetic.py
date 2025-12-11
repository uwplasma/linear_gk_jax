#For electrons ∂

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import diffrax
import matplotlib.pyplot as plt

#Parameters
Lz = 2.0 * jnp.pi      # domain length in z
Nz = 128              # grid points in z
Nv_par = 64          # grid points in v_parallel
Nv_perp = 32          # grid points in v_perp
v_max = 3.0            # velocity range
k_perp = 0.5           # perpendicular wavenumber
q, T, m = 1.0, 1.0, 1.0
rho = jnp.sqrt(T / m)

# Grid
z = jnp.linspace(0.0, Lz, Nz, endpoint=False)
dz = Lz / Nz
v_par = jnp.linspace(-v_max, v_max, Nv_par)
v_perp = jnp.linspace(0.0, v_max, Nv_perp)

# Spectral wavenumbers for ∂/∂z
kz = jnp.fft.fftfreq(Nz, d=dz) * 2.0 * jnp.pi
ikz = 1j * kz

# Maxwellian fM(v_par, v_perp)
v_th = jnp.sqrt(2.0 * T / m)
fM_par = (1.0 / jnp.sqrt(jnp.pi * v_th**2)) * jnp.exp(-(v_par**2) / (v_th**2))
fM_perp = (v_perp / (v_th**2)) * jnp.exp(-(v_perp**2) / (v_th**2))
fM_2D = fM_par[None, :, None] * fM_perp[None, None, :]  # (1,Nv_par,Nv_perp)

# Profiles
omega_d = 0.2 * (1.0 + 0.1 * jnp.sin(2.0 * jnp.pi * z / Lz))[:, None, None]
omega_star = 0.05 * (1.0 + 0.5 * jnp.cos(jnp.pi * z / Lz))

# Bessel Function J0(k_perp * rho(v_perp))
J0_krho = jsp.special.i0(k_perp * v_perp / v_th)  # (Nv_perp,)

#Functions
def spectral_dz(field_z):
    fk = jnp.fft.fft(field_z)
    dfk = ikz * fk
    return jnp.fft.ifft(dfk)

def flatten_state(h):
    re, im = jnp.real(h), jnp.imag(h)
    return jnp.concatenate([re.ravel(), im.ravel()])

def unflatten_state(y):
    N = Nz * Nv_par * Nv_perp
    re = y[:N].reshape((Nz, Nv_par, Nv_perp))
    im = y[N:].reshape((Nz, Nv_par, Nv_perp))
    return re + 1j * im

# Potential φ(z,t)
def phi_of_tz(t):
    amp = 1e-3 * jnp.exp(-0.1 * t)
    return amp * jnp.sin(2.0 * jnp.pi * z / Lz)

# RHS of distribution
def rhs_fun(t, y_real, args):
    h = unflatten_state(y_real)  # (Nz, Nv_par, Nv_perp)

    # ∂h/∂z for each (v_par,v_perp)
    def dz_col(col):
        return spectral_dz(col)
    dz_vmap = jax.vmap(jax.vmap(dz_col, in_axes=1, out_axes=1), in_axes=2, out_axes=2)
    dh_dz = dz_vmap(h)

    streaming = - (v_par[None, :, None] * dh_dz)
    drift = -1j * omega_d * h

    phi_z = phi_of_tz(t)
    drive = 1j * (q/T) * (omega_star[:, None, None] * fM_2D *
                           phi_z[:, None, None] * J0_krho[None, None, :])

    dhdt = streaming + drift + drive
    return flatten_state(dhdt)

#Initial condition
key = jax.random.PRNGKey(0)
h0 = 1e-6 * (jax.random.normal(key, (Nz, Nv_par, Nv_perp)) +
             1j * jax.random.normal(key + 1, (Nz, Nv_par, Nv_perp)))
y0 = flatten_state(h0)

#Integrate
term = diffrax.ODETerm(rhs_fun)
solver = diffrax.Tsit5()
t0, t1, dt0 = 0.0, 1.0, 1e-3
saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 21))
controller = diffrax.PIDController(rtol=1e-6, atol=1e-9)

sol = diffrax.diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt0,
                          y0=y0, saveat=saveat,
                          stepsize_controller=controller)

#Diagnostics
def L2_norm(y_real):
    h = unflatten_state(y_real)
    return jnp.sqrt(jnp.sum(jnp.abs(h)**2) * dz *
                    (2*v_max/Nv_par) * (v_max/Nv_perp))
norms = jax.vmap(L2_norm)(sol.ys)

print("Times:", sol.ts)
print("Final norm:", norms[-1])

#Plotting
h_final = unflatten_state(sol.ys[-1])

# Choose two v_perp values and one v_par slice
v_perp_indices = [0, Nv_perp//2, Nv_perp-1]
v_par_index = Nv_par // 2

plt.figure(figsize=(8,5))
for i in v_perp_indices:
    plt.plot(z, jnp.real(h_final[:, v_par_index, i]),
             label=f"v_perp={v_perp[i]:.2f}")
plt.xlabel("z")
plt.ylabel("Re[h](z)")
plt.title("Final Re[h(z)] at fixed v_par")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
for i in v_perp_indices:
    plt.plot(z, jnp.imag(h_final[:, v_par_index, i]),
             label=f"v_perp={v_perp[i]:.2f}")
plt.xlabel("z")
plt.ylabel("Im[h](z)")
plt.title("Final Im[h(z)] at fixed v_par")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(6,4))
plt.plot(sol.ts, norms)
plt.xlabel("t")
plt.ylabel("||h||₂")
plt.title("L2 norm")
plt.tight_layout()
plt.show()
