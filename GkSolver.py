

def solver(gradpar, theta, v_perp, v_th,N, q, T, v_par, omega_d, omega_star, fm_2d ):

    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    Nz, Nv_par, Nv_perp = N
    
    abs_gp = jnp.abs(gradpar)
    theta_sorted = theta
    dtheta = theta_sorted[1] - theta_sorted[0]

    z_raw = jnp.zeros(Nz)
    for i in range(1, Nz):
        z_raw = z_raw.at[i].set(z_raw[i-1] + dtheta / abs_gp[i-1])

    Lz = 2 * jnp.pi
    scale = Lz / z_raw[-1]
    z = scale * z_raw

    dz = z[1] - z[0]

    kz = jnp.fft.fftfreq(Nz, d=dz) * 2.0 * jnp.pi
    ikz = 1j * kz

    def phi_of_tz(t):
        amp = 1e-3 * jnp.exp(-0.1 * t)
        return amp * jnp.sin(2.0 * jnp.pi * z / Lz)


    k_perp = 0.5
    J0_krho = jsp.special.i0(k_perp * v_perp / v_th)

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




    def rhs_fun(t, y_real, args):
        h = unflatten_state(y_real)

        def dz_col(col):
            return spectral_dz(col)

        dz_vmap = jax.vmap(
            jax.vmap(dz_col, in_axes=1, out_axes=1),
            in_axes=2, out_axes=2
        )
        dh_dz = dz_vmap(h)

        streaming = - (v_par[None, :, None] * dh_dz)


        drift = -1j * omega_d * h

        phi_z = phi_of_tz(t)
        drive = 1j * (q/T) * (omega_star[:, None, None] *
                            fM_2D *
                            phi_z[:, None, None] *
                            J0_krho[None, None, :])

        dhdt = streaming + drift + drive
        return flatten_state(dhdt)
    
    return 