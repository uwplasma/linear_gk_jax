
#UNFINISHED, Just checking to see if this is on the right track

def rhs_func(field, V):
    '''
    Takes a field class(field), and velocity grid(V) as arguments
    Returns the drift, drive, and frequency(omega_d, omega_star, Omega) terms
    '''
    import numpy as np
    import jax.numpy as jnp
    from _field import Field

    omega_d = field.Bcross_gradB_grad_alpha[:, None, None] * vd_prefactor
    Omega = q * B[:, None, None] / m
    return drift, drive, omega_d, omega_star, Omega


