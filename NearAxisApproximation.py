import jax.numpy as jnp
import jax.scipy as jsp


L = jnp.pi
nt = 100 
nTheta = 200

theta = jnp.linspace(-L, L, num = 100)
Theta = jnp.meshgrid(theta, ) 

#placeholder scalars
kPsi = 1  
kAlpha = 1 
q = 1 #Charge
mass = 1 
vPerp = 1

def kappa(a):
    x,y,z = a
    return jnp.array(x, y, z)

def B(a):
    x,y,z = a
    return jnp.array([jnp.sin(x), jnp.cos(y), jnp.tan(z)])
magB = jnp.linalg.norm(B)
gradB = jnp.gradient(B)

b = B / magB

def Psi(a):
    x,y,z = a
    return jnp.array([x, y, z])
gradPsi = jnp.gradient(Psi)

def Alpha(a):
    x,y,z = a
    return jnp.array([jnp.cos(x), jnp.cos(y), jnp.cos(z)])
gradAlpha = jnp.gradient(Alpha)

gradPsiMag = jnp.linalg.norm(gradPsi)
gradAlphaMag = jnp.linalg.norm(gradAlpha)

dotPsiAlpha = jnp.dot(gradPsi, gradAlpha)

eq16 = jnp.dot(jnp.cross(b, gradB), gradAlpha)
eq17 = jnp.dot(jnp.cross(b, gradB), gradPsi)
eq18 = jnp.dot(jnp.cross(b, kappa), gradAlpha)

kPerp = jnp.array([(kPsi*gradPsi) + (kAlpha*gradAlpha)])
kPerpMag = kPsi*(gradPsiMag**2) + kAlpha*(gradAlphaMag**2) + 2*kPsi*kAlpha*(dotPsiAlpha)

freqCyclotron = (q * magB)/mass
besselArg = (vPerp * jnp.sqrt(kPerpMag)) / freqCyclotron
J0 = jsp.special.i0(besselArg)

freqDrift = ((kPsi*eq16) + (kAlpha*eq17) + eq18) / freqCyclotron






