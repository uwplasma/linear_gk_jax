import jax.numpy as jnp
import jax.scipy as jsp
from scipy import stats


L = jnp.pi
nt = 100 
nTheta = 200
deltaTheta = L/nTheta
theta = jnp.linspace(-L-deltaTheta, L, num = 100)
Theta = jnp.meshgrid(theta, ) 
m = 1 # Mass placeholder
T = 1 #Temperature placeholder

#placeholder scalars
kPsi = 1  
kAlpha = 1 
q = 1 #Charge
mass = 1 
vPerp = 1

#The defined variables and gradients are placeholders to be replaced with the already created Equations 11-18 code

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

v = 1
nV = 100
vThe = 0.7
vDe = 5
vMine, vMaxe = -16*vThe, 16*vThe
ve = jnp.linspace(vMine, vMaxe, nV)
Theta, Ve = jnp.meshgrid(theta, ve, indexing= 'ij')
waveNumberElectrons = 0.8
Ae = 1
n0e = 4

F0 = n0e / jnp.linspace(2.*jnp.pi *vThe**2)*(0.5 * jnp.exp(-(Ve-vDe*vThe)**2 / (2.*vThe**2)) +0.5 * jnp.exp(-(Ve+vDe*vThe)**2 / (2.*vThe**2))) * (1 + Ae * jnp.sin(waveNumberElectrons * 2 * jnp.pi * Theta / L))
magF0 = jnp.linalg.norm(F0)

eta = 1 #Placeholder

gradF0 = jnp.gradient(F0)
den = m*freqCyclotron*T*magF0
bCrossK = jnp.cross(b, kPerp)
freqDiamagnetic = (jnp.dot(bCrossK, gradF0)) / den

velDepFreqDiamagnetic = freqDiamagnetic*(1 + eta*((v**2 / vThe**2) - 3/2))


vDrift = jnp.array([1,1,1]) #Replace this with cross product representation
freqMagneticDrift = jnp.dot(kPerp, vDrift)


Term = 