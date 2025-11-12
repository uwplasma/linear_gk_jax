
import jax.numpy as jnp
import jax
from jax import jit
from netCDF4 import Dataset
import h5py as h5 
import interpax 
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
import equinox as eqx

class Field(eqx.Module):
    """Magnetic field configuration terms

    """

  
    ntheta: int  #int = eqx.field(static=True)
    theta_grid: Float[Array, "n_theta"]
    grad_alpha_squared: Float[Array, "n_theta"]
    grad_alpha_dot_grad_psi: Float[Array, "n_theta"]
    grad_psi_squared: Float[Array, "n_theta"]   
    B: Float[Array, "n_theta"]  
    Bcross_gradB_grad_psi: Float[Array, "n_theta"] 
    Bcross_gradB_grad_alpha: Float[Array, "n_theta"] 
    Bcross_kappa_grad_psi: Float[Array, "n_theta"] 
    Bcross_kappa_grad_alpha: Float[Array, "n_theta"] 
    gradpar: Float[Array, "n_theta"]
    grad_psi: Float[Array, "n_theta"]   
    

    def __init__(
        self,
        ntheta: int,
        theta_grid : Float[Array,'...'],
        grad_alpha_squared : Float[Array,'...'],
        grad_alpha_dot_grad_psi : Float[Array,'...'],
        grad_psi_squared: Float[Array,'...'],
        B : Float[Array,'...'],
        Bcross_gradB_grad_psi : float,
        Bcross_gradB_grad_alpha : Float[Array,'...'],
        Bcross_kappa_grad_psi : Float[Array,'...'],
        Bcross_kappa_grad_alpha : Float[Array,'...'],
        gradpar : Float[Array,'...'],
        grad_psi : Float[Array,'...']
    ):


        self.ntheta=ntheta
        self.theta_grid=theta_grid
        self.grad_alpha_squared=grad_alpha_squared
        self.grad_alpha_dot_grad_psi=grad_alpha_dot_grad_psi
        self.grad_psi_squared=grad_psi_squared
        self.B=B
        self.Bcross_gradB_grad_psi=Bcross_gradB_grad_psi
        self.Bcross_gradB_grad_alpha=Bcross_gradB_grad_alpha
        self.Bcross_kappa_grad_psi=Bcross_kappa_grad_psi
        self.Bcross_kappa_grad_alpha=Bcross_kappa_grad_alpha
        self.gradpar=gradpar
        self.grad_psi=grad_psi


    @classmethod
    def read_from_eik(cls,
        eik_file: str, 
    ):
        """Construct Field from gx/gs2-like eik file.

        Parameters
        ----------
        eik_file : path-to-eik-file
        """
        from netCDF4 import Dataset

        #This should go to equilibrium reader
        efile = Dataset(eik_file, mode="r")

        theta_grid = efile.variables["theta"][:].filled()
        ntheta = len(theta_grid)
        grad_alpha_squared = efile.variables["gds2"][:].filled()
        grad_alpha_dot_grad_psi = efile.variables["gds21"][:].filled()
        grad_psi_squared = efile.variables["gds22"][:].filled()
        B = efile.variables["bmag"][:].filled()
        Bcross_gradB_grad_psi = efile.variables["gbdrift0"][:].filled()
        Bcross_gradB_grad_alpha = efile.variables["gbdrift"][:].filled()
        Bcross_kappa_grad_psi = efile.variables["cvdrift0"][:].filled()
        Bcross_kappa_grad_alpha = efile.variables["cvdrift"][:].filled()
        gradpar = efile.variables["gradpar"][:].filled()
        grad_psi = efile.variables["grho"][:].filled()

        efile.close()

        data = {}
        data["ntheta"] = ntheta
        data["theta_grid"] = theta_grid
        data["grad_alpha_squared"] = grad_alpha_squared
        data["grad_alpha_dot_grad_psi"] = grad_alpha_dot_grad_psi
        data["grad_psi_squared"] = grad_psi_squared
        data["B"] = B
        data["Bcross_gradB_grad_psi"] = Bcross_gradB_grad_psi
        data["Bcross_gradB_grad_alpha"] = Bcross_gradB_grad_alpha
        data["Bcross_kappa_grad_psi"] = Bcross_kappa_grad_psi
        data["Bcross_kappa_grad_alpha"] = Bcross_kappa_grad_alpha
        data["gradpar"] = gradpar
        data["grad_psi"] = grad_psi

        return cls(**data)




    @classmethod
    def read_from_pyqsc(cls,s: float,
     r: float,
     config_id:int,
    ntheta:int, 
      nphi:int=51
    ):
        """Construct Field from pyqsc database

        Parameters
        ----------
        s : radial position (toroidal flux normalized)
        r : minor radius at which to compute the field (m), aka boundary radius
        config_id : pyQSC configuration ID from the database
        ntheta : number of poloidal grid points for discretization
        nphi : number of toroidal grid points for pyqsc evaluations
        """
        from netCDF4 import Dataset
        import requests
        from scipy.interpolate import interp1d
        from qsc import Qsc

        #URL with database 
        url="https://stellarator.physics.wisc.edu/backend/api/configs"

        config = next(cfg for cfg in requests.get(url).json()["configs"] if cfg["id"] == config_id)

        #Running pyQSC to retrieve configurartion geometry
        stel = Qsc(rc=[1, config["rc1"], config["rc2"], config["rc3"]],
                            zs=[0, config["zs1"], config["zs2"], config["zs3"]],
                            nfp=config["nfp"], etabar=config["etabar"],
                            I2=0., order="r3", B2c=config["B2c"], p2=config["p2"],nphi=nphi)
        
        #Assert some of the parameters (sanity check)
        assert stel.iota - config["iota"]<1.e-5
        assert stel.min_L_grad_B -config["min_L_grad_B"]<1.e-5
        print("Test Passed!") # Mercier criterion parameter DMerc multiplied by r^2


        #Calculate shear assuming B31c=0
        #Other values could be used. There is not correct choice in the Near Axis Expansion
        #As long as comparisons of the final heat flux are done at the same B31c=0
        stel.calculate_shear()
        iota2=stel.iota2

        #In case one would like to plot the configuration this commented lines could be used
        #stel.plot_boundary() # Plot the flux surface shape at the default radius r = 0.1
        #stel.plot() # Plot relevant near axis parameters

        #Get parameters from database and run pyQSC to obtain extra necessary parameters for input creation
        iota = abs(stel.iota)
        sigmaSol = stel.sigma
        sprime = stel.d_l_d_phi
        curvature = stel.curvature
        phi = stel.phi
        dpds=stel.p2
        nNormal = stel.iotaN - stel.iota
        NFP=stel.nfp
        etabar=stel.etabar
        B0=stel.B0
        Bbar=stel.Bbar
        spsi=stel.spsi

        pi = jnp.pi
        mu0 = 4*pi*10**(-7)

        ##Starting to work on GX input file parameters 
        nperiod  = 1                          #nperiod variable, keep nperiod=1
        drhodpsi = 1.0                        #Keep drhodpsi=1
        Aminor =  r                           #This should be the r, used when converting from VMEC, in general cases choosing r is the same as choosing Aminor_p variable from VMEC
        rmaj     =jnp.average(stel.R0)/r       #Aspect ratio
        phiEDGE=r * r * stel.spsi * stel.Bbar/2. #from to_vmec definition
        kxfac    = 1.0      #Keep kxfac=1


        ## Resolution
        ntheta_input=ntheta+1  #Has to be the same as ntheta in GX input ntheta=nz-1=96, then ntheta_input=ntheta+1 , value used by M. Landremann
        nz=ntheta_input
        ntgrid=int(jnp.floor(ntheta_input/2))


        ## Geometry and normalizations
        alpha = 0.0            # field line label, we will keep 0, which is a standard value used on GX database calculation (corresponds to a field line at the bad curvature zone usually)
        normalizedtorFlux = s  # normalization for the toroidal flux, corresponds to s_VMEC at which simulation is to be performed


        ## rVMEC=sqrt(s)*Aminor**2 or the definition below, basically toroidal radial coordinate at which computation is to be done, or r in NAExpansion 
        s_psi_VMEC=1  #For Vmnec comparisons change to -1
        Bbar=s_psi_VMEC*Bbar
        rVMEC=jnp.sqrt((2*phiEDGE*normalizedtorFlux)/B0)*spsi*s_psi_VMEC


        #Approximation of shear assuming B31c=0
        shat=-2.*rVMEC**2*iota2/(iota+iota2*rVMEC**2)
    
        alphaVMEC=alpha
        #Phi resolution
        Nphi=len(phi)-1

        #Reference values
        Lref=Aminor
        Bref=2.*phiEDGE/Lref**2   #This is equivalent to B0 for the current definition of Lref=r



        ## Interpolate near-axis sigma, sprime, and curvature functions for this specific theta grid. Curvature and sigma are important for nabla alpha and nabla psi calculations 
        #values depend a lot on Nphi
        sigmaTemp  = interp1d(phi,sigmaSol, kind="cubic")
        sprimeTemp = interp1d(phi,sprime, kind="cubic")
        curvTemp   = interp1d(phi,curvature, kind="cubic")
    
        #phi grid for each field period
        period=2*pi*(1-1/Nphi)/NFP
        def phiToNFP(phi):
            if phi==0:
                phiP=0
            else:
                phiP=-phi%period
            return phiP

        #Defining the sigma, sprime and curvature functions
        def sigma(phi):      return sigmaTemp(phiToNFP(phi))
        def sprimeFunc(phi): return sprimeTemp(phiToNFP(phi))
        def curvFunc(phi):   return curvTemp(phiToNFP(phi))

        # defining phi_boozer and vartheta_boozer
        def Phi(theta):         return (theta - alpha)/(iota)
        def Theta(theta):         return theta-nNormal*Phi(theta)

        #Magnetic field strength
        def bmagNew(theta):     return B0*(1+rVMEC*etabar*jnp.cos(Theta(theta)))/Bref

        ####Not used
        def gradparNew(theta):  
            return   Lref*iota*(1+rVMEC*etabar*jnp.cos(Theta(theta)))/sprimeFunc(Phi(theta))


        def gds2New(theta):     
            return Lref**2*normalizedtorFlux*B0**2/(Bbar**2*rVMEC**2*etabar**2*curvFunc(Phi(theta))**2)*(etabar**4*jnp.cos(Theta(theta))**2 + curvFunc(Phi(theta))**4*(jnp.sin(Theta(theta))+jnp.cos(Theta(theta))*sigma(Phi(theta)))**2)

        #|nabla alpha||nabla psi|        
        def gds21New(theta):    
            return shat/Bref*(B0/(2.*Bbar*etabar**2*curvFunc(Phi(theta))**2))*((etabar**4+curvFunc(Phi(theta))**4*(sigma(Phi(theta))**2-1))*jnp.sin(2.*Theta(theta))-2.*curvFunc(Phi(theta))**4*sigma(Phi(theta))*jnp.cos(2.*Theta(theta)))

        #|nabla psi|**2
        def gds22New(theta):   
            return (shat**2/(Lref**2*Bref**2*normalizedtorFlux))*(B0**2*rVMEC**2)/(etabar**2*curvFunc(Phi(theta))**2)*(etabar**4*jnp.sin(Theta(theta))**2+(curvFunc(Phi(theta))**4)*(jnp.cos(Theta(theta))-jnp.sin(Theta(theta))*sigma(Phi(theta)))**2)

        #gbdrift       
        def gbdriftNew(theta):  
            return 2.*Bref*Lref**2*jnp.sqrt(normalizedtorFlux)*(etabar/(Bbar*rVMEC))*(jnp.cos(Theta(theta)))*(1-rVMEC*etabar*jnp.cos(Theta(theta)))
        
        #cvdrift    
        def cvdriftNew(theta):  return gbdriftNew(theta)+2.*Bref*Lref**2/B0**2*jnp.sqrt(normalizedtorFlux)*(mu0*dpds/Bbar)*(1-2.*rVMEC*etabar*jnp.cos(Theta(theta))) #Added pressure term

        #gbdrift0
        def gbdrift0New(theta): 
            return shat*2./(jnp.sqrt(normalizedtorFlux))*(rVMEC*etabar*jnp.sin(Theta(theta)))*(1-rVMEC*etabar*jnp.cos(Theta(theta)))#-2*np.sqrt(2)*np.sqrt(phiEDGE/B0)*shat*etabar*np.sin(theta)*(1-0*2*rVMEC*etabar*np.cos(theta))

        #cvdrift0=gbdrift0
        def cvdrift0New(theta): return gbdrift0New(theta)

        #grho
        def grho(theta): return jnp.sqrt(gds22New(theta)/shat**2)
    

        #Finding parallel z(theta_B) grid for GX
        npol=1
        thetaB_grid=jnp.linspace(-npol*jnp.pi, npol*jnp.pi, ntheta_input)

        #Allocating GX vectors
        gbdriftNew_val=jnp.zeros(ntheta_input)
        cvdriftNew_val=jnp.zeros(ntheta_input)
        gds2New_val=jnp.zeros(ntheta_input)
        bmagNew_val=jnp.zeros(ntheta_input)
        gds21New_val=jnp.zeros(ntheta_input)
        gds22New_val=jnp.zeros(ntheta_input)
        gbdrift0New_val=jnp.zeros(ntheta_input)
        cvdrift0New_val=jnp.zeros(ntheta_input)  
        gardparNew_val=jnp.zeros(ntheta_input)    
        grho_val=jnp.zeros(ntheta_input)       


        #Generating struct to of GX quantities
        gbdriftNew_temp=jnp.zeros(ntheta_input)
        cvdriftNew_temp=jnp.zeros(ntheta_input)
        gds2New_temp=jnp.zeros(ntheta_input)
        bmagNew_temp=jnp.zeros(ntheta_input)
        gds21New_temp=jnp.zeros(ntheta_input)
        gds22New_temp=jnp.zeros(ntheta_input)
        gbdrift0New_temp=jnp.zeros(ntheta_input)
        cvdrift0New_temp=jnp.zeros(ntheta_input)  
        gradpar_temp=jnp.zeros(ntheta_input)    
        grho_temp=jnp.zeros(ntheta_input)    

        for i in range(len(thetaB_grid)):
            gbdriftNew_temp=gbdriftNew_temp.at[i].set(gbdriftNew(thetaB_grid[i]))
            cvdriftNew_temp=cvdriftNew_temp.at[i].set(cvdriftNew(thetaB_grid[i]))
            gds2New_temp=gds2New_temp.at[i].set(gds2New(thetaB_grid[i]))
            bmagNew_temp=bmagNew_temp.at[i].set(bmagNew(thetaB_grid[i]))
            gds21New_temp=gds21New_temp.at[i].set(gds21New(thetaB_grid[i]))  
            gds22New_temp=gds22New_temp.at[i].set(gds22New(thetaB_grid[i]))                                          
            gbdrift0New_temp=gbdrift0New_temp.at[i].set(gbdrift0New(thetaB_grid[i]))
            cvdrift0New_temp=cvdrift0New_temp.at[i].set(cvdrift0New(thetaB_grid[i]))
            grho_temp=grho_temp.at[i].set(grho(thetaB_grid[i]))
            gradpar_temp=gradpar_temp.at[i].set(gradparNew(thetaB_grid[i]))

        ## Note: gradpar_half_grid has 1 less grid point than gradpar_temp
        gradpar_half_grid=jnp.zeros((ntheta_input-1))
        for itheta in range((ntheta_input-2)):
            gradpar_half_grid=gradpar_half_grid.at[itheta].set( 0.5 * (jnp.abs(gradpar_temp[itheta]) + jnp.abs(gradpar_temp[itheta+1])))
        gradpar_half_grid=gradpar_half_grid.at[-1].set(gradpar_half_grid[0])

        temp_grid = jnp.zeros((ntheta_input))
        z_on_theta_grid = jnp.zeros((ntheta_input))
        uniform_zgrid = jnp.zeros((ntheta_input))

        
        dtheta = thetaB_grid[1] - thetaB_grid[0]  ## dtheta in Boozer coordinates
        dtheta_pi = jnp.pi/ntgrid  ## dtheta on the SCALED uniform -pi,pi grid with 2*nzgrid+1 points
        index_of_middle = ntgrid



        for itheta in range(1,ntheta_input):
            temp_grid=temp_grid.at[itheta].set(temp_grid[itheta-1] + dtheta * (1. / jnp.abs(gradpar_half_grid[itheta-1])))

        for itheta in range(ntheta_input):
            z_on_theta_grid=z_on_theta_grid.at[itheta].set(temp_grid[itheta] - temp_grid[index_of_middle])

        desired_gradpar =jnp.pi/jnp.abs(z_on_theta_grid[0])

        for itheta in range(ntheta_input):
            z_on_theta_grid=z_on_theta_grid.at[itheta].set(z_on_theta_grid[itheta] * desired_gradpar)
            gardparNew_val=gardparNew_val.at[itheta].set(desired_gradpar) # setting entire gradpar array to the constant value "desired_gradpar"

    
        for itheta in range(ntheta_input):
            uniform_zgrid=uniform_zgrid.at[itheta].set(z_on_theta_grid[0] + itheta*dtheta_pi)



        #Interpolate quantities on the final uniform z_grid
        bmagNew_val=interpax.Interpolator1D(z_on_theta_grid,bmagNew_temp,extrap=True)(uniform_zgrid)
        grho_val=interpax.Interpolator1D(z_on_theta_grid,grho_temp,extrap=True)(uniform_zgrid)
        gbdriftNew_val=interpax.Interpolator1D(z_on_theta_grid,gbdriftNew_temp,extrap=True)(uniform_zgrid)
        cvdriftNew_val=interpax.Interpolator1D(z_on_theta_grid,cvdriftNew_temp,extrap=True)(uniform_zgrid)
        gbdrift0New_val=interpax.Interpolator1D(z_on_theta_grid,gbdrift0New_temp,extrap=True)(uniform_zgrid)
        cvdrift0New_val=interpax.Interpolator1D(z_on_theta_grid,cvdrift0New_temp,extrap=True)(uniform_zgrid)
        gds2New_val=interpax.Interpolator1D(z_on_theta_grid,gds2New_temp,extrap=True)(uniform_zgrid)    
        gds21New_val=interpax.Interpolator1D(z_on_theta_grid,gds21New_temp,extrap=True)(uniform_zgrid)    
        gds22New_val=interpax.Interpolator1D(z_on_theta_grid,gds22New_temp,extrap=True)(uniform_zgrid)    




        data = {}
        data["ntheta"] = nz
        data["theta_grid"] = uniform_zgrid
        data["grad_alpha_squared"] = gds2New_val
        data["grad_alpha_dot_grad_psi"] = gds21New_val
        data["grad_psi_squared"] = gds22New_val
        data["B"] = bmagNew_val
        data["Bcross_gradB_grad_psi"] = gbdrift0New_val
        data["Bcross_gradB_grad_alpha"] = gbdriftNew_val
        data["Bcross_kappa_grad_psi"] = cvdrift0New_val
        data["Bcross_kappa_grad_alpha"] = cvdriftNew_val
        data["gradpar"] = gardparNew_val
        data["grad_psi"] = grho_val

    
        return cls(**data)
