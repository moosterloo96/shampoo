import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import prodimopy.read as pread
import prodimopy.plot as pplot
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp
from scipy.stats import loguniform
from time import process_time
from scipy.special import expn
from timeout_decorator import timeout
import os
import pickle

# Shampoo main file.
#
# Written and documented by Mark Oosterloo
#
# Version: 15-03-2024

class Model:
    """
    The master class which stores all the information of the full simulation.
    """

    def __init__(self, disk=None, species=["H2O", "CO", "CO2", "CH4", "NH3", "H2S"], disk_folder="../ShampooBackground", parameter_folder=None, verbose=None, migration=None, diffusion=None, collisions=None, trackice=None, debug=False, save_csv=[False, None], phi=None, breakIce=False, legacyIce=False, colEq=False, storage_style=0, readsorption=False, supverbose=False, qTest=False, qTestAds=False, qTestMig=False, activeML=True):
        """
        disk: Instance of Disk which contains the output from ProDiMo. If None a new instance is created.
        monomer: Instance of Monomer. If None a new instance is created.
        species: The molecules which are considered in calculating the composition.
        disk_folder: The folder which contains the ProDiMo output of this model.

        verbose:
            1 - only warings
            2 - warnings and initialization/integration notifications 
            
        migration: switch for radial and vertical migration of the monomers.
        diffusion: if migration is true, this switch determines whether we calculate pure aerodynamic drag or also turbulent diffusion.
        collisions: switch for the coagulation module
        trackice: switch for the condensation/evaportation of ices
        debug: debug mode; currently unused
        save_csv: if save_csv[0]==True we regularize the output and write it to a .csv. Data is taken at at the times (in kyr) given in array-like save_csv[1].        
        """
        
        if supverbose:
            self.supverbose = True
        else:
            self.supverbose = False
        
        if not self.supverbose:
            self.print_titlecard()

            print("Attempting to load parameters from folder: ",parameter_folder)
        self.paraDict = self.loadPara(parameter_folder=parameter_folder)
        
        if not self.supverbose:
            print("Attempt succesful!")

        if self.supverbose:
            self.verbose = -1000
        else:
            if verbose==None:
                self.verbose = int(self.paraDict["verbose"])
            else:
                self.verbose = verbose

        ### Debugging levers
        self.breakIce = breakIce
        self.legacyIce = legacyIce
        self.readsorption = readsorption
        self.qTest = qTest
        self.qTestAds = qTestAds
        self.qTestMig = qTestMig
        
        ### Setting the parameters for the ice integration
        self.integrator = ("LSODA","LSODA") # 0 is exposed, 1 is unexposed
        self.atol = (1e-6, 1e-6) # These are the default values in the integrators.
        self.rtol = (1e-3, 1e-3) # Default values. Used much lower values in planetbox model.
        self.floorval = 1e-22 # the floor value below which we set the derivatives 0. Default is 1e-22   
        self.iceScaleFact = 1e5 # offset factor for numerical scaling in units of mMon.
        
        # Warning flags. Makes sure we print warnings only once.
        self.printIntWarning = False

        
        ### Functionality levers!
        # For dynamics...
        if migration == None:
            self.migration = bool(self.paraDict["migration"]) # DYNAMICS
        else:
            self.migration = migration
        
        # ...collisions...
        if collisions == None:
            self.collisions = bool(self.paraDict["collisions"]) # KINETICS
        else:
            self.collisions = collisions
        
        # and ice evolution!
        if trackice == None:
            self.trackice = bool(self.paraDict["trackice"])
        else:
            self.trackice = trackice
        
        try:
            self.colEq = bool(self.paraDict["col_eq"])
        except:
            self.colEq = colEq
        try:
            self.activeML = bool(self.paraDict["activeML"])
        except:
            self.activeML = activeML
            
        self.printStatistics = bool(self.paraDict["print_stat"])
        if self.verbose>-1:
            tic = process_time()
            print("Initializing model")

        self.diskFolder = disk_folder
        self.debug = debug
        self.save_csv = save_csv[0]
        self.eval_csv = save_csv[1]

        self.pisoBenchmark = bool(self.paraDict["piso"]) 
        self.cieslaBenchmark = bool(self.paraDict["ciesla"])
        
        if disk==None:
            self.disk = Disk(species=species, folder=self.diskFolder)
        else:
            self.disk = disk
            self.diskFolder = disk.diskFolder
        
        if self.trackice:
            self.iceNum = len(self.disk.iceList)
        else:
            self.iceNum = 0

        ### Storage management
        try:
            self.store=int(self.paraDict["storage_style"])
        except:
            self.store = storage_style # Do we want to keep track of the local environment of the monomer/home aggregate?
        
        if self.store==0:
            self.trackListEnv = ["rhog", "rhod", "Tg", "Td", "chiRT", "nd"]
            self.trackListFun = ["St", "iceRates", "delta_t"]        
            
        self.initPara(filepath=self.diskFolder)

        if self.migration:
            self.diffusion = bool(self.paraDict["diffusion"]) 
        else:
            self.diffusion = False

        if self.pisoBenchmark:
            self.constDelP = False
            self.testTrueDelP = False        
        else:
            self.constDelP = False
            self.testTrueDelP = True # Use simplification on the calculation of pressure gradient if false

        self.constTimestep = [bool(self.paraDict["const_delt"]), float(self.paraDict["const_delt_val"])]
                     
        self.deterministic = bool(self.paraDict["deterministic"])
        
        self.fixR = bool(self.paraDict["fixR"])
        self.fixZ = bool(self.paraDict["fixZ"])
        
        if not self.migration:
            self.fixR = True
            self.fixZ = True

            
        self.innerRdraw = float(self.paraDict["rInnerDraw"]) # inner radial sampling distance in AU if monomer initial positions are randomly picked.    
        try:
            self.outerRdraw = float(self.paraDict["rOuterDraw"]) # outer radial sampling distance in AU if monomer initial positions are randomly picked.    
        except:
            if self.verbose>1:
                print("No outer drawing distance found, using tapering radius instead...")
            self.outerRdraw = self.para["Rtaper"]/self.auTOm
            
        self.xi = 1/3 # numerical parameter related to the random turbulent kicks

        self.feps = float(self.paraDict["rat_group"]) # mass ratio where we start to group collision partners
        self.mrat = float(self.paraDict["rat_erode"]) # mass ratio of colliders where erosion and fragmentation transition

        self.ftub = float(self.paraDict["ftub"]) # fraction of the drift/turbulent timescale
        self.fcol = float(self.paraDict["fcol"])  # fraction of the collision timescale
        self.fdyn = float(self.paraDict["fdyn"]) 
        self.fice = float(self.paraDict["fice"]) 
        
        self.phi = phi # Auxiliary parameter which allows for easy changing of the porosity.
        
        self.diffType = "mcfost"
        self.desType = "new" # Use old (Piso+ 2015) or new (Cuppen+ 2017) desorption formalism.
        self.flag = False # The warning flag if our mass change in ice becomes too large.
        self.delta_t_floor = float(self.paraDict["delta_t_floor"]) # Floor to variable timestep in yr.
        
        if self.verbose>-1:
            toc = process_time()
            print("Finished model initialization in {:.2f} CPU s".format(toc-tic))
            if self.verbose>0:
                print("-----Model properties-----")
                props = vars(self)
                for item in props.keys():
                    if item == "para":
                        paradict = props[item]
                        for subitem in paradict.keys():
                            if subitem[0] not in ["m", "E"]:
                                print(subitem, "-----", paradict[subitem])
                    else:
                        print(item, "-----", props[item])

    def print_titlecard(self):
        msg1 = "Welcome to SHAMPOO"
        msg2 = "StocHAstic Monomer PrOcessOr"
        msg3 = "Dynamical, collisional and ice processing since 2022"
        maxLen = len(msg3)
     
        msgList = [msg1, msg2, msg3]
        
        print(" "+"-"*(maxLen+20)+" ")
        print("|"+" "*(maxLen+20)+"|")
        for msg in msgList:
            msgLen = len(msg)
            space = " "*(int((maxLen-msgLen)/2)+10)
            print("|"+space+msg+space+"|")
            print("|"+" "*(maxLen+20)+"|")
            
        print(" "+"-"*(maxLen+20)+" ")

        
        
    def loadPara(self, parameter_folder=None):
        if parameter_folder!=None:
            path = parameter_folder
        else:
            path = ""
            
        paraDict = pd.read_csv(path+"shampooInput.csv", header=None, index_col=0, usecols=(0,1), squeeze=True, comment="#", delimiter="\t", skipinitialspace=True)
        for item in paraDict.keys():
            if paraDict[item]=="True":
                paraDict[item] = True
            elif paraDict[item]=="False":
                paraDict[item] = False
    
        return paraDict
            
    
    def initPara(self, filepath):
        """
        Initializes the fundamental parameters associated with a model.
        """

        self.para = {}

        # Natural constants
        self.para["kB"] = 1.38064852e-23
        self.para["Rg"] = 8.314
        self.para["u"] = 1.660539067e-27
        self.para["NA"] = 6.02214e23
        self.para["mp"] = 1.6726219e-27
        self.para["G"] = 6.67408e-11
        self.para["Msun"] = 1.9884e30
        
        # Element masses
        self.para["mElements"] = 1e-3*np.array([1.00797, 12.011, 14.0067, 15.9994, 32.06])
        # molar masses in kg/mol for the elements H, C, N, O and S

        # Unit conversion factors	
        self.sTOyr = 3600*24*365.25
        self.sTOMyr = self.sTOyr*1e6 
        self.auTOm = 1.496e11

        # Adsorption energies and masses
        dataName = np.loadtxt(filepath+"/AdsorptionEnergies.in", dtype="str", comments="#", usecols=(0), encoding=None)
        dataNum = np.loadtxt(filepath+"/AdsorptionEnergies.in", dtype="float", comments="#", usecols=(1,2), encoding=None)
        N = len(dataName)
        for n in range(N):
            self.para["Eads"+dataName[n]] = dataNum[n,0]*self.para["kB"]
            self.para["m"+dataName[n]] = dataNum[n,1]*self.para["u"]
            
        if self.pisoBenchmark:
            self.para["EadsH2O"] = 5800*self.para["kB"]
        
        # Ice formation properties
        if self.pisoBenchmark:
            self.para["NadsRef"] = 1e19
        else:
            self.para["NadsRef"] = float(self.paraDict["NadsRef"])
        
        # Molecule sticking properties; see He+ 2016. 
        self.para["alphaStick"] = float(self.paraDict["alphaStick"])
        self.para["betaStick"] = float(self.paraDict["betaStick"])
        self.para["gammaStick"] = float(self.paraDict["gammaStick"])
        

        # Disk model parameters - Assigned from self.disk.parameter; we here do the dtype and unit conversions.
        # This is needed since the params-object from prodimopy has all data from parameter.out in the form of strings and cgs units.
        self.para["a_settle"] = float(self.disk.parameters["A_SETTLE"]) #the turbulent viscosity/dust settling parameters
        self.para["Mstar"] = float(self.disk.parameters["MSTAR"])/1e3 # mass of central star in kg
        self.para["Rtaper"] = float(self.disk.parameters["RTAPER"])/1e2 # tapering radius 

        # Dust parameters 
        self.para["nsize"] = len(self.disk.model.dust.asize) # number of mass bins
        self.para["rho_mat"] = 2094 # Material density of the background dust # FIND THIS BACK IN PARAMETER.OUT!!!!

        # Scale height/flaring properties
        self.para["H0"] = float(self.disk.parameters["MCFOST_H0"])*self.auTOm
        self.para["r0"] = float(self.disk.parameters["MCFOST_RREF"])*self.auTOm
        self.para["beta_flare"] = float(self.disk.parameters["MCFOST_BETA"])

        # Inner and outer disk radii (in meters)
        self.para["r_inner"] = self.disk.model.x[0,0]*self.auTOm
        self.para["r_outer"] = self.disk.model.x[-1,0]*self.auTOm
        self.para["r_inner_cutoff"] = float(self.paraDict["r_inner_cutoff"])

        # Chemical parameters
        self.para["sticking"] = 1 # the sticking factor.
        self.para["sig_mol"] = 2e-19 # Molecular cross section, taken from Krijt+ 2018 / Okuzumi+ 2012
        self.para["ds"] = 1e-10 # Thickness of a single ice monolayer (see Oosterloo+ 2023)
        self.para["Ediff/Eads"] = 1/2 # Ratio between diffusion energy barrier and binding energy
        self.para["tauMin"] = 1*self.sTOyr      # Minimum and maximum diffusion timescales.
        self.para["tauMax"] = 1e7*self.sTOyr # Maximum value for MFP diffusion timescale (convert locally to sAgg timescale).
        self.para["tauMinInt"] = 1*self.sTOyr      # Dummy variables to be updated
        self.para["tauMaxInt"] = 1e7*self.sTOyr 
        if self.pisoBenchmark:
            self.para["muH"] = 2.35
        else:
            self.para["muH"] = 2.3 # Informed from Krijt+ 2018
        self.oldNu = False # Use the desorption attempt frequence from Tielens+ 198x
        
        
        # Radiative properties
        self.para["FDraine"] = 1.921e12 # Draine flux of UV photons between 91.2 nm and 205 nm in m-2s-1 Taken from Woitke+ 2009

        # Coagulation properties
        self.grainSizes = self.disk.model.dust.asize*1e-6 # convert to SI
        self.para["v_frag"] = float(self.paraDict["v_frag"]) # Fragmentation velocity (value taken from Krijt & Ciesla 2016). 1 m/s might be better (Guttler et al. 2010; Birnstiel et al. 2010)
        self.para["del_v_frag"] = self.para["v_frag"]/5
        self.para["x_frag"] = float(self.paraDict["x_frag"]) # the power law slope for collision fragments (value from Birnstiel+ 2010)
        
    def initGradients(self, size=None):
        """
        Initializes the gradients required for turbulent diffusion formalism.
        """

        toc = process_time()

        rVals, zVals = self.disk.model.x*self.auTOm, self.disk.model.z*self.auTOm
        quant = self.calculateDiffusivity(rVals, zVals, size=size)
        self.diffusivity = quant
        self.disk.data["dDdzSq"] = self.calculateDerivative(quant, "z")
        self.disk.data["dDdrSq"] = self.calculateDerivative(quant, "r")

        quant = self.disk.model.rhog*1e3
        self.disk.data["drhogdzSq"] = self.calculateDerivative(quant, "z")
        self.disk.data["drhogdrSq"] = self.calculateDerivative(quant, "r")

        self.disk.data["dDdz"] = self.disk.data["dDdzSq"]
        self.disk.data["dDdr"] = self.disk.data["dDdrSq"]
        self.disk.data["drhogdz"] = self.disk.data["drhogdzSq"]
        self.disk.data["drhogdr"] = self.disk.data["drhogdrSq"]

        for name in ["dDdz", "dDdr", "drhogdz", "drhogdr"]:
            self.disk.interpol[name] = RectBivariateSpline(self.disk.rVals, self.disk.zVals, self.disk.data[name], kx=self.disk.order, ky=self.disk.order)

        tic = process_time()

    
    def initDensities(self, prodimo=False):
        """
        Prepares the spatially interpolated number and mass densities for all mass bins considered in the model.
        """
        
        r = self.disk.model.x
        z = self.disk.model.z/self.disk.model.x
        
        M = len(self.grainSizes)
        R = self.disk.model.nx
        Z = self.disk.model.nz
        
        for m in range(M):
            if self.verbose>-99:
                print("Calculating size "+str(m+1)+"/"+str(M), end="\r")
            size = self.grainSizes[m] # convert from micron to SI
  
            self.probeEnvironment(r, z*r, 0, inside_loop=False)
            Hp = self.calculateScaleHeight(r*self.auTOm, method="mcfost", kind="dust", size=size)/(r*self.auTOm)
            expArg  = -(z**2)/(2*Hp**2)
            
            sigmaa = self.disk.model.dust.sigmaa[m,:]*1e4/1e3 # convert from g/cm2 to kg/m2 
            sig2D = np.zeros((R,Z))
            for zNo in range(Z):
                sig2D[:,zNo] = sigmaa
            
            N = 2*sig2D/(np.sqrt(2*np.pi)*Hp*r*self.auTOm) # We multiply as sigmaa is only 1/2 of the total surface
            # density
            mda = N*np.exp(expArg)
            
            ma = self.para["rho_mat"]*4/3*np.pi*(size)**3
            nda = mda/ma

            rhoName = "rhoda"+str(m)
            numName = "numda"+str(m)
                        
            # Make sure we add a floor
            minval = 1e-20
            nda[nda<minval] = minval
            mda[mda<minval] = minval*1e-30
            
            self.disk.data[rhoName] = np.log10(mda)
            self.disk.data[numName] = np.log10(nda)
            self.disk.interpol[rhoName] = RectBivariateSpline(self.disk.rVals, self.disk.zVals, self.disk.data[rhoName], kx=self.disk.order, ky=self.disk.order)
            self.disk.interpol[numName] = RectBivariateSpline(self.disk.rVals, self.disk.zVals, self.disk.data[numName], kx=self.disk.order, ky=self.disk.order)
            self.environment = None
      
    
    def initAggregate(self, t_in, r_in, z_in):
        
        self.redo_sAgg(t_in, r_in, z_in)
        self.redo_zMon(t_in, r_in, z_in)
        
    
    
    def redo_sAgg(self, t_in, r_in, z_in):
        """
        Auxiliary function to simulate collisional equilibrium. Draws the home aggregate size based off the background dust mass
        distribution.
    
        r_in - Initial radial monomer position in AU
        z_in - Initial vertical monomer position in AU
        """

        M = len(self.grainSizes)
        compDensity = np.zeros(M)

        for m in range(M):
            rhoName = "rhoda"+str(m)
            compDensity[m]=self.evaluateQuant(rhoName, r_in*self.auTOm, z_in*self.auTOm)

        probs = compDensity/(np.sum(compDensity))

        sAgg =  np.random.choice(self.grainSizes, size=None, replace=True, p=probs)
        self.monomer.homeAggregate.prop["sAgg"] = sAgg
        self.monomer.homeAggregate.prop["mAgg"] = 4/3*np.pi*self.monomer.homeAggregate.prop["rhoAgg"]*sAgg**3

    
    def redo_zMon(self, t_in, r_in, z_in):
        """
        Auxiliary function for initAggregate, redetermines the initial monomer depth, which usually is set to 0. Is called 
        after the new initial home aggregate size has been calculated by redo_sAgg. Also determines whether the monomer
        starts exposed.
        
        r_in - Initial radial monomer position in AU
        z_in - Initial vertical monomer position in AU
        """
        
        self.monomer.prop["zMon"] = self.determineRandomPos()
        
        
  
        if self.monomer.prop["zMon"]<=self.monomer.prop["zCrit"]:
            self.monomer.exposed = True
        else:
            pExp = self.determinePExp()
            if np.random.rand()<=pExp: 
                self.monomer.exposed = True
            else:
                self.monomer.exposed = False
            
            self.seedNo += 1
            np.random.seed(self.seedNo)
        

    
    
    def initIces(self, t_in, r_in, z_in):
        """
        Initiates the data structures in which we store the ice abundances of the monomer.
        """
        self.monomer.ice_sol = {}
        self.monomer.atom_sol = {}
        
        self.probeEnvironment(r_in, z_in, t_in, inside_loop=False)
        r, z, t = self.unpackVars(r_in, z_in, t_in) # Convert input from AU and kyr to SI units.
            
        # Calculate initial individual and total ice budget on the monomer.    
        iceTot = 0
        for n in range(self.iceNum):
            rhoDust = self.environment["rhod"] # Gives the dust mass density (kg/m3)
            rhoIce = self.environment["iceAbun"+self.disk.iceList[n]]*self.para["m"+self.disk.iceList[n]]
            mIce = rhoIce/rhoDust*self.monomer.prop["mMon"] # Ice mass in kg 
            if self.pisoBenchmark:
                if self.disk.iceList[n]=="H2O":
                    mIce = self.monomer.prop["mMon"]
                else:
                    mIce = 0                
            
            self.monomer.ice_sol[self.disk.iceList[n]] = [mIce]
            iceTot += mIce
        
        self.monomer.iceTot_sol = [iceTot] # In kg!
            
        # Calculate the desorption/adsorption rates.
        
        if self.store==0 and self.legacyIce:
            for n in range(self.iceNum):
                self.monomer.ice_sol["ads"+self.disk.iceList[n]] = [self.rateAdsorption(t, r, z, self.disk.iceList[n], init=True)]
                self.monomer.ice_sol["tds"+self.disk.iceList[n]] = [self.rateDesorptionThermal(t, r, z, self.disk.iceList[n], init=True)]
                self.monomer.ice_sol["pds"+self.disk.iceList[n]] = [self.rateDesorptionPhoto(t, r, z, self.disk.iceList[n], init=True)]
                self.monomer.ice_sol["qFactor"+self.disk.iceList[n]] = [np.nan]
    
    
    def evaluateQuant(self, name, r, z):
        # r & z are in m.
        if self.verbose>2:
            print("Interpolating "+name)

        tic = process_time()

        rEva = r/self.auTOm
        
        zEva = abs(z/r)

        if (name in ["rhog", "rhod", "nd", "chiRT"] or (name[0:3] in ["gas", "ice", "tot"]) or (name[0:5] in ["rhoda", "numda"]) or ("Abun" in name)):
                        
            quant = 10**(self.disk.interpol[name](rEva, zEva, grid=False)) 
            if (self.pisoBenchmark)and(name=="rhog"):
                H = 1/self.Omega(0,r,0/r)*np.sqrt(self.para["kB"]*120*(rEva)**(-3/7)/(self.para["muH"]*self.para["mp"]))
                quant =  20000*((rEva)**(-1))/(H*np.sqrt(2*np.pi))
  
        elif name in ["dDdz", "drhogdz"]:
            if isinstance(z, (np.floating, float)):
                if z<0:
                    quant = -self.disk.interpol[name](rEva, zEva, grid=False)
                else:
                    quant = self.disk.interpol[name](rEva, zEva, grid=False)
            else:
                quant = self.disk.interpol[name](rEva, zEva, grid=False)
                quant[zEva<0] *= -1   
            
        else:
            if isinstance(zEva, (np.floating, float)):
                quant = float(self.disk.interpol[name](rEva, zEva, grid=False))	
            else:
                quant = self.disk.interpol[name](rEva, zEva, grid=False)	
            if (self.pisoBenchmark)and(name in ["Tg","Td"]):
                quant = quant/quant*120*(rEva)**(-3/7)
            elif (self.pisoBenchmark)and(name in ["soundspeed", "soundspeed0"]):
                T = quant/quant*120*(rEva)**(-3/7)
                quant = np.sqrt(self.para["kB"]*T/(self.para["muH"]*self.para["mp"]))
            elif (self.cieslaBenchmark)and(name=="soundspeed"):
                quant = self.disk.interpol["soundspeed"](rEva, 0*zEva, grid=False)	

        toc = process_time()
        if self.verbose>2:	
            print("Interpolation complete: Elapsed CPU time is {:.2f} s".format(toc-tic))

        return quant

    
    def calculateDiffusivity(self, rVals, zVals, size=None):
        """
        Auxiliary function for calculating a grid of diffusivities outside the "environment"-framework.
        """

        # Evaluate the soundspeed--------
        soundspeed = self.disk.data["soundspeed"]
        soundspeed0 = np.zeros(soundspeed.shape)
        for i in range(0, len(soundspeed[0,:])):
            soundspeed0[:,i] = soundspeed[:,0]
        # And gas density
        gasDensity = self.disk.data["rhog"]

        # Calculate omega--------
        d = np.sqrt(rVals**2+zVals**2)
        omega = np.sqrt(self.para["G"]*self.para["Mstar"]/d**3)

        # Calculate the mean free path
        den = np.sqrt(2)*gasDensity/(self.para["muH"]*self.para["u"])*self.para["sig_mol"]

        mfp = 1/den

        # Calculate the stokesNumber--------
        if size==None:
            size = self.monomer.homeAggregate.prop["sAgg"]
            density = self.monomer.homeAggregate.prop["rhoAgg"]
        else:
            density = 2094

        cond = size>9*mfp/4

        num = 4*density*size**2
        den = 9*gasDensity*soundspeed*mfp
        stokesSt = np.sqrt(np.pi/8)*num/den*omega
        epsteiSt = np.sqrt(np.pi/8)*size*density/(soundspeed*gasDensity)*omega
        stokesNumber = np.where(cond, stokesSt, epsteiSt)	

        gDiffusivity = self.para["a_settle"]*soundspeed*self.calculateScaleHeight(rVals, method="mcfost", kind="gas", size=None)

        pDiffusivity = gDiffusivity/(1+stokesNumber**2)

        self.diffusivity=pDiffusivity
        
        return pDiffusivity

    
    def calculateScaleHeight(self, r, method="mcfost", kind="gas", size=None):
        """
        r: Distance from star in m.
        Calculates the scale height in m.
        """
        # Alternatively, we calculate the pressure scale height
        if method=="pressure":

            soundspeed = self.environment["soundspeed0"]

            omega = self.Omega(0,r,0/r)

            H = soundspeed/omega

        elif method=="mcfost":

            H = self.para["H0"]*(r/self.para["r0"])**self.para["beta_flare"]

            
        if (kind=="dust" or kind=="dustProdimo"):
            fact = np.sqrt(self.para["a_settle"]/(np.sqrt(3)*self.Stokes(0, r, 0, size=size, midplane=True)+self.para["a_settle"])) # Using the expression of Youdin & Lithwick 2007
            H *= fact
        return H

    
    def calculateDerivative(self, quant, type):
        """
        Calculates the spatial gradients in quant, which is of the same shape as the ProDiMo model output.
        """

        # dDp/dz

        rVals, zVals = self.disk.model.x*self.auTOm, self.disk.model.z*self.auTOm

        form = self.disk.model.x.shape
        if type=="z":

            dQdZ = np.zeros(form)

            # Convert to log-space for numerical stability.
            dQdZ1 = np.log(quant[:,0:-1])
            dQdZ2 = np.log(quant[:,1::])

            dQdZint = (dQdZ2-dQdZ1)/(zVals[:,1::]-zVals[:,0:-1])

            a1 = (zVals[:,1:-1]-zVals[:,0:-2])/(zVals[:,2::]-zVals[:,0:-2])
            a2 = (zVals[:,2::]-zVals[:,1:-1])/(zVals[:,2::]-zVals[:,0:-2])

            dQdZ[:,1:-1] = a1*dQdZint[:,0:-1] + a2*dQdZint[:,1::]

            return dQdZ*quant

        elif type=="r":
            dQdR = np.zeros(form)

            dQdR1 = np.log(quant[0:-1,:])
            dQdR2 = np.log(quant[1::,:])


            dQdRint = (dQdR2-dQdR1)/(rVals[1::,:]-rVals[0:-1,:])

            a1 = (rVals[1:-1,:]-rVals[0:-2,:])/(rVals[2::,:]-rVals[0:-2,:])
            a2 = (rVals[2::,:]-rVals[1:-1,:])/(rVals[2::,:]-rVals[0:-2,:])

            dQdR[1:-1,:] = a1*dQdRint[0:-1,:] + a2*dQdRint[1::,:]

            return dQdR*quant

#################################################################################################
### Dynamics: Auxiliary Functions.
#################################################################################################

    # Auxiliary chemical functions

    
    def meanFreePath(self, t, r, z):

        gasDensity = self.environment["rhog"]

        den = np.sqrt(2)*gasDensity/(self.para["muH"]*self.para["u"])*self.para["sig_mol"]

        mfp = 1/den

        return mfp

    
    def thermalGasVelocity(self, t, r, z, species="H2", gasT=None):

        if type(gasT)==type(None):
            gasTemperature = self.environment["Tg"]
        else:
            gasTemperature = gasT

        num = self.para["kB"]*gasTemperature
        den = 2*np.pi*self.para["m"+species]

        velocity = np.sqrt(num/den)

        return velocity	
    
    
    def Omega(self, t, r, z):

        d = np.sqrt(r**2+z**2)

        omega = np.sqrt(self.para["G"]*self.para["Mstar"]/d**3)

        return omega

    
    def nuTur(self, t, r, z, midplane=False):
        """
        Calculates the turbulent viscosity as nuT = alpha*cs**2/omega
        """
        if midplane:
            soundspeed = self.environment["soundspeed0"]
        else:
            soundspeed = self.environment["soundspeed"]

        omega = self.Omega(t, r, z)

        nuT = self.para["a_settle"]*soundspeed*self.calculateScaleHeight(r, method="mcfost", kind="gas", size=None)

        return nuT

    
    def nuMol(self, t, r, z):
        """
        Calculates the molecular viscosity using the expression givin in Krijt+ 2018
        """

        vThermal = np.sqrt(8/np.pi)*self.environment["soundspeed"]

        mfp = self.meanFreePath(t, r, z)
        
        nuM = vThermal*mfp/2

        return nuM

    
    def pressureGradient(self, t, r, z):

        if self.constDelP:
            eta = 1e-3 
        elif self.testTrueDelP:
            soundspeed =  self.environment["soundspeed"]
            gradient = self.environment["drhogdr"]
            gasDensity = self.environment["rhog"]
            
            factor1 = 1/(2*r*gasDensity)
            factor2 = (soundspeed/self.Omega(t,r,z))**2
            
            eta = -factor1*factor2*gradient
        elif self.pisoBenchmark: # Use calculation from Piso+ 2015
            soundspeed = self.environment["soundspeed"]
            
            num = soundspeed**2
            den = 2*(r*self.Omega(t,r,z))**2
            
            eta = num/den     
            
        else:

            soundspeed = self.environment["soundspeed"]
            eta = (soundspeed/(r*self.Omega(t,r,z)))**2

        return eta

    
    def ReynoldsT(self, t, r, z, midplane=False):
        """
        Calculates the turbulence Reynolds number as the ratio between the turbulent and molecular viscosity.
        """

        nuT = self.nuTur(t, r, z, midplane=midplane)
        nuM = self.nuMol(t, r, z)

        reynoldsNumber = nuT/nuM

        return reynoldsNumber

    
    def ReynoldsP(self, t, r, z, midplane=False, size=None):

        if size is None:
            size = self.monomer.homeAggregate.prop["sAgg"]


        num = 4*size*self.velocity_r(t, r, z, size=size)
        den = self.thermalGasVelocity(t, r, z, species="H2")*self.meanFreePath(t, r, z)

        reynoldsNumber = num/den

        return reynoldsNumber

    
    def Stokes(self, t, r, z, size=None, rhoMat=2094, midplane=False):

        if size is None:
            size = self.monomer.homeAggregate.prop["sAgg"]
            rhoMat = self.monomer.homeAggregate.prop["rhoAgg"]            

        if midplane:
            soundspeed = self.evaluateQuant("soundspeed", r, 0*r)
            gasDensity = self.evaluateQuant("rhog", r, 0*r)
        else:
            soundspeed = self.environment["soundspeed"]
            gasDensity = self.environment["rhog"]

        omega = self.Omega(t, r, z)

        mfp = self.meanFreePath(t, r, z)

        cond = size>9*mfp/4

        try:
            if cond:
                if self.verbose>0:
                    print("Home aggregate is in the Stokes regime")
                num = 4*rhoMat*size**2
                den = 9*gasDensity*soundspeed*mfp
                stokesNumber = np.sqrt(np.pi/8)*num/den*omega
            else:
                stokesNumber = np.sqrt(np.pi/8)*size*rhoMat/(soundspeed*gasDensity)*omega
        except:
            num = 4*rhoMat*size**2
            den = 9*gasDensity*soundspeed*mfp
            stokesSt = np.sqrt(np.pi/8)*num/den*omega
            epsteiSt = np.sqrt(np.pi/8)*size*rhoMat/(soundspeed*gasDensity)*omega
            stokesNumber = np.where(cond, stokesSt, epsteiSt)	

        
            
        return stokesNumber

    
    def Schmidt(self, t, r, z):

        schmidtNumber = 1+self.Stokes(t,r,z)**2

        return schmidtNumber

    
    def gasDiffusivity(self, t, r ,z):
        """
        Calculated according to Krijt+ 2018 and Ciesla 2010
        """
        soundspeed = self.environment["soundspeed"] # maybe do this at midplane?

        # Note that we here implicitly calculate the pressure scale height as cs/omega
        gDiffusivity = self.para["a_settle"]*soundspeed*self.calculateScaleHeight(r, method="mcfost", kind="gas")

        return gDiffusivity

    
    def particleDiffusivity(self, t, r, z):

        gDiffusivity = self.gasDiffusivity(t,r,z)
        stokesNumber = self.Stokes(t,r,z)

        pDiffusivity = gDiffusivity/(1+stokesNumber**2)
        
        return pDiffusivity

    
    def velocity_r(self, t, r, z, size=None, stokes=None):

        if stokes==None:
            stokesNumber = self.Stokes(t,r,z, size)
        else:
            stokesNumber = stokes
        v_r = -2*self.pressureGradient(t,r,z)*r*self.Omega(t,r,z)*stokesNumber/(1+stokesNumber**2)
    
        return v_r

    
    def velocity_z(self, t, r, z, size=None, stokes=None):

        if stokes==None:
            stokesNumber = self.Stokes(t,r,z, size)
        else:
            stokesNumber = stokes
        v_z = -self.Omega(t,r,z)*z*stokesNumber

        return v_z

    
    def velocity_eff_r(self, t, r, z):

        v_r = self.velocity_r(t,r,z)
        term1 = self.environment["dDdr"]

        gasDensity = self.environment["rhog"]

        term2a = self.particleDiffusivity(t,r,z)/gasDensity

        term2b = self.environment["drhogdr"]

        v_eff_r = v_r + term1 + term2a*term2b

        return v_eff_r

    
    def velocity_eff_z(self, t, r, z):

        v_z = self.velocity_z(t,r,z)
        term1 = self.environment["dDdz"] # Krijt 2018: term1 = 0

        gasDensity = self.environment["rhog"]

        term2a = self.particleDiffusivity(t,r,z)/gasDensity
        # And again...
        term2b = self.environment["drhogdz"]

        v_eff_z = v_z + term1 + term2a*term2b

        return v_eff_z


#################################################################################################
### Dynamics: Functions for calculating the particle displacement.
#################################################################################################

    
    def z_der(self, r_in, z_in, t_in, rand):

        r, z, t = self.unpackVars(r_in, z_in, t_in)

        if self.diffusion==True:
            randZ = rand*np.sqrt(2/self.xi*self.particleDiffusivity(t,r,z)*self.delta_t)
            z_der = self.velocity_eff_z(t,r,z)*self.delta_t + randZ
        elif self.diffusion==False:
            z_der = self.velocity_z(t,r,z)*self.delta_t

        z_der /= self.auTOm # convert from m to AU
        
        if self.fixZ:
            z_der *= 0
        return z_der

    
    def r_der(self, r_in, z_in, t_in, rand):

        r, z, t = self.unpackVars(r_in, z_in, t_in)

        if self.diffusion==True:
            randR = rand*np.sqrt(2/self.xi*self.particleDiffusivity(t,r,z)*self.delta_t)
            r_der = self.velocity_eff_r(t,r,z)*self.delta_t + randR

        elif self.diffusion==False:
            r_der = self.velocity_r(t,r,z)*self.delta_t

        r_der /= self.auTOm # convert from m to AU

        if self.fixR:
            r_der *= 0

        return r_der 
    
    
#################################################################################################
### Coagulation: Auxiliary Functions. All units in SI.
#################################################################################################    
    
    
    def vRelBrown(self, t, r, z):
        """
        Calculates difference in Brownian motion.
        """
    
        mLarge = self.para["mLarge"]
        mSmall = self.para["mSmall"]
    
        num = 8*self.para["kB"]*self.environment["Td"]*(mLarge+mSmall)
        den = np.pi*mLarge*mSmall
    
        vBM = np.sqrt(num/den)
    
        return vBM

    
    def vRelDrift(self, t, r, z):
        """
        Calculates differential drift relative velocity.
        """
    
        v1 = self.velocity_r(t, r, z, size=self.para["sLarge"], stokes=self.vrel_Stlarge)
        v2 = self.velocity_r(t, r, z, size=self.para["sSmall"], stokes=self.vrel_Stsmall)
    
        vRD = abs(v1-v2)
    
        return vRD, v1, v2

    
    def vRelSettling(self, t, r, z):
        """
        Calculates differential settling relative velocity.
        """
    
        v1 = self.velocity_z(t, r, z, size=self.para["sLarge"], stokes=self.vrel_Stlarge)
        v2 = self.velocity_z(t, r, z, size=self.para["sSmall"], stokes=self.vrel_Stsmall)
    
        vZD = abs(v1-v2)
    
        return vZD

    
    def vRelAzimuthal(self, t, r, z, v1r, v2r):
        """
        Calculates the relative velocity contribution due to differences in azimuthal velocity.
        """
    
        v1 = v1r/(2*self.vrel_Stlarge)
        v2 = v2r/(2*self.vrel_Stsmall)
    
        vAD = abs(v1-v2)
    
        return vAD


    
    def vRelTurbulent(self, t, r, z):
        """
        Calculates the relative velocity difference due to turbulence using the approximations presented 
        by Ormel & Cuzzi 2007
        """
        omega = self.Omega(t, r, z)
        Re = self.ReynoldsT(t, r, z, midplane=True)
    
        tstop = self.vrel_Stlarge/omega
        omg = 1/omega #== tL when following Ormel & Cuzzi 2007.
        teta = omg/np.sqrt(Re) # tL = 1/omega
        

        if tstop<=teta: 
            Q = (Re)**0.25*abs(self.vrel_Stlarge-self.vrel_Stsmall)
        elif tstop>=omg:
            Q = np.sqrt(1/(1+self.vrel_Stlarge)+1/(1+self.vrel_Stsmall))
        else:
            Q = 1.55*np.sqrt(self.vrel_Stlarge)  

        vTM = np.sqrt(self.para["a_settle"])*self.environment["soundspeed0"]*Q
    
        return vTM


    
    def calcTotVRel(self, t, r, z, doPrint=False):
        """
        Calculate relative velocities for a home aggregate of a certain size s1 with a collision partner of size s2.
        """

        sagg = self.monomer.homeAggregate.prop["sAgg"]
        scol = self.monomer.homeAggregate.prop["sCol"]
    
        if scol>sagg:
            denSmall = 2094
            denLarge = 2094
            self.para["sLarge"] = scol
            self.para["sSmall"] = sagg 
        else:
            denSmall = 2094
            denLarge = 2094
            self.para["sLarge"] = sagg
            self.para["sSmall"] = scol
            
        self.para["mSmall"] = 4/3*np.pi*denSmall*self.para["sSmall"]**3#magg
        self.para["mLarge"] = 4/3*np.pi*denLarge*self.para["sLarge"]**3#mcol
        
        if doPrint:
            print("Properties passed:")
            print("sagg: ",sagg, "scol:", scol)
    
        self.vrel_Stlarge = self.Stokes(t, r, z, size=self.para["sLarge"], rhoMat=denLarge)
        self.vrel_Stsmall = self.Stokes(t, r, z, size=self.para["sSmall"], rhoMat=denSmall)
    
        v1 = self.vRelBrown(t, r, z)
        v2, vr1, vr2 = self.vRelDrift(t, r, z)
        v3 = self.vRelSettling(t, r, z)
        v4 = self.vRelAzimuthal(t, r, z, vr1, vr2)
        v5 = self.vRelTurbulent(t, r, z)
    
        v_tot = np.sqrt(v1**2+v2**2+v3**2+v4**2+v5**2)
    
    
        if doPrint:
            print("relative velocity: ",v_tot,"m/s")
            
        if (self.debug)and(self.monomer.homeAggregate.prop["sAgg"]>self.grainSizes[-1]):
            print("Relative velocity report:")
            print("sagg: ",sagg, "scol:", scol, "magg:", magg, "mcol:", mcol)
            print("Brownian:", v1)
            print("Radial:", v2)
            print("Settling:", v3)
            print("Azimuthal:", v4)
            print("Turbulence:",v5)
            print("Total: ",v_tot)
            print("Largest:", self.para["sLarge"], "Smallest:", self.para["sSmall"])
            print("t=",t,"r=",r,"z=",z)
    
           
    
        return v_tot 
    
        # Collision rates
    
    
    def sigCol(self, t, r, z, s=None):
        """
        Calculates the collisional cross section between the home aggregate and a particle of size s.
        """
    
        if s is None:
            s = self.monomer.homeAggregate.prop["sCol"]    
    
        sig_col = np.pi*(self.monomer.homeAggregate.prop["sAgg"]+s)**2
    
        return sig_col


    
    def collisionRate(self, t, r, z, s=None):
    
        if s is None:
            s = self.monomer.homeAggregate.prop["sCol"]
    
        name = "numda"+str(np.argmin(abs(self.grainSizes-s)))

        nDen = self.evaluateQuant(name, r, z)

        vRel = self.calcTotVRel(t, r, z)

        
        sig = self.sigCol(t, r, z)
        
        if self.verbose>3:
            print("nDen = {:.2e} /m3".format(nDen), "vRel = {:.2e} m/s".format(vRel), "sig = {:.2e} m2".format(sig))

        colrate = nDen*vRel*sig

        return colrate

    def calcColRates(self, t_now, r_now ,z_now, size=None):
        """
        Calculates the collision rates for the home aggregate at its given position. Is a key function to be evaluated 
        every time step.
        """

        r,z,t = self.unpackVars(r_now, z_now,t_now)
        sizes = self.grainSizes

        if size==None:
            sAgg = self.monomer.homeAggregate.prop["sAgg"]
        else:
            sAgg = size

        # calculate the unaltered collision rates
        self.colRates = np.zeros(self.para["nsize"])

        for s in range(self.para["nsize"]):
            self.monomer.homeAggregate.prop["sCol"] = sizes[s]
            self.colRates[s] = self.collisionRate(t, r, z)
           
        # determine whether effective cross section is needed
        volFact = (sizes/sAgg)**3

        effTrue = volFact/self.feps
        effFalse = np.ones(self.para["nsize"])

        self.effectiveFact = np.where(volFact<=self.feps, effTrue, effFalse)

        
        self.effColRates = self.effectiveFact*self.colRates


#################################################################################################
### Ice Formation: Auxiliary Functions. All units in SI.
#################################################################################################    
   
    
    def stickingFactor(self, t, r, z, iceName, dustT=None):
        
        if dustT==None:
            dustT = self.environment["Td"]
            
        tanhArg = self.para["betaStick"]*(dustT-self.para["gammaStick"]*self.para["Eads"+iceName]/self.para["kB"])
        sticking = self.para["alphaStick"]*(1-np.tanh(tanhArg))
        
        return sticking
    
    
    
    def rateAdsorption(self, t, r, z, iceName, init=False, dustT=None, gasT=None, iceList=None):
        """
        Calculates the surface-specific adsorption rate. (kg/m2/s)
        """

        if self.pisoBenchmark:
            rateAds = 0
        else:
            
            nSpecies = self.environment["gasAbun"+iceName]

            if dustT==None:
                dustT = self.environment["Td"]
            vthermal = self.thermalGasVelocity(t, r, z, species=iceName, gasT=gasT)

            sticking = self.stickingFactor(t, r, z, iceName, dustT=dustT)

            rateAds = nSpecies*vthermal*sticking*self.para["m"+iceName] 
 
        return rateAds
    
    
    
    def iceFraction(self, t, r, z, iceName, iceList=None):
        """
        Auxiliary function for the desorption rates to calculate the number fraction of ice species iceName in the ice mantle 
        (# of iceName molecules per total amount of ice molecules).
        """
        #print(iceList)
        if self.legacyIce:
            if (self.monomer.iceTot_sol)[-1]>0:  
                if (self.monomer.ice_sol[iceName])[-1]>0:
                    iceTot = 0
                    for n in range(self.iceNum):
                        iceTot += (self.monomer.ice_sol[self.disk.iceList[n]])[-1]/self.para["m"+self.disk.iceList[n]]
                    fx = ((self.monomer.ice_sol[iceName])[-1]/self.para["m"+iceName])/iceTot
                else:
                    fx = 1
            else:
                fx = 1
        
        else:
            iceTot = 0 # count the total number of ice molecules
            for n in range(self.iceNum):
                iceTot += iceList[self.disk.iceList[n]]/self.para["m"+self.disk.iceList[n]]
            fx = iceList[iceName]/self.para["m"+iceName]/iceTot
        
        return fx
    
    def iceSpots(self, t, r, z, iceList=None):
        """
        Auxiliary function which calculates the total number of ice molecules present on the monomer.
        """
        
        nice = 0
        if self.legacyIce:
            for n in range(self.iceNum):
                nice += (self.monomer.ice_sol[self.disk.iceList[n]])[-1]/self.para["m"+self.disk.iceList[n]]
        else:
            for n in range(self.iceNum):
                nice += iceList[self.disk.iceList[n]]/self.para["m"+self.disk.iceList[n]]
       
        return nice
    
    
    def vibrationFrequency(self, t, r, z, iceName):
        
        nu = None
        
        if self.oldNu:
            fact1 = 1.6e11*np.sqrt(self.para["Eads"+iceName]*self.para["mp"]/(self.para["kB"]*self.para["m"+iceName]))
            fact2 = (self.monomer.prop["Nads"]/self.para["NadsRef"])**(0.5)
            nu = fact1*fact2
        else:
            # The default expression we always want to use.
            num = 2*self.monomer.prop["Nads"]*self.para["Eads"+iceName]
            den = self.para["m"+iceName]*np.pi**2
            nu = np.sqrt(num/den)
        
        return nu
    
    
    
    def rateDesorptionThermal(self, t, r, z, iceName, dustT=None, init=False, fx=None, iceList=None):
        """
        Calculates the surface-specific thermal desorption rate. (kg/m2/s)
        """
        
        if dustT==None:
            dustTemperature = self.environment["Td"]
        else:
            dustTemperature = dustT
            
        if self.pisoBenchmark:
            fact1 = 1.6e11*np.sqrt(self.para["Eads"+iceName]*self.para["mp"]/(self.para["kB"]*self.para["m"+iceName]))
            fact2 = self.monomer.prop["Nads"]*self.para["m"+iceName]

            rateDesT = fact1*fact2*np.exp(-self.para["Eads"+iceName]/(self.para["kB"]*dustTemperature))
        elif self.activeML:
            
            nu = self.vibrationFrequency(t, r, z, iceName)
            
            kdes = nu*np.exp(-self.para["Eads"+iceName]/(self.para["kB"]*dustTemperature))
            
            # We calculate the total number of active spots for the whole monomer
            nact = self.monomer.prop["Nact"]*self.monomer.prop["Nads"]*4*np.pi*(self.monomer.prop["sMon"])**2 
            
            # And compare this to the number of occupied spots on the monomer
            nice = self.iceSpots(t, r, z, iceList=iceList) 
            
            if nice<nact: 
                # In this case there are more active spots than ice molecules on the monomer, and thus desorption
                # is limited by the amount of molecules.
                if self.legacyIce:
                    iceAmount = (self.monomer.ice_sol[iceName])[-1]
                else:
                    iceAmount = iceList[iceName]
                
                # Divide by surface area of monomer to calculate # of spots per m2   
                corfactDen = self.para["m"+iceName]*4*np.pi*(self.monomer.prop["sMon"])**2
                
                nspots = iceAmount/corfactDen
                
            else:
                # In this case desorption is limited by the number of active spots.
                if fx==None:
                    fracIce = self.iceFraction(t, r, z, iceName, iceList=iceList)
                else:
                    fracIce = fx
   
                nspots = self.monomer.prop["Nact"]*self.monomer.prop["Nads"]*fracIce                

            rateDesT = kdes*nspots*self.para["m"+iceName]
            
        else:
            
            nu = self.vibrationFrequency(t, r, z, iceName)
            
            kdes = nu*np.exp(-self.para["Eads"+iceName]/(self.para["kB"]*dustTemperature))
            
            if fx==None:
                fracIce = self.iceFraction(t, r, z, iceName, iceList=iceList)
            else:
                fracIce = fx
            
            nspots = self.monomer.prop["Nact"]*self.monomer.prop["Nads"]*fracIce
        
            rateDesT = kdes*nspots*self.para["m"+iceName]	            
   
        return rateDesT

    
    
    def aggregateMFP(self, t, r, z):
        
        mfp = 4*self.monomer.prop["sMon"]/(3*self.monomer.homeAggregate.prop["phi"])
        
        return mfp
    
    
    
    def moleculeDiffusivity(self, t, r, z, iceName):
        
        # Call on necessary environment properties the diffusivity
        dustTemperature = self.environment["Td"]
        
        # And intermediate quantities
        mfp = self.aggregateMFP(t, r, z)
        sticking = self.stickingFactor(t, r, z, iceName, dustT=dustTemperature)
        nu0 = self.vibrationFrequency(t, r, z, iceName)
                
        numTot = nu0*mfp**2
        
        expArg = self.para["Eads"+iceName]/(self.para["kB"]*dustTemperature)
        term1 = (mfp/self.para["ds"])*np.exp(self.para["Ediff/Eads"]*expArg)
        term2 = sticking*np.exp(expArg)
        
        denTot = term1+term2
        
        diffusivity = numTot/denTot
        
        return diffusivity
        
      
    
    def calcQFactor(self, t, r, z, tau, iceName=None):
        """
        Calculates the q factor in the intermediate tau_diff regime.
        """
        
        qFactor = (np.log10(tau)-np.log10(self.para["tauMinInt"]))/(np.log10(self.para["tauMaxInt"])-np.log10(self.para["tauMinInt"]))        
               
        return qFactor
    
    
    
    def rateDesorptionInternal(self, t, r, z, iceName, iceList=None):
        """
        Alternative function for thermal desorption in non-exposed aggregates, 
        accounting for diffusion in a dirty way. See Oosterloo+ 2023b/2024.
        """
        
        # step 1: Calculate tau
        # step 2: Check whether we are in q=1 or q=0 regime --> 
        # step 3: if neither true, calculate QFactor with timescale approach
        # step 4: calculate thermal desorption rate if q>0
        
        size = self.monomer.homeAggregate.prop["sAgg"]
        tau = self.tauDiff(t, r, z, size, iceName) # note that this is the aggregate diffusion timescale
        
        mfp = self.aggregateMFP(t, r, z)
        
        
        self.para["tauMinInt"] = self.para["tauMin"]*(size/mfp)**2
        self.para["tauMaxInt"] = self.para["tauMax"]*(size/mfp)**2 # convert from mfp to agg timescale
        #print(tau, self.para["tauMinInt"],self.para["tauMaxInt"])
        if tau>self.para["tauMaxInt"]:
            # In this case, diffusion goes very slow.
            qFactor = 1
            rateDesI = 0
        elif tau<self.para["tauMinInt"]:
            # In this case, diffusion goes very fast.
            qFactor = 0
            rateDesI = self.rateDesorptionThermal(t, r, z, iceName, dustT=None, iceList=iceList)
        else:
            # Otherwise we're in the transitional regime where 0<q<1.
            qFactor = self.calcQFactor(t, r, z, tau, iceName)
            #print("q =",qFactor)
            rateDesI = (1-qFactor)*self.rateDesorptionThermal(t, r, z, iceName, dustT=None, iceList=iceList)
        
        
        return rateDesI, qFactor
    
    
    def rateDesorptionPhoto(self, t, r, z, iceName, init=False, fx=None, iceList=None):
        """
        Calculates the surface-specific UV photon desorption rate. (kg/m2/s)
        """

        if self.pisoBenchmark:
            rateDesP = 0
        else:
            chiRT = self.environment["chiRT"]
            if fx==None:
                fracIce = self.iceFraction(t, r, z, iceName, iceList=iceList)
            else:
                fracIce = fx
            
            # 1st order vs 2nd order desorption regime. Volume vs surface. See Woitke+ 2009; active layer concept.
            # yield: Leiden group has done better yield-measurements for H2O. (Start at Öberg+ 2009; Arasa+ 2010).
            # Semenov & Kamp: yield may not be that important.
            rateDesP = fracIce*self.monomer.prop["yield"]*self.para["FDraine"]*chiRT*self.para["m"+iceName]

        return rateDesP
    
    
    def calcIceTot(self):
        
        iceTot = 0
        for n in range(self.iceNum):
            iceTot += (self.monomer.ice_sol[self.disk.iceList[n]])[-1]
        
        return iceTot
        

#################################################################################################
### Operations related to the environment
#################################################################################################

    
    def probeEnvironment(self, r_now, z_now, t_now, inside_loop=True):
        """
        Infers the local disk properties to use in a given timestep. Input in AU and kyr.
        """

        if t_now==0:
            self.environment = {}

        r_in, z_in, t_in = self.unpackVars(r_now, z_now,t_now)

        # For now we calculate the gas/dust density, temperature and soundspeed.
        for cond in ["rhog", "rhod", "Tg", "Td", "nd", "soundspeed", "chiRT", "nd"]:
            self.environment[cond] = self.evaluateQuant(cond, r_in, z_in)	
        
        if self.diffusion or self.collisions:
            for cond in ["dDdz", "dDdr", "drhogdz", "drhogdr"]:
                self.environment[cond] = self.evaluateQuant(cond, r_in, z_in)	
            
            
        self.environment["soundspeed0"] = self.evaluateQuant("soundspeed", r_in, 0/r_in)
        
        if self.trackice:
            for spec in self.disk.iceList:
                self.environment["gasAbun"+spec] = self.evaluateQuant("gasAbun"+spec, r_in, z_in)
                self.environment["iceAbun"+spec] = self.evaluateQuant("iceAbun"+spec, r_in, z_in)
                self.environment["totAbun"+spec] = self.evaluateQuant("totAbun"+spec, r_in, z_in)

    
    def storeEnvironment(self, r_now, z_now, t_now):
        """
        Stores the environment in dedicated arrays.
        """

        r_in, z_in, t_in = self.unpackVars(r_now, z_now,t_now)
        if t_in==0:
            for cond in self.trackListEnv:
                self.monomer.sec_sol[cond] = [self.environment[cond]]
            for cond in self.trackListFun:
                if cond=="St":
                    self.monomer.sec_sol[cond] = [self.Stokes(t_in, r_in, z_in)]
                elif cond=="delta_t":
                    self.monomer.sec_sol[cond] = [self.delta_t]
            if self.trackice:
                for species in self.disk.iceList:
                    self.monomer.sec_sol["gasAbun"+species] = [self.environment["gasAbun"+species]]
                    self.monomer.sec_sol["iceAbun"+species] = [self.environment["iceAbun"+species]]
        else:
            for cond in self.trackListEnv:
                self.monomer.sec_sol[cond].append(self.environment[cond])
            for cond in self.trackListFun:
                if cond=="St":
                    self.monomer.sec_sol[cond].append(self.Stokes(t_in, r_in, z_in))
                elif cond=="delta_t":
                    self.monomer.sec_sol[cond].append((self.monomer.t_sol[-1]-self.monomer.t_sol[-2])*self.sTOyr*1e3)
            if self.trackice:
                for species in self.disk.iceList:
                    self.monomer.sec_sol["gasAbun"+species].append(self.environment["gasAbun"+species])
                    self.monomer.sec_sol["iceAbun"+species].append(self.environment["iceAbun"+species])

#################################################################################################
### Operations related to dynamics
#################################################################################################

    
    def moveGrain(self, r_old, z_old, t, n):
        """
        Calculates the displacement of the home aggregate/monomer
        """
        # We here take abs(z) as we assume our disk to be symmetric around the midplane.
        
        randr, randz = 2*np.random.rand(2)-1

        self.seedNo += 1
        np.random.seed(self.seedNo)
        
               
        if self.qTest and (not self.qTestMig):
            del_r = -100*self.delta_t/(100*1e3*self.sTOyr)
        elif (self.fixR)or(self.breakIce):
            del_r = 0            
        else:
            del_r = self.r_der(r_old, z_old, t, randr)

        if self.qTest and (not self.qTestMig):
            del_z = 0
        elif (self.fixZ)or(self.breakIce):
            del_z = 0
        else:
            del_z = self.z_der(r_old, z_old, t, randz)
        
        r_new = r_old + del_r
        
        if (r_new < self.para["r_inner_cutoff"]):
            print("The monomer has drifted interior to the inner disk wall.")
            self.delta_t = self.t_stop*1e3*self.sTOyr
            r_new = self.para["r_inner_cutoff"]
        elif (r_new > 0.999*self.para["r_outer"]/self.auTOm):
            print("The monomer has drifted out of the disk.")
            self.delta_t = self.t_stop*1e3*self.sTOyr	
            r_new = self.para["r_outer"]/self.auTOm
        
        z_new = z_old + del_z

        if self.breakIce:
            r_new = 20-t
            z_new = 0.01*r_new
                
        if abs(z_new/r_new)>0.5:
            print("The monomer has drifted above abs(z/r)=0.5.")
            z_new = 0.5*r_new
            
        return r_new, z_new

#################################################################################################
### Operations related to coagulation
#################################################################################################
    
    
    def calcProbCol(self, t, r, z):
        """
        Calculates the probability of a collision happening in the time interval. Returns effective probability.
        """

        delT = self.delta_t
        
        self.PCol = 1-np.exp(-delT*np.sum(self.colRates))
        self.effPCol = 1-np.exp(-delT*np.sum(self.effColRates))
        
        return self.effPCol

    
    def determineCollisionP(self, t, r ,z):
        """
        The function which determnines whether a collision has happened during a timestep or not. Returns True if a collision
        happened; false if not.
        """

        pCol = self.calcProbCol(t, r, z)

        if np.random.rand()<=pCol:

                
            collision = True
        else:
            collision = False
            
        self.seedNo += 1
        np.random.seed(self.seedNo)

        return collision 

    
    def determineFragmentMass(self, mMax):
        """
        Auxiliary function in determining collision outcome. If fragmentation/erosion occurs, the 
        monomer may end up in one of the fragments. This function determines the fragment mass in
        which the monomer is embedded if this occurs.
        
        mMax - Maximum fragment mass allowed by dynamics.
        
        See also Birnstiel et al. 2010; 2011
        """
        
        
        x = np.random.power(6-3*self.para["x_frag"]+1, size=1)[0] # See Birnstiel+ 2011; we draw a value between 0 and 1, which is the range of possible values.
        self.seedNo += 1
        np.random.seed(self.seedNo)
        
        sMax = (3*mMax/(4*np.pi*self.monomer.homeAggregate.prop["rhoAgg"]))**(1/3)

        size = x*(sMax-self.monomer.prop["sMon"])+self.monomer.prop["sMon"]

        mass = 4/3*np.pi*self.monomer.homeAggregate.prop["rhoAgg"]*size**3

        return mass

    
    def determineCollisionPartner(self, t, r, z):
        """
        Determines the size and mass of the collision partner.
        """
        
        colSum = np.sum(self.effColRates)

        colNum = colSum*np.random.rand()

        self.seedNo += 1
        np.random.seed(self.seedNo)
            

        compSum = 0
        ind = 0

        while (compSum<colNum)and(compSum<colSum):
            compSum += self.effColRates[ind]
            ind += 1

        if ind>=100:
            size = self.grainSizes[-1]
            self.monomer.homeAggregate.prop["NCol"] = 1/self.effectiveFact[-1]
        else:
            size = self.grainSizes[ind]
            self.monomer.homeAggregate.prop["NCol"] = 1/self.effectiveFact[ind]

        # Determines cloud size
        self.monomer.homeAggregate.prop["sCol"] = size
        self.monomer.homeAggregate.prop["mCol"] = 4/3*np.pi*self.monomer.homeAggregate.prop["sCol"]**3*self.monomer.homeAggregate.prop["rhoAgg"]
        
        if (self.debug)and(self.monomer.homeAggregate.prop["sAgg"]>self.grainSizes[-1]):
            print("Collision rates: ", self.effColRates, "/s")
            print("Determined collision partner size: ", self.monomer.homeAggregate.prop["sCol"])


    
    def determineCollisionOutcome(self, t, r, z, size=None):
        """
        Determines whether erosion, fragmentation or coagulation takes place. All input is in SI.
        """
        
        vRel = self.calcTotVRel(t, r, z, doPrint=False) ### Note that calcTotVrel takes input in SI
        self.vRel = vRel

        cond = 99
        # Determine whether fragmentation occurs
        if vRel>=self.para["v_frag"]:
            cond = 1
            fragmentation = True
        elif vRel<(self.para["v_frag"]-self.para["del_v_frag"]):
            cond = 2
            fragmentation = False
        else:
            cond = 3
            pFrag = 1-(self.para["v_frag"]-vRel)/self.para["del_v_frag"]
            if np.random.rand()<=pFrag:

                fragmentation = True
            else:
                fragmentation = False
                
            self.seedNo += 1
            np.random.seed(self.seedNo)
        
        if fragmentation: # if fragmentation we need to pick a new, smaller size of our home aggregate

            # does catastrophic disruption or erosion occur?
            massRat = self.monomer.homeAggregate.prop["mCol"]/self.monomer.homeAggregate.prop["mAgg"]
            mCol = self.monomer.homeAggregate.prop["mCol"]
            mAgg = self.monomer.homeAggregate.prop["mAgg"]

            mMin = self.monomer.prop["mMon"]

            if massRat<=self.mrat:
                # In this case the home aggregate is being eroded.

                mTot = 2*mCol*self.monomer.homeAggregate.prop["NCol"] # multiply with cloud size,

                # Is the monomer in the excavated mass? Two masses worth of mCol are excavated from 
                # the home aggregate. Assuming random location,
                # the ejection probability is given by:

                pNotEj = (1-mTot/mAgg)**self.monomer.homeAggregate.prop["NCol"] # probability of not 
                #being ejected
                pEj = 1-pNotEj

                if np.random.rand()<=pEj: # the monomer is ejected and we need to determine the size of the fragment.           
                    mMax = mCol
                    self.monomer.homeAggregate.prop["mAgg"] = self.determineFragmentMass(mMax)
                    message = "home aggregate eroded, monomer ejected"
                    outcome = "ejection"
                    interaction_id = 4
                else: # some mass is eroded away from the home aggregate, but the monomer remains in the home aggregate.
                    self.monomer.homeAggregate.prop["mAgg"] -= mTot/2 
                    message = "home aggregate eroded by {:.1e} particles, monomer remained".format(self.monomer.homeAggregate.prop["NCol"])
                    outcome = "erosion"
                    interaction_id = 3

                self.seedNo += 1
                np.random.seed(self.seedNo)

            elif massRat>=1/self.mrat:
                # In this case the home aggregate is the eroding particle.
                # Note that in this case, the collision partner is never a cloud.
                mTot = 2*mAgg
                # Is the monomer in the excavated mass? ---> Assume no? May read a bit better into this.
                # - Brauer, Dullemond & Henning (2008)  
                # - Hasegawa et al. (2021) - Simulations, v_frag is function of mass ratio. 
                # For now, lets assume bullets: 2x mAgg of the collision partner are excavated such that the collision partner is the new home aggregate.
                # the impactor burries itself deep enough such that the excavated mass is solely originating from the collision partner.
                self.monomer.homeAggregate.prop["mAgg"] = self.monomer.homeAggregate.prop["mCol"] - mTot/2
                message = "home aggregate impacted"
                outcome = "impact"
                interaction_id = 5

            else:
                # In all the other cases the home aggregate is catastrophically disrupted.
                self.monomer.homeAggregate.prop["mAgg"] = self.determineFragmentMass(max([mAgg, mCol]))
                # In which fragment is the monomer? What is the size of this fragment?
                message = "catastrophic disruption"
                outcome = "fragmentation"
                interaction_id = 2
                    
        else: # otherwise we update it. Coagulation takes place.
            self.monomer.homeAggregate.prop["mAgg"] += self.monomer.homeAggregate.prop["mCol"]*self.monomer.homeAggregate.prop["NCol"]
            message = "coagulation with {:.1e} particle(s) of size {:.1e} m".format(self.monomer.homeAggregate.prop["NCol"], self.monomer.homeAggregate.prop["sCol"])
            outcome = "coagulation"
            interaction_id = 1


        # In any case the new home aggregate size is calculated via the new mass.
        self.monomer.homeAggregate.prop["sAgg"] = (3*self.monomer.homeAggregate.prop["mAgg"]/(4*np.pi*self.monomer.homeAggregate.prop["rhoAgg"]))**(1/3)   
        
        if self.monomer.homeAggregate.prop["sAgg"]<self.monomer.prop["sMon"]:
         # If the fragment is smaller than the monomer size there will be a free monomer.
            self.monomer.homeAggregate.prop["sAgg"] = self.monomer.prop["sMon"]
            self.monomer.homeAggregate.prop["mAgg"] = self.monomer.prop["mMon"]
        
        
        if (self.debug)and(self.monomer.homeAggregate.prop["sAgg"]>self.grainSizes[-1]):
            print("Home aggregate exceeded maximum mass in background distribution distribution")
            print("Event: ", outcome)
            print("Partner: ", self.monomer.homeAggregate.prop["sCol"],"m")
            print("Group size: ", self.monomer.homeAggregate.prop["NCol"])
            print("v_rel: ", vRel,"m/s")
            print(t,r,z)
        
        
        # Determine new depth of the monomer, and (if needed) if we are still exposed to the gas phase.
        if self.trackice:
            self.determineMonomerExposed(outcome)
        

        return message, interaction_id
    
    
    def doCollisions(self, r_in, z_in, t_in):
        
        r, z, t = self.unpackVars(r_in, z_in, t_in)
        
        collision = self.determineCollisionP(t, r ,z)

        if collision:
            self.determineCollisionPartner(t, r, z)
            log, interaction_id = self.determineCollisionOutcome(t, r, z)
            self.sizeChanged = True
        else:
            log = "no collision happened"
            interaction_id = 0
            self.sizeChanged = False
            
        if t==0:
            (self.monomer.sec_sol["interaction"]).append(interaction_id)
            (self.monomer.sec_sol["exposed"]).append(int(self.monomer.exposed))
        else:
            (self.monomer.sec_sol["interaction"]).append(interaction_id)
            (self.monomer.sec_sol["exposed"]).append(int(self.monomer.exposed))
      
    
    def determinePExp(self):

        tau = 3/4*(self.monomer.prop["zMon"]-self.monomer.prop["zCrit"])/self.monomer.prop["sMon"]*(self.monomer.homeAggregate.prop["phi"])
        pVals = expn(2, tau) # We solve the exponential integral
        return pVals

    
    def determineRandomPos(self):

        depth = self.monomer.homeAggregate.prop["sAgg"]*(1-np.random.rand()**(1/3))
        self.seedNo += 1
        np.random.seed(self.seedNo)
    
        return depth
    
          
    def determineMonomerExposed(self, outcome, redoAgg=False):
        """
        Determines the depth in the new home aggregate at which the monomer is located. Method depends on collision outcome. Requires the new aggregate size
        to be known.
        
        Note to self: We still have the possibility to make the exposure probability vary as a function of distance from the home aggregate core. For now it is
        0 if z>z_crit or 1 if z<z_crit
        """
        
        # First we determine the new depth
        if outcome in ["coagulation", "erosion"]:
            sOld = self.monomer.sAgg_sol[-1]
            sNew = self.monomer.homeAggregate.prop["sAgg"]
            self.monomer.prop["zMon"] += sNew-sOld
            if self.monomer.prop["zMon"]<0:
                self.monomer.prop["zMon"] = 0 # if erosion of the home aggregate leads to a negative z, we set it zero.
        elif outcome in ["fragmentation", "ejection", "impact"]: 
            self.monomer.prop["zMon"] = self.determineRandomPos() 
            
        # Subsequently whether the monomer is exposed at that depth
        if self.monomer.prop["zMon"]<=self.monomer.prop["zCrit"]:
            self.monomer.exposed = True
        else:
            pExp = self.determinePExp()
            if np.random.rand()<=pExp: 
                self.monomer.exposed = True
            else:
                self.monomer.exposed = False
            
            self.seedNo += 1
            np.random.seed(self.seedNo)
                

#################################################################################################
### Operations related to ice formation
#################################################################################################

    
    def calcElements(self):
        """
        This method returns the respective masses of H, C, N, O & S in kg present in the monomer ice mantle. 
        
        This method should only be used after the main integrateMonomer loop.
        """
        
        self.monomer.ele_sol=np.zeros((len(self.monomer.ice_sol[self.disk.iceList[0]]),5)) # TODO: Get rid of the magic element number.
        
        #self.monomer.ice_sol[self.disk.iceList[n]]
        
        for n in range(self.iceNum):
            iceName = self.disk.iceList[n]
        
            # convert to moles
            conv = self.para["m"+iceName]/self.para["u"]*1e-3 #m+Icename is in kg, so conv represents kg/mol for molecule X

            massMol = self.monomer.ice_sol[iceName]/conv # mass of molecule X in moles
        
            molElement = self.disk.elementDict[iceName] # gives the number of moles of each element in one mole of molecule X
            # calculate how many moles there of every element due to the presence of molecule X.
            
            kgElement = molElement*self.para["mElements"]# Convert from moles to kg.
            #print(np.outer(massMol kgElement).shape)
            self.monomer.ele_sol += np.outer(massMol, kgElement) # gives the mass change in moles for each element. Multiply with molar mass to find gain in kg.
        
        
        return self.monomer.ele_sol

    
    def iceMassChange(self, t, r, z, iceName, iceList=None):
        """
        Calculates the gain/loss of a certain ice species in kg/s for the whole monomer ice mantle.
        """

        surface = 4*np.pi*self.monomer.prop["sMon"]**2
        cross = 2*np.pi*self.monomer.prop["sMon"]**2 
        if self.qTest:
            if self.qTestAds:
                self.monomer.exposed = True
            else:
                self.monomer.exposed = False
                
        if self.monomer.exposed:
            # If the monomer is exposed we do adsorption, thermal desorption and photodesorption.
            ads = self.rateAdsorption(t, r, z, iceName, iceList=iceList)            
            tds = self.rateDesorptionThermal(t, r, z, iceName, dustT=None, iceList=iceList)
            pds = self.rateDesorptionPhoto(t, r, z, iceName, iceList=iceList)        
            
            # prodimo uses surface, we use cross section
            diffMass = cross*ads -surface*tds -cross*pds
        
            qFactor = np.nan
        else:

            if self.readsorption:
            # if this lever is true we assume all thermally desorbed ice is readsorbed.
                diffMass = 0
                tds = 0
                pds = 0
                ads = 0
                qFactor = 1
            elif self.readsorption==None:
                tds, qFactor = self.rateDesorptionInternal(t, r, z, iceName, iceList=iceList)########## TO DO: iceList
                # Note that in this case the q-factor is defined in the above method.
                diffMass = -surface*tds
                pds = 0
                ads = 0
            else:
                ## Otherwise we only do thermal desorption.
                tds = self.rateDesorptionThermal(t, r, z, iceName, dustT=None, iceList=iceList)
                diffMass = -surface*tds
                # Otherwise nothing happens
                pds = 0
                ads = 0
                qFactor = 0
               
        
        if self.legacyIce:
            return diffMass, ads, tds, pds, qFactor # In the old formalism we can return the rates
        else:
            return diffMass
        

    def doIceEvolutionMyOwn(self, r_in, z_in, t_in):
        
        r,z,t = self.unpackVars(r_in, z_in, t_in)

        n=0
        diffMassPrelim = {}
        adsPrelim = {}
        tdsPrelim = {}
        pdsPrelim = {}
        qPrelim = {}
        
        deltaTfloor = self.delta_t_floor*self.sTOyr
        
        zeroAccumulateFlag = {}
        
        #------------------------------------------------------------------------
        
        while n<self.iceNum:
            
            # print("Calculating for "+self.disk.iceList[n]+" ice")
            diffMass, adsPrelim[self.disk.iceList[n]], tdsPrelim[self.disk.iceList[n]], pdsPrelim[self.disk.iceList[n]], qPrelim[self.disk.iceList[n]] = self.iceMassChange(t, r, z, self.disk.iceList[n])
            diffMassPrelim[self.disk.iceList[n]] = diffMass*self.delta_t
            
            # Calculate the ratio, make sure to catch errors where we have 0/0.
            diffZeroBool = diffMassPrelim[self.disk.iceList[n]]==0
            icesZeroBool = (self.monomer.ice_sol[self.disk.iceList[n]])[-1]==0.
           
            if icesZeroBool:
                ratio = 1
            elif icesZeroBool and diffZeroBool:
                ratio = 0
            else:
                ratio = diffMassPrelim[self.disk.iceList[n]]/(self.monomer.ice_sol[self.disk.iceList[n]])[-1]
            
            # is the relative mass change for species n exceeded?
            ratioBool = ratio>self.fice

            # have we not yet reached the minimum allowed timestep?
            floorBool = self.delta_t>deltaTfloor 

            # Was the monomer not just iceless?
            bareBool = (self.monomer.ice_sol[self.disk.iceList[n]])[-1]<1e-50 

            # do we not use constant timestep?
            notConstTimeBool = not self.constTimestep[0] 
            
            n += 1
            
            if (ratioBool) and (floorBool) and (notConstTimeBool) and (not bareBool):
                self.delta_t /= 10 
                # Then we adjust the timestep if needed; note that we do this AFTER the shortest timescale has been chosen.
                if self.delta_t<deltaTfloor:
                    self.delta_t=deltaTfloor
                    zeroAccumulateFlag[self.disk.iceList[n-1]] = True
                    n=0
                else:
                    zeroAccumulateFlag[self.disk.iceList[n-1]] = False    
                    n=0   
                    
            elif (ratioBool) and (bareBool):
                # This is a warning condition for when we accumulate too much ice too quickly.
                # We accept the change right away and calculate the desorption rate with the new changes later.
                # In this way we prevent unnecesary decreases in timestep down to the floor of delta_t.
                zeroAccumulateFlag[self.disk.iceList[n-1]] = True
            else:
                zeroAccumulateFlag[self.disk.iceList[n-1]] = False                
        
        #------------------------------------------------------------------------
        
        for n in range(self.iceNum):
            #First make the preliminary ads/tds/pds rates final.
            (self.monomer.ice_sol["ads"+self.disk.iceList[n]]).append(adsPrelim[self.disk.iceList[n]])
            (self.monomer.ice_sol["tds"+self.disk.iceList[n]]).append(tdsPrelim[self.disk.iceList[n]])
            (self.monomer.ice_sol["pds"+self.disk.iceList[n]]).append(pdsPrelim[self.disk.iceList[n]])
            (self.monomer.ice_sol["qFactor"+self.disk.iceList[n]]).append(qPrelim[self.disk.iceList[n]])
            
            
            iceNew = (self.monomer.ice_sol[self.disk.iceList[n]])[-1] + diffMassPrelim[self.disk.iceList[n]]

            
            self.monomer.ice_sol[self.disk.iceList[n]].append(max([0,iceNew])) # at the end we add the new ice solution
            
        # Once we are done with dealing with the various ices, we calculate the new total amount of ice on the monomer.
        iceTot = self.calcIceTot()
        self.monomer.iceTot_sol.append(iceTot)
        
        #------------------------------------------------------------------------
        
        diffMassPrelimZ = {}
        adsPrelimZ = {}
        tdsPrelimZ = {}
        pdsPrelimZ = {}
        qPrelimZ = {}
                
        for n in range(self.iceNum):
        # An extra loop to check if there were any zero-accumulations. If yes, check whether everything is
        # lost again next iteration to avoid oscilatory ice behaviour.
        
            if zeroAccumulateFlag[self.disk.iceList[n]]:
                diffMass, adsPrelimZ[self.disk.iceList[n]], tdsPrelimZ[self.disk.iceList[n]], pdsPrelimZ[self.disk.iceList[n]], qPrelimZ[self.disk.iceList[n]] = self.iceMassChange(t, r, z, self.disk.iceList[n])
                diffMassPrelimZ[self.disk.iceList[n]] = diffMass*self.delta_t            
            
                if -diffMassPrelimZ[self.disk.iceList[n]]>=(self.monomer.ice_sol[self.disk.iceList[n]])[-1]:
                    iceNew = (self.monomer.ice_sol[self.disk.iceList[n]])[-1] + diffMassPrelimZ[self.disk.iceList[n]]
                    (self.monomer.ice_sol[self.disk.iceList[n]])[-1] = max([0,iceNew])                   

                    (self.monomer.ice_sol["ads"+self.disk.iceList[n]])[-1] = adsPrelimZ[self.disk.iceList[n]]
                    (self.monomer.ice_sol["tds"+self.disk.iceList[n]])[-1] = tdsPrelimZ[self.disk.iceList[n]]
                    (self.monomer.ice_sol["pds"+self.disk.iceList[n]])[-1] = pdsPrelimZ[self.disk.iceList[n]]
                    (self.monomer.ice_sol["qFactor"+self.disk.iceList[n]])[-1] = qPrelimZ[self.disk.iceList[n]]
                    
        self.monomer.iceTot_sol[-1] = self.calcIceTot()
        
        if self.pisoBenchmark:
            self.monomer.prop["mMon"] = self.monomer.iceTot_sol[-1]
            self.monomer.prop["sMon"] = (3*self.monomer.prop["mMon"]/(4*np.pi*self.monomer.prop["rho_mat"]))**(1/3)
            self.monomer.homeAggregate.prop["mAgg"] = self.monomer.iceTot_sol[-1]
            self.monomer.homeAggregate.prop["sAgg"] = (3*self.monomer.homeAggregate.prop["mAgg"]/(4*np.pi*self.monomer.homeAggregate.prop["rhoAgg"]))**(1/3)
    
    
    
    def scipyAuxFunc(self, t_in, y_in, *args):
        """
        Auxilary function which calculates the derivatives for the ice changes.
        
        Units:
        Position: AU
        y_in: potatoes (ice mass)
        t_in: kyr (time)
        """

        r_in = args[0]
        z_in = args[1]
        t_in += args[2]

        r,z,t = self.unpackVars(r_in, z_in, t_in)
        
        dydt = np.zeros(len(y_in))
        
        iceList = {}

        for n in range(self.iceNum): 
            y_in[n] = max(y_in[n],0)
            iceList[self.disk.iceList[n]] = y_in[n]*self.numFact # This is more conistent with ice bookkeeping in self.iceMassChange. Maybe get rid of dictionaries altogether?
            # convert in icelist from potatoes to kg

        for n in range(self.iceNum): 
            # note we have to convert from kg/s to potatoes/kyr          
            dydt[n] = self.iceMassChange(t, r, z, self.disk.iceList[n], iceList=iceList)*(1e3*self.sTOyr)/(self.numFact)
            
        ####### Failsave for when ice abundances get too low.
            if iceList[self.disk.iceList[n]]/self.monomer.prop["mMon"]<self.floorval and dydt[n]<0:
                dydt[n] = 0

        return dydt
    
    @timeout(1.5)
    def advanceIceMantle(self, t_start, t_stop, y0, position, integrator):
        
        success=False
        self.printFlag = True

        t_eval = None

        t_eval = np.linspace(t_start, t_stop, 2)

        if self.monomer.exposed:
            atol = self.atol[0]
            rtol = self.rtol[0]
            method = integrator 
            sol = solve_ivp(self.scipyAuxFunc, (t_start,t_stop), y0, t_eval=t_eval, args=position, method=method, atol=atol, rtol=rtol)
        else:
            atol = self.atol[1]
            rtol = self.rtol[1]
            method = integrator
            sol = solve_ivp(self.scipyAuxFunc, (t_start,t_stop), y0, t_eval=t_eval, args=position, method=method, atol=atol, rtol=rtol)
        if sol["status"]==0:
            success = sol["success"]
        else:
            success = False
      
        return sol, success
    
    
    def doIceEvolutionSciPy(self, r_in, z_in, t_in):
        """
        New code which solves the time evolution of the monomer ice using pre-existing ODE-integration routines.  
        
        
        Classification of exit index:
        0 - Integrator worked as intended
        1 - Monomer is in the inner disk numerical regime, no integration peformed.
        2 - Integrator worked as intended in this timestep, but there have been timeouts in the past.
        3 - Integrator worked as intended in this timestep, but there have been manual resets in the past.
        ---- Default cutoff ----
        4 - Integrator worked as intended in this timestep, but there have been convergence errors in the past.
        5 - Timeout error.
        6 - Convergence error, ice mantle reset due to high local UV radiation.
        7 - Convergence error.
        """
   
        self.numFact = self.monomer.iceTot_sol[-1]*self.iceScaleFact*self.monomer.prop["mMon"] # kg/potatoes
        y0 = np.array([(self.monomer.ice_sol[self.disk.iceList[n]])[-1]/self.numFact for n in range(self.iceNum)])
        # such that y0 is in potatoes
        
        if self.migration or self.collisions:
            delt = self.delta_t/(1e3*self.sTOyr)
        else:
            delt = self.t_stop

        t_start = 0
        t_stop = delt
        
        integratorList = ["LSODA", "Radau", "BDF"] # LSODA should be most flexible, but can be worth trying others in case of failure (Radau & BDF are specifically written for stiff systems).
        I = len(integratorList)
        i = 0
        
        success = False
        self.monomer.exitIndex = 7 # Exit index 7 is the worst, it means an unhandled convergence error.
        
        if r_in>1:
            while ((not success) and (i<I)):
                try:
                    sol, success = self.advanceIceMantle(t_start, t_stop, y0, (r_in, z_in, t_in), integrator=integratorList[i])
                except:
                    # If integrators take too long, we keep the ice mantle from previous timestep.
                    self.monomer.exitIndex = 5 # For now we have a timeout of the routines.
                    # Success remains false in this case.
                    sol = {}
                    
                    sol["y"] = self.floorval*self.monomer.prop["mMon"]/self.numFact*np.ones((self.iceNum, 2)) 
                    sol["y"][:,0] = y0
                    sol["y"][:,1] = y0
                    sol["t"] = np.linspace(t_start, t_stop, 2)
                i += 1    
            
            # An extra clause to reset ice mantles strongly affected by UV radiation (sometimes causes numerical issues).
            if (not success)and(self.environment["chiRT"]>1)and(self.monomer.exposed):
                self.monomer.exitIndex = 6
                sol = {}
                sol["y"] = self.floorval*self.monomer.prop["mMon"]/self.numFact*np.ones((self.iceNum, 2)) 
                sol["t"] = np.linspace(t_start, t_stop, 2)
            
            # Make sure we have the correct exit index if errors occured in the past.
            if success:
                if self.monomer.corrupt:
                    self.monomer.exitIndex = np.max(self.monomer.exitTracker)-3
                else:
                    self.monomer.exitIndex = 0
        else:
            self.monomer.exitIndex = 1
            success = True
            sol = {}
            sol["y"] = self.floorval*self.monomer.prop["mMon"]/self.numFact*np.ones((self.iceNum, 2)) 
            sol["t"] = np.linspace(t_start, t_stop, 2)
        
        if (i-1)>0:
            print("Default ice routine failed, monomer exited index ",self.monomer.exitIndex," with backup routine ", integratorList[i-1], ". Success: ", success)            
            
        solution = sol["y"]*self.numFact   
        
        limit = self.floorval*self.monomer.prop["mMon"]
        
        solution[solution<limit] = limit 
    
        if self.migration or self.collisions:
            for n in range(self.iceNum):
                self.monomer.ice_sol[self.disk.iceList[n]].append(solution[n,-1])
        else:
            self.monomer.t_sol = sol["t"]
            for n in range(self.iceNum):
                self.monomer.ice_sol[self.disk.iceList[n]] = solution[n,:]

        # Once we are done with dealing with the various ices, we calculate the new total amount of ice on the monomer.
        iceTot = self.calcIceTot()
        self.monomer.iceTot_sol.append(iceTot)    
        
        # Update corruption tracker        
        if (not success):
            self.monomer.corrupt = True
        
        #print(self.monomer.t_sol, self.monomer.ice_sol["H2O"])
        
    def doIceEvolution(self, r_in, z_in, t_in):
        """
        At each timestep, we run this algorithm to update the abundances in the ice.
        """
        
        if self.legacyIce:
            self.doIceEvolutionMyOwn(r_in, z_in, t_in)
        else:
            self.doIceEvolutionSciPy(r_in, z_in, t_in)
 
#################################################################################################
# Timestep functions
#################################################################################################
    
    
    def tauMz(self, t, r, z, Hp=None, size=None):
        "Returns the vertical settling timescale in seconds. Traversed distance is one particle scale height"    
        
        vz = abs(self.velocity_z(t, r, z, size=size))
        
        if Hp==None:
            Hp = self.calculateScaleHeight(r, method="mcfost", kind="gas", size=size)
        
        tauMz = Hp/vz
        
        return tauMz
    
    
    def tauMr(self, t, r, z, Hp=None, size=None):
        "Returns the radial drift timescale in seconds. Traversed distance is one particle scale height, unless PisoBenchmark" 
    
        vr = abs(self.velocity_r(t, r, z, size=size))
        if Hp==None:
            Hp = self.calculateScaleHeight(r, method="mcfost", kind="gas", size=size)

        if self.pisoBenchmark:
            Hp = r
        
        tauMr = Hp/vr
    
        return tauMr
    
    
    def tauTub(self, t, r, z, Hg=None):
        "Returns the turbulent stirring timescale in seconds." 
        
        Hg = self.calculateScaleHeight(r, method="mcfost", kind="gas")
        viscosity = self.nuTur(t, r, z, midplane=True)
        
        tauTub = Hg**2/viscosity
        
        return tauTub
    
    def tauCol(self, t, r, z, physical=False):
        """
        Returns the shortest collision timescale in seconds.
        """
        
        if physical:
            tauCol = 1/np.sum(self.colRates)
        else:
            tauCol = 1/np.sum(self.effColRates)
        
        return tauCol
    
    def tauAds(self, t, r, z, size, species):
        """
        Returns the adsorption timescale in seconds.
        """
        
    
        tauAds = (self.monomer.ice_sol[species])/(4*np.pi*(size**2)*self.rateAdsorption(t, r, z, species))
       
        return tauAds[0]
    
    def tauDes(self, t, r, z, size, species):
        """
        Returns the total desorption timescale in seconds.
        """
    
        rateDes = self.rateDesorptionThermal(t, r, z, species, fx=1) + self.rateDesorptionPhoto(t, r, z, species, fx=1)
        tauDes = (self.monomer.ice_sol[species])/(4*np.pi*(size**2)*rateDes)
       
    
        return tauDes[0]
    
    def tauDiff(self, t, r, z, size, species):
        """
        Calculates the diffusion timescale in a given aggregate for rms-displacement "size" for species "species"
        at position (r,z). All units in SI.
        """
        
        mDiffusivity = self.moleculeDiffusivity(t, r, z, species)
        
        tauDiff = (size**2)/(6*mDiffusivity)
        
        return tauDiff
    
    
    def tauTds(self, t, r, z, size=None):
        """
        Calculates the thermal desorption timescale in accordance with Piso+ 2015 in seconds.
        
        !!! Do not use outside of PisoBenchmark !!!
        """
               
        
        if size==None:
            size = self.monomer.prop["sMon"]
        
        spec = self.rateDesorptionThermal(t, r, z, "H2O", dustT=None, init=False)
        
        tauTds = size*self.monomer.prop["rho_mat"]/(3*spec)
    
        return tauTds
    
    def determineDeltaT(self, t_in, r_in, z_in):
        
        r,z,t = self.unpackVars(r_in, z_in, t_in)
        
        tauList = [1e6*self.sTOyr]
        
        if self.collisions: # assumes that the effective collision rates are already calculated.
            tauCol = self.tauCol(t, r, z)
            tauList.append(tauCol)
            
        if self.migration:
            tauMz = self.tauMz(t, r, z)
            tauMr = self.tauMr(t, r, z)
            tauList.append(self.fdyn*tauMz)
            tauList.append(self.fdyn*tauMr)
            if (t_in==0)and(self.verbose>0):
                print("Migration timescale for a grain of size {:.2f} m is {:.2e} yr".format(self.monomer.prop["sMon"],tauMr/self.sTOyr))
        
            if self.diffusion:
                tauTub = self.tauTub(t, r, z)
                tauList.append(self.ftub*tauTub)

        # Note that the timescale is chosen to be shorter if this is required by the ice formation algorithm.
        if (self.collisions or self.migration):
            tauMin = min(tauList)
            deltaT = max([self.delta_t_floor*self.sTOyr,tauMin])
        else:
            # If do not do collisions or migration we set the global timestep to 1/10th the local orbital period.
            deltaT = 0.1*2*np.pi/(self.Omega(t, r, z))
        
        return deltaT
    
    
    
#################################################################################################
# Main functions
#################################################################################################

    def probeTimescales(self, size, species):
        """
        Calculates the timescales for a monomer of given size (in m) and species (label). t,r,z are in kyr/AU
        """
        
        R = len(self.disk.model.x[:,0])
        Z = len(self.disk.model.z[0,:])
        
        tauSol = np.zeros((R,Z,6))
        
        self.initGradients(size=size) # Repeat whenever size changes!!!
        self.initDensities()
    
        
        for rInd in range(R):
            for zInd in range(Z):
                print("Calculating... progress: {:.0f}/{:.0f} points".format((rInd*Z+zInd+1),R*Z), end="\r")
                r_in = self.disk.model.x[rInd,zInd]
                z_in = self.disk.model.z[rInd,zInd]
                r,z,t = self.unpackVars(r_in, z_in, 0)
                self.monomer = Monomer(self, r, z, size=size)
        
                self.initIces(0, r_in, z_in)
                self.probeEnvironment(r_in, z_in, 0)
                self.calcColRates(0, r_in, z_in)
                
                tauSol[rInd,zInd,0] = self.tauMr(t, r, z, Hp=None, size=size)                                    
                tauSol[rInd,zInd,1] = self.tauMz(t, r, z, Hp=None, size=size)                                   
                tauSol[rInd,zInd,2] = self.tauTub(t, r, z, Hg=None)
                tauSol[rInd,zInd,3] = self.tauCol(t, r, z, physical=False)
                tauSol[rInd,zInd,4] = self.tauAds(t, r, z, size, species)
                tauSol[rInd,zInd,5] = self.tauDes(t, r, z, size, species)
        
        return tauSol
        

    def drawRandomPos(self):
        """
        If integrateMonomer receives randomize=True, we draw our monomer based on the 2D density distribution of the background model.
        """
        
        r0 = loguniform.rvs(self.innerRdraw, self.outerRdraw)
        z0 = r0*(0.2*np.random.rand()-0.1)

        return r0, z0
    
    def unpackVars(self, r_in,z_in,t_in):
        """
        Auxiliary function in the dynamics module to convert the position in space and time from AU/kyr to SI. 
        """

        r = r_in*self.auTOm # convert from AU to SI
        z = z_in*self.auTOm # convert from AU to SI
        t = t_in*1e3*self.sTOyr # convert from kyr to SI

        return r, z, t

    
    def integrateMonomer(self, size=None, t_stop_in=None, t_start=0, r0=None, z0=None, randomize=None, discretePos=None, timescaleMode=False, seed=None):
        """
        Evolves a monomer over time.
        
        size        - Monomer size in m
        t_stop_in   - Final integration time in yr
        t_start     - Starting integration time. Would not set this to anything different than zero.
        r0          - Initial radial position in AU.
        z0          - Initial vertical position in AU.
        """
        
        if not self.supverbose:
            print(" ")
            print(50*"-")    
            print(" ")
            print("Initializing monomer... ")
            print(" ")
            print(50*"-")    
            print(" ")
        
        # Input management
        if randomize==None:
            randomize = bool(self.paraDict["randomize"])
        
        if randomize:
            r0, z0 = self.drawRandomPos()
            self.paraDict["r0"] = r0
            self.paraDict["z0"] = z0
        elif (r0==None)or(z0==None):
            r0 = float(self.paraDict["r0"])
            z0 = float(self.paraDict["z0"])
        
        if not self.supverbose:
            print("Set initial monomer position to: r0 = {:.2f} AU and z0/r0 = {:.2f}".format(r0,z0/r0))
            
        if size==None:
            size = float(self.paraDict["sMon"])
            
        if t_stop_in==None:
            self.t_stop = float(self.paraDict["t_stop"])
        else:
            self.t_stop = t_stop_in/1e3 # convert to kyr
        
        # Choose random number setup
        # --------------------------
        if self.deterministic:
            if seed==None:
                try:
                    self.seedStart = int(self.paraDict["seed"])
                    self.seedNo = int(self.paraDict["seed"])
                except:
                    self.seedStart = np.random.randint(0, high=(2**32 - 1), size=None, dtype=int)
                    self.seedNo = self.seedStart
            else:
                self.seedStart = seed
                self.seedNo = seed
            
        else:
            self.seedStart = np.random.randint(0, high=(2**32 - 1), size=None, dtype=int)
            self.seedNo = self.seedStart

        if not self.supverbose:
            print("Monomer seed is: "+str(self.seedStart))
        np.random.seed(self.seedNo)
        
        # Initialize monomer
        # ------------------
        self.monomer = Monomer(self, r0, z0, size=size)

        self.monomer.r_sol = [self.monomer.initR]
        self.monomer.z_sol = [self.monomer.initZ]
        self.monomer.t_sol = [0]
        self.monomer.sAgg_sol = [self.monomer.homeAggregate.prop["sAgg"]]
        self.monomer.zMon_sol = [0]
        self.monomer.sec_sol = {}
        # We here also define a auxiliary array which attempts to keep track of whether data points have
        # become "corrupt" due to convergence failures of the integration routines.
        self.monomer.corrupt = False
        self.monomer.corruption = [int(self.monomer.corrupt)]     
        
        self.monomer.exitIndex = 0
        self.monomer.exitTracker = [0]
        
        tn = 0
        ticTot = process_time()
       
        
        self.initGradients() ##############! Repeat whenever size changes!!!
            
        if self.collisions: ### calculate the size-dependent grain sizes
            self.initDensities()
            self.monomer.sec_sol["interaction"] = [0]
            verbose_string = "exposed"
            if self.colEq: ### Update sAgg and zMon.
                self.initAggregate(0, r0, z0)
                self.monomer.sAgg_sol = [self.monomer.homeAggregate.prop["sAgg"]]
                self.monomer.zMon_sol = [self.monomer.prop["zMon"]]
                self.monomer.sec_sol["exposed"] = [int(self.monomer.exposed)]
                if int(self.monomer.exposed)==0:
                    verbose_string = "unexposed"
            else:
                self.monomer.sec_sol["exposed"] = [1]
        if not self.supverbose:
            print("Monomer placed in aggregate:")
            print("sAgg:"+10*" "+"{:.2e} m".format(self.monomer.homeAggregate.prop["sAgg"]))
            print("zMon:"+10*" "+"{:.2e} m".format(self.monomer.prop["zMon"]))
            if self.monomer.exposed:
                verbose_string = "exposed"
            else:
                verbose_string = "unexposed"
            print("State:"+10*" "+verbose_string)

            
       # if self.trackice: # define the data structures for ice budgets.
        self.initIces(0, r0, z0)
        
        if not self.supverbose:
            if self.trackice:
                print("Initialized ice budget.")
            else:
                print("Skipped ice initialization.")

            
            print(50*"-")    
            print(" ")
            print("Finished monomer initialization, commencing time evolution...")
            print(" ")
            print(50*"-")    
            print(" ")
        
        
        
        probeEnvironmentDat = []
        collisionRateDat = []
        moveGrainDat = []
        doCollisionDat = []
        doIceEvolutionDat = []

        self.delta_t = self.constTimestep[1]*self.sTOyr
        
        
        
        
        # The main integration time loop
        # -------------------------------------------------------------------
                
        while (tn<self.t_stop):
            
            tocTot = process_time()
            progress = tn/self.t_stop
            if progress>0:
                self.left = (tocTot-ticTot)/progress*(1-progress)
            else:
                self.left = 0
            progress *= 100
            if not self.supverbose:
                print("Integration progress: {:.2f} % complete, estimated time left: {:.2f} s, current step size: {:.1e} yr".format(progress, self.left, self.delta_t/self.sTOyr)+" "*8, end="\r")


            n = len(self.monomer.t_sol)
            if self.verbose>2:
                print("Performing timestep "+str(n)+" at t = {:.2e} yr".format(self.monomer.t_sol[-1]*1e3))

            r_now = self.monomer.r_sol[-1]
            z_now = self.monomer.z_sol[-1]
            t_now = self.monomer.t_sol[-1]
#             if len(self.monomer.r_sol)<10:
#                 print("Initialized sizes:", self.monomer.prop["sMon"], self.monomer.homeAggregate.prop["sAgg"])
            tic = process_time()
            self.probeEnvironment(r_now, z_now, t_now)# Calculate interpolated quantities
            toc = process_time()
            probeEnvironmentDat.append(toc-tic)            
            
            # Calculate collision rates
            if self.collisions:
                tic = process_time()
                self.calcColRates(t_now, r_now, z_now)
                toc = process_time()
                collisionRateDat.append(toc-tic)
            else:
                collisionRateDat.append(0)
                
            # Set the timestep
            # ------------------------
            if self.migration or self.collisions:
                if self.constTimestep[0]:
                    self.delta_t = self.constTimestep[1]*self.sTOyr
                else:
                    self.delta_t = self.determineDeltaT(t_now, r_now, z_now) 
            else:
                self.delta_t = self.t_stop*1e3*self.sTOyr
                
                
            # Store the environment here   
            # ------------------------
            if self.store==0:
                self.storeEnvironment(r_now, z_now, t_now)
           
            # delta_t is in seconds, because we only use it inside the routine. 

            # Do ice formation
            if self.trackice:
                tic = process_time()
                self.doIceEvolution(r_now, z_now, t_now) # Note that ice formation may decrease the stepsize if it turns out that ice formation goes very fast.
                toc = process_time()
                doIceEvolutionDat.append(toc-tic)
            else:
                doIceEvolutionDat.append(0)
           
            # Do collisions before migration
            if self.collisions:
                tic = process_time()
                self.doCollisions(r_now, z_now, t_now)
                if self.sizeChanged:
                    self.initGradients() # once done, re-calculate the gradients because grain diffusion slopes have changed. Only do this if
                    # collision occured; otherwise the home aggregate size remains the same.
                toc = process_time()
                doCollisionDat.append(toc-tic)
                if self.store==0:
                    if t_now==0:
                        try:
                            self.monomer.vRel_sol = [self.vRel]
                        except:
                            self.vRel = 0
                            self.monomer.vRel_sol = [self.vRel]

                    else:
                        self.monomer.vRel_sol.append(self.vRel)
                self.monomer.sAgg_sol.append(self.monomer.homeAggregate.prop["sAgg"])
                self.monomer.zMon_sol.append(self.monomer.prop["zMon"])
            else:
                self.monomer.sAgg_sol.append(self.monomer.homeAggregate.prop["sAgg"])
                self.monomer.zMon_sol.append(0)
                if self.store==0:
                    if t_now==0:
                        self.monomer.vRel_sol = [0]
                    else:
                        self.monomer.vRel_sol.append(0)

            
            # Move the grain if we have set self.migration==True. We do this last; because we can treat all other processes locally.
            if self.migration:
                tic = process_time()
                
                r_new, z_new = self.moveGrain(r_now, z_now, t_now, n) 
                # Move the grain to its new position
                
                toc = process_time()
                moveGrainDat.append(toc-tic)

                self.monomer.r_sol.append(r_new)
                self.monomer.z_sol.append(z_new)
            else:
                self.monomer.r_sol.append(r_now)
                self.monomer.z_sol.append(z_now)
                moveGrainDat.append(0)
                      
            # Update corruption array
            self.monomer.corruption.append(int(self.monomer.corrupt))
            self.monomer.exitTracker.append(self.monomer.exitIndex)
                      
            # Advance timestep
            t_new = self.monomer.t_sol[-1] + self.delta_t/(1e3*self.sTOyr) # Advance timestep  

            tn = t_new
            
            if self.migration or self.collisions:
                self.monomer.t_sol.append(t_new)          
        
            
        T = len(self.monomer.t_sol)
        if self.store==0:
            self.storeEnvironment(self.monomer.r_sol[-1], self.monomer.z_sol[-1], self.monomer.t_sol[-1]) 
            try:
                self.monomer.vRel_sol.append(self.vRel)    
            except:
                self.vRel = 0
                self.monomer.vRel_sol.append(self.vRel)
        
        tocTot = process_time()
        if self.verbose>-1:
            print("")
            print("Integration complete, elapsed CPU time is {:.2f} s".format(tocTot-ticTot))
            if self.printStatistics:
                print("")
                print("Function statistics:")
                print("-"*50)
                print("self.calcColRates")
                tcollisionRateTot = np.sum(np.array(collisionRateDat))
                print("Total call time : {:.2f} CPUs".format(tcollisionRateTot))
                print("Average time per call : {:.2e} CPUs".format(tcollisionRateTot/T))
                print("")
                print("self.probeEnvironment")
                tProbeEnvironmentTot = np.sum(np.array(probeEnvironmentDat))
                print("Total call time : {:.2f} CPUs".format(tProbeEnvironmentTot))
                print("Average time per call : {:.2e} CPUs".format(tProbeEnvironmentTot/T))
                print("")
                print("self.doIceEvolution")
                tDoIceEvolutionTot = np.sum(np.array(doIceEvolutionDat))
                print("Total call time : {:.2f} CPUs".format(tDoIceEvolutionTot))
                print("Average time per call : {:.2e} CPUs".format(tDoIceEvolutionTot/T))
                print("")
                print("self.moveGrain ")
                tMoveGrainTot = np.sum(np.array(moveGrainDat))
                print("Total call time : {:.2f} CPUs".format(tMoveGrainTot))
                print("Average time per call : {:.2e} CPUs".format(tMoveGrainTot/T))
                print("")
                print("self.doCollisions")
                tdoCollisionTot = np.sum(np.array(doCollisionDat))
                print("Total call time : {:.2f} CPUs".format(tdoCollisionTot))
                print("Average time per call : {:.2e} CPUs".format(tdoCollisionTot/T))
                print("-"*50)

        if self.migration or self.collisions:
            self.monomer.r_sol = np.array(self.monomer.r_sol)
            self.monomer.z_sol = np.array(self.monomer.z_sol)
            self.monomer.t_sol = np.array(self.monomer.t_sol)
            self.monomer.sAgg_sol = np.array(self.monomer.sAgg_sol)
            self.monomer.zMon_sol = np.array(self.monomer.zMon_sol)
            self.monomer.corruption = np.array(self.monomer.corruption)
            self.monomer.exitTracker = np.array(self.monomer.exitTracker)
        else:
            self.monomer.r_sol = self.monomer.r_sol[-1]*np.ones(T)
            self.monomer.z_sol = self.monomer.z_sol[-1]*np.ones(T)
            self.monomer.sAgg_sol = self.monomer.sAgg_sol[-1]*np.ones(T)
            self.monomer.zMon_sol = self.monomer.zMon_sol[-1]*np.ones(T)
            self.monomer.corruption = self.monomer.corruption[-1]*np.ones(T)
            self.monomer.exitTracker = self.monomer.exitTracker[-1]*np.ones(T)
        

        if self.store==0:
            if self.collisions:
                self.monomer.vRel_sol = np.array(self.monomer.vRel_sol)
            for item in self.monomer.sec_sol.keys():
                self.monomer.sec_sol[item] = np.array(self.monomer.sec_sol[item])
                
        if self.trackice:
            # Calculate element ratios.
            #iceMassSum = np.zeros(len(self.monomer.t_sol))
            for n in range(self.iceNum):
                self.monomer.ice_sol[self.disk.iceList[n]] = np.array(self.monomer.ice_sol[self.disk.iceList[n]])
                #iceMassSum += self.monomer.ice_sol[self.disk.iceList[n]]
            self.monomer.iceTot_sol = np.array(self.monomer.iceTot_sol)
            if self.store==0:
                self.monomer.ele_sol = self.calcElements()
            # Abundance ratio (number densities, not mass ratio)
            #1.00797, 12.011, 14.0067, 15.9994, 32.06
                self.monomer.sec_sol["ratC/O"] = self.monomer.ele_sol[:,1]/self.monomer.ele_sol[:,3]*(15.9994/12.011)
                self.monomer.sec_sol["ratN/O"] = self.monomer.ele_sol[:,2]/self.monomer.ele_sol[:,3]*(15.9994/14.0067)
                self.monomer.sec_sol["ratS/O"] = self.monomer.ele_sol[:,4]/self.monomer.ele_sol[:,3]*(15.9994/32.06)
	
            self.integrationTime = tocTot-ticTot
            self.corruptionAmount = np.sum(self.monomer.corruption*self.monomer.t_sol)/np.sum(self.monomer.t_sol)
        
    
    

class Disk:

    
    def __init__(self, species=["H2", "H2O", "CO"], folder="../BackgroundModels/ShampooCodingPaper/vFrag5", modelName="ProDiMo.out", t_index="{:04d}".format(5), order=1, verbose=-1):

        
        self.verbose=verbose
        self.order = order
        self.diskFolder = folder
        if species!="all":
            self.species = species
            self.all = False
        else:
            self.species = (np.loadtxt(folder+"/AdsorptionEnergies.in", dtype="str", comments="#", usecols=(0), encoding=None)).tolist()
            self.all = True
            
        print(self.species)
        print(self.diskFolder)
        try:
            self.model = pread.read_prodimo(self.diskFolder, filename=modelName, td_fileIdx=t_index)
            self.model.dust = pread.read_dust(self.diskFolder)
        except:
            print("Appropriate background disk model timestamp not found, falling back on default disk loading routine...")
            self.model = pread.read_prodimo(self.diskFolder, td_fileIdx=None)
            self.model.dust = pread.read_dust(self.diskFolder)

        self.parameters = self.model.params.mapping # disk parameters from parameter.out are loaded in this dictionary. Used to assign to background model parameters in SHAMPOO.
        self.prepareInterpol()
        self.doInterpol()

    
    def prepareInterpol(self):
        #print("Starting interpolation")

        # tg, td, muH, rhog, rhod, ng, nd, pressure, soundspeed, nmol[:,:,self.model.spnames["234"]]
        # spnames: translation table from species to nspec

        # Old version
        #self.rVals = self.model.x.flatten()
        #self.zVals = self.model.z.flatten()/self.rVals
        #self.points = np.stack((self.rVals, self.zVals), axis=1)
        self.rVals = self.model.x[:,0]
        self.zVals = (self.model.z/self.model.x)[0,:]

        self.data = {}

        # Prepare basic quantities
        self.data["Tg"] = self.model.tg
        self.data["Td"] = self.model.td

        # Note that we actually calculate log10 of these quantities.
        self.data["rhog"] = np.log10(1e3*self.model.rhog)
        self.data["rhod"] = np.log10(1e3*self.model.rhod)
        self.data["nd"] = np.log10(1e6*self.model.nd)	


        #self.pressure = LinearNDInterpolator(zipped, self.model.pressure.flatten()) #### CONVERT TO SI
        self.data["soundspeed"] = 1e3*self.model.soundspeed
        self.data["pressure"] = 1e-1*self.model.pressure

        self.data["chiRT"] = np.log10(self.model.chiRT)


        # Initialize chemical species abundances
        self.iceList = []
        self.elementDict = {} # A dictionary containing all ice molecules considered, and the number of each element present.
        # currently we only consider H,C,N,O,S. With the value of elementDict containing a np array with 5 elements representing the number of atoms of each element in a given molecule.
        
        N = len(self.species)
        n = 0
        if self.verbose>-1:
            print("Received",N,"species. Attempting to load background model number densities...")
        while n<N:
            spec = self.species[n]
            #print(spec)
            if (spec in self.model.spnames.keys())and(spec+"#" in self.model.spnames.keys())and("-" not in spec)and("+" not in spec):
                indexGas = self.model.spnames[spec]
                self.data["gasAbun"+spec] = np.log10(1e6*self.model.nmol[:,:,indexGas])
                self.iceList.append(spec)
                self.elementDict[spec] = self.elementCount(spec)
                                
                indexIce = self.model.spnames[spec+"#"]
                self.data["iceAbun"+spec] = np.log10(1e6*self.model.nmol[:,:,indexIce])
                self.data["totAbun"+spec] = np.log10(1e6*(self.model.nmol[:,:,indexGas]+self.model.nmol[:,:,indexIce]))
                #print("totAbun"+spec, self.data["totAbun"+spec])
                #print((self.data["gasAbun"+spec])[10,0], (self.data["iceAbun"+spec])[10,0],(self.data["totAbun"+spec])[10,0])
                n +=1
            else:
                print("Omitting species : "+spec)
                (self.species).remove(spec)
                N = len(self.species)
        
        if self.verbose>-1:
            print("Sucessfully loaded",N, "species.")
            print(self.species)
            self.showElementCountResults()
        
        
            
                
    
    def elementCount(self, name):
        """
        Counts the number of atoms of each element for each ice species.
        """
        
        L = len(name)
        l = 0
        elements = np.zeros(5, dtype="int")
        elementlist = ["H", "C", "N", "O", "S"]
    
        while l<L:
            if name[l] in elementlist:
                if l<L-1:
                    if name[l+1].isnumeric():
                        elements[elementlist.index(name[l])] += int(name[l+1])
                    elif name[l+1] in ["a","i","e"]:
                        pass
                    else:
                        elements[elementlist.index(name[l])] += 1
                else:
                    elements[elementlist.index(name[l])] += 1
            l += 1

        return elements
    
    def showElementCountResults(self):
        """
        Auxiliary function to check whether elementCount did its work correctly.
        """
        
        longestName = max([len(spec) for spec in self.species])
        
        print(" "*(longestName+2),"H","C","N","O","S")
        for spec in self.species:
            
            print(spec," "*(longestName-len(spec))+":",(self.elementDict[spec])[0],(self.elementDict[spec])[1], (self.elementDict[spec])[2], (self.elementDict[spec])[3], (self.elementDict[spec])[4])
            
    
    def doInterpol(self):
        self.interpol = {}

        ##print("Doing interpolation")
        for name in self.data.keys():
            
            self.interpol[name] = RectBivariateSpline(self.rVals, self.zVals, self.data[name], kx=self.order, ky=self.order)
        print("Finished doing interpolation")

        
    def expectedIce(self, rEva, zEva, species=None, label="ice"):
        """
        Auxiliary function to find the expected amount of ice in monomer masses based off the loaded background model.
               
        rEva is in AU
        zEva is in AU
        species - if None the total amount of ice is calculated. Otherwise a string denoting the chemical species for which
        the abundance has to be calculated. (e.g. "H2O", "CO")
        
        return - Average ice mass in monomer masses expected on a locally processed monomer given the background model.
        
        TO DO: Find a less clunky way to deal with the loading of dataName and dataNum, now adsorption energies are reloaded on every
        function call.
        """

        filepath = self.diskFolder
        masses = {}
        dataName = np.loadtxt(filepath+"/AdsorptionEnergies.in", dtype="str", comments="#", usecols=(0), encoding=None)
        dataNum = np.loadtxt(filepath+"/AdsorptionEnergies.in", dtype="float", comments="#", usecols=(1,2), encoding=None) 

        uTOkg = 1.660539066e-27
        auTOm = 1.496e11
        
        N = len(dataName)
        for n in range(N):
            masses["m"+dataName[n]] = dataNum[n,1]*uTOkg

        rhod = 10**(self.interpol["rhod"](rEva, zEva/rEva, grid=False)) 
        
        if species==None:

            absice = 0
            N = len(self.iceList)
            for n in range(N):
                diff = masses["m"+self.iceList[n]]*10**(self.interpol[label+"Abun"+self.iceList[n]](rEva, zEva/rEva, grid=False)) 
                absice += diff
            iceAbun = absice/rhod
        else:
            specice = masses["m"+species]*10**(self.interpol[label+"Abun"+species](rEva, zEva/rEva, grid=False)) 
            
            iceAbun = specice/rhod
        
        return iceAbun


class HomeAggregate:

    
    def __init__(self, model, size):
        """
        Initializes a variety of parameters associated with the home aggregate.
        """

        self.initProps(size, model)


    
    def initProps(self, size, model):
        self.prop = {}

        if model.pisoBenchmark:
            self.prop["phi"] = 1
            self.prop["rhoAgg"] = 2000
        elif model.phi!=None:
            self.prop["phi"] = model.phi
            self.prop["rhoAgg"] = 2094*self.prop["phi"]
        else:
            self.prop["phi"] = float(model.paraDict["phi"])
            self.prop["rhoAgg"] = 2094*self.prop["phi"]
        self.prop["sAgg"] = size # in m; initial size is equal to the monomer size.
        
        self.prop["mAgg"] = 4/3*np.pi*self.prop["sAgg"]**3*self.prop["rhoAgg"]


class Monomer:
    
    
    def __init__(self, model, r0, z0, home_aggregate=None, size=0.05e-6):
        """
            Initializes a monomer. For convenience, r0 and z0 are in AU.
        """

        if home_aggregate==None:
            self.homeAggregate = HomeAggregate(model, size=size)
        else:
            self.homeAggregate = home_aggregate
        self.initR = r0
        self.initZ = z0

        self.initProps(model, size)

    
    def initProps(self, model, size):
        self.prop = {}

        self.exposed = True
        self.prop["zMon"] = float(model.paraDict["zMon"])
        self.prop["sMon"] = size # in m
        try:
            self.prop["zCrit"] = model.paraDict["zCrit"]*size
        except:
            self.prop["zCrit"] = 2*size # critical depth inside a home aggregate where the monomer is no longer exposed
        if model.pisoBenchmark:
            self.prop["rho_mat"] = 2000
        else:
            self.prop["rho_mat"] = float(model.paraDict["rhoMon"]) # monomer material density in kg/m3. Informed by ProDiMo
        self.prop["mMon"] = 4/3*np.pi*self.prop["rho_mat"]*self.prop["sMon"]**3
        self.prop["yield"] = float(model.paraDict["yield"]) # Yield rate for UV photodesorption. Experimental values for all major molecules are within 1 order of magnitude of this value (Oberg 2009).
        self.prop["Nads"] = float(model.paraDict["Nads"]) # Number of adsorption sites per square meter (Piso+ 2015; Hollenbach+ 2009) 
        self.prop["Nact"] = int(model.paraDict["Nact"]) # Number based on the range of values 2-4 stated in Cuppen+ 2017.
  


##############################################################################################################
# Start of monomer processing module, used for statistical analysis 
# NOTE! Requires significant memory for >100 monomers (order of GB to 10s of GB) for 10000+ monomers.
##############################################################################################################

class Analysis:
    
    def __init__(self, disk, homepath="/net/zach/data/users/moosterloo/PhD/Papers/ShampooSciencePaper"):
        
        self.disk = disk
        self.homepath = homepath
        
        # Calculate the pdfs for the weight calculation.
        self.init_pdfs()
   
        self.auTOm = 1.496*1e11
    
    def init_pdfs(self):
        
        self.Pn = 1/self.disk.model.x
        self.Pp = self.disk.model.rhod/np.sum(self.disk.model.rhod)
        
        R,Z = self.disk.model.x.shape        
        self.zGrid = np.zeros((R,Z))
        self.rGrid = np.zeros((R,Z))

        for r in range(0,R):
            for z in range(0,Z):
                zj = self.disk.model.z[r,z]
                ri = self.disk.model.x[r,z]

                if z>0 and z<(Z-1):
                    zup = self.disk.model.z[r,z+1]/self.disk.model.x[r,z+1]-self.disk.model.z[r,z]/self.disk.model.x[r,z]
                    zdown = self.disk.model.z[r,z]/self.disk.model.x[r,z]-self.disk.model.z[r,z-1]/self.disk.model.x[r,z-1]

                elif z==0:
                    zup = self.disk.model.z[r,1]/self.disk.model.x[r,1] # may require an extra factor 2? Check!
                    zdown = 0
                else:
                    zup = self.disk.model.z[r,-1]/self.disk.model.x[r,-1]-self.disk.model.z[r,z]/self.disk.model.x[r,z]
                    zdown = self.disk.model.z[r,z]/self.disk.model.x[r,z]-self.disk.model.z[r,z-1]/self.disk.model.x[r,z-1]

                self.zGrid[r,z] = 2*(zup+zdown) 

                if r>0 and r<(R-1):
                    rup = self.disk.model.x[r+1,z]-self.disk.model.x[r,z]
                    rdown = self.disk.model.x[r,z]-self.disk.model.x[r-1,z]
                elif r==0:
                    rup = self.disk.model.x[1,z]
                    rdown = 0.08
                else:
                    rup = self.disk.model.x[-1,z]-self.disk.model.x[r,z]
                    rdown = self.disk.model.x[r,z]-self.disk.model.x[r-1,z]

                self.rGrid[r,z] = 2*(rup+rdown)
        
    
    def initializeMonomerData(self):
        """
        Auxiliary function that (re-)initializes the main analysis data structure.
        """
        
        self.monomerData = {}
        
        if self.removeCorrupted != "noIce":
            self.keyList = ["n","seed", "t","r","z","nx","nz","zm","sa","exposed","weights", "corruption", "exitIndex"]
        else:
            self.keyList = ["n","seed","t","r","z","nx","nz","weights"]
        
        
        self.keyList += self.environmentList + self.extremeList
        
        for key in self.keyList:
            self.monomerData[key] = [None]*self.monoNum
        if self.removeCorrupted != "noIce":
            for ice in self.disk.iceList:
                self.monomerData[ice] = [None]*self.monoNum              
        
            
    def loadModels(self, loadPath="./Simulations/NonLocal1/", monoNum=100, read=False, cleaningIndex=7, removeCorrupted="selective", environmentList=["Tg","Td"], extremeList=["maxTd","minTd"], refractories=False, refractoryPath=None, refractoryExcl=[]):
        """
        Keyword explanation:
        
        monoNum:
        
        read:
            ----
        cleaningIndex:
            Remove monomer data points where the ice evolution exit index is higher than this value.
            Note that the index is included in the data set (i.e. if cleaningIndex==3, 3 is still included in 
            the data.)
        removeCorrupted:
            "selective": Remove monomer data points with exit index higher than the cleaningIndex.
            "noIce": Do not load any data related to ice evolution.
        """
         
        
        if loadPath==None:
            loadPath = self.homepath+"/Simulations/NonLocal2"
        if refractoryPath==None:
            self.refractoryPath = "/net/zach/data/users/moosterloo/PhD/Papers/ShampooGGCHEMPaper/GGCHEM/bulkoutput/"
        else:
            self.refractoryPath = refractoryPath
        
        self.monoNum = monoNum
        self.cleaningIndex = cleaningIndex
        self.removeCorrupted = removeCorrupted
        self.environmentList = environmentList
        self.extremeList = extremeList
        self.refractories = refractories
        self.refractoryExcl = refractoryExcl # list with molecules excluded from analysis
        
        self.initializeMonomerData()
        if self.refractories:
            self.initRefr = True

        self.avgQuants = {}
        
        self.modNum = 0
        print("Loading ",monoNum," models...")
        toc = process_time()
        for filename in os.listdir(loadPath[0:-1]):
            if (filename.endswith(".pickle")) and (self.modNum<self.monoNum):               
                try:
                    mod = pickle.load(open(loadPath+filename, "rb"))

                    if self.refractories:
                        self.appendGGCHEMData(mod)
                    if np.any(mod.monomer.corruption==1)and(self.removeCorrupted=="rigorous"):
                        self.monoNum -= 1
                    else:               
                        self.appendModelData(mod)
                        self.modNum += 1

                    tic = process_time()
                    elapsed = tic-toc
                    estleft = elapsed/(self.modNum)*(monoNum-self.modNum)
                    print("Progress: {:.2f} %, Est. time: {:.2f} s".format((self.modNum)/monoNum*100, estleft), end="\r")
                except:
                    pass
        print("Concatenating monomers...")
        for item in self.monomerData.keys():
            self.monomerData[item] = np.concatenate(self.monomerData[item], axis=None)
        
        # See scipy ice evolution method for the meaning of the various indices. Ideally you want to clean above 3.
        if self.removeCorrupted != "noIce":
            print("Cleaned index statistics")
            for n in range(8):
                indexPerc = 100*len((self.monomerData["exitIndex"])[self.monomerData["exitIndex"]==n])/len(self.monomerData["exitIndex"])
                print("Index ",n,"-   {:.5f} %".format(indexPerc))
        
        self.monomerData = pd.DataFrame.from_dict(self.monomerData)      

        print("Loaded",self.modNum,"monomers")   
               
    
    
                
    def appendModelData(self, mod):
                
        (self.monomerData["n"])[self.modNum] = np.ones(len(mod.monomer.t_sol))*self.modNum # label to track unique monomers
        (self.monomerData["seed"])[self.modNum] = np.ones(len(mod.monomer.t_sol))*mod.seedStart# seed to track unique monomers
        (self.monomerData["t"])[self.modNum] = mod.monomer.t_sol
        (self.monomerData["r"])[self.modNum] = mod.monomer.r_sol
        (self.monomerData["z"])[self.modNum] = abs(mod.monomer.z_sol)
        
        if self.removeCorrupted != "noIce":
            (self.monomerData["corruption"])[self.modNum] = mod.monomer.corruption
            (self.monomerData["exitIndex"])[self.modNum] = mod.monomer.exitTracker
            
            (self.monomerData["zm"])[self.modNum] = mod.monomer.zMon_sol
            (self.monomerData["sa"])[self.modNum] = mod.monomer.sAgg_sol

            (self.monomerData["exposed"])[self.modNum] = np.array(mod.monomer.sec_sol["exposed"])
        
        nx, nz = self.assignGridPoint(mod.monomer.r_sol ,mod.monomer.z_sol)
        
        (self.monomerData["nx"])[self.modNum] = nx
        (self.monomerData["nz"])[self.modNum] = nz
  
        
        weights = self.calculateWeights(mod, nx, nz)
        (self.monomerData["weights"])[self.modNum] = weights
        
        # Deal with environment tracking
        for name in self.environmentList:
            quant = self.disk.interpol[name](mod.monomer.r_sol, abs(mod.monomer.z_sol)/mod.monomer.r_sol, grid=False)
            
            if (name in ["rhog", "rhod", "nd", "chiRT"] or (name[0:3] in ["gas", "ice", "tot"]) or (name[0:5] in ["rhoda", "numda"]) or ("Abun" in name)):
                (self.monomerData[name])[self.modNum] = 10**quant
            else:
                (self.monomerData[name])[self.modNum] = quant
        
        # Adds maxTd and minTd, which track the max and min dust temperature so far.
        self.trackExtremes(mod)
        
                
        if self.removeCorrupted != "noIce":
            for ice in self.disk.iceList:
                # Do these flags follow a particle?
                iceList = mod.monomer.ice_sol[ice]

                mod.monomer.corruption[iceList<0] = 1
                iceList[iceList<0] = 0 # This might be an issue...

                mod.monomer.corruption[np.isinf(iceList) | np.isnan(iceList)] = 1
                iceList[np.isinf(iceList) | np.isnan(iceList)] = 0 # This probably solves itsself

                (self.monomerData[ice])[self.modNum] = iceList

            if self.removeCorrupted=="selective":
                cond = (self.monomerData["exitIndex"])[self.modNum]<=self.cleaningIndex
                for item in self.monomerData.keys():
                    #print(item, type((self.monomerData[item])[self.modNum]))
                    (self.monomerData[item])[self.modNum] = ((self.monomerData[item])[self.modNum])[cond]
        
        
    def assignGridPoint(self, rArr,zArr):
        """
        Discretizes the radial and vertical coordinates.
        """
        
        T = len(rArr)
        nxArr = np.zeros(T)
        nzArr = np.zeros(T)
        
        for t in range(T):
            r = rArr[t]
            z = zArr[t]
            
            nxArr[t] = np.argmin(abs(r-self.disk.model.x[:,0]))
            nzArr[t] = np.argmin(abs(abs(z/r)-self.disk.model.z[-1,:]/self.disk.model.x[-1,:]))

        return nxArr, nzArr        
    
    def calculateWeights(self, mod, nx, nz):
        """
        Calculates the monomer data weights, informed from
        
        - the timestep size
        - the density at the origin compared to the density at the gridpoint (ratio)
        """
        
        # First weight: The timesteps:
        T = len(mod.monomer.t_sol)
        
        timesteps = np.zeros(T)
        
        timesteps[0:-1] = mod.monomer.t_sol[1::]-mod.monomer.t_sol[0:-1]
        timesteps[-1] = 0 # we do not want to include the last timestep of each simulation        
    
        # The second weight comes from differences in the mass pdf from the disk density structure and 
        # the pdf used in the sampling.
        xind = int(nx[0])
        zind = int(nz[0])        
        
        proprat = np.ones(T)*self.Pp[xind, zind]/(self.Pn[xind,zind]*self.rGrid[xind,zind]*self.zGrid[xind,zind])
    
        # Calculate the total weight.
        weights = timesteps*proprat

        return weights     
    
    
    
    def calcAvgQuant(self, quant):
        """
        Auxiliary function which calculates the weighted average grid when called on in a plotting script.
        """

        R = self.disk.model.nx
        Z = self.disk.model.nz

        avg_grid = np.zeros((R,Z))
        
        for r in range(R):
            for z in range(Z):
                print(r,z,end="\r")
                redData = self.monomerData[(self.monomerData["nx"]==r) & (self.monomerData["nz"]==z)]  
                if quant=="zm/sa":
                    quantArr = redData["zm"].values/redData["sa"].values
                else:
                    quantArr = redData[quant].values
                weightsArr = redData["weights"].values
                avg_grid[r,z] = np.sum(quantArr*weightsArr)/np.sum(weightsArr)
        return avg_grid

# Analysis functions related to the refractories.    
    
    
    def trackExtremes(self, mod):
        
        T = len(mod.monomer.t_sol)
        
        for name in self.extremeList:
            # Define the arrays in the dataframe directly (check whether this is efficient)
            (self.monomerData[name])[self.modNum] = np.zeros(T)
        
        
        extreme_max_value = ((self.monomerData["Td"])[self.modNum])[0]
        extreme_min_value = ((self.monomerData["Td"])[self.modNum])[0]
        for t in range(0,T):
            current_value = ((self.monomerData["Td"])[self.modNum])[t]
            extreme_max_value = max(extreme_max_value, current_value)   
            extreme_min_value = min(extreme_min_value, current_value)   
            
            ((self.monomerData["maxTd"])[self.modNum])[t] = extreme_max_value 
            ((self.monomerData["minTd"])[self.modNum])[t] = extreme_min_value
            
    def determineComposition(self, path, name):

        file = path+name

        data   = open(file)
        dummy  = data.readline()
        dimens = data.readline()
        dimens = np.array(dimens.split())
        NELEM  = int(dimens[0])
        NMOLE  = int(dimens[1])
        NDUST  = int(dimens[2])
        NPOINT = int(dimens[3])
        header = data.readline()
        data.close()
        dat = np.loadtxt(file,skiprows=3)
        keyword = np.array(header.split())
        NPOINT = len(dat[0:])

        bar   = 1.E+6                    # 1 bar in dyn/cm2 
        Tg    = dat[:,0]                 # T [K]
        nHtot = dat[:,1]                 # n<H> [cm-3]
        lognH = np.log10(nHtot)          
        press = dat[:,2]                 # p [dyn/cm2]
        Tmin  = np.min(Tg)
        Tmax  = np.max(Tg)

        iii   = np.where((Tg>Tmin) & (Tg<Tmax))[0]

        idx = (np.abs(Tg[iii] - 600)).argmin()
        el = int(np.where(keyword == 'el')[0])

        # Look up all condensates at the end of the condensation sequence.

        with open('GGchem_sol.txt', 'w') as f:
            print('T', Tg, file=f)

        #names = list(keyword[el+1:el+1+NELEM])
        #names = ['CaMgSi2O6', 'Mg2SiO4', 'MgSiO3', 'FeS', 'FeS2', 'SiO2']
        names = ["H", "He", "C", "N", "O", "Na", "Mg", "Si", "Fe", "Al", "Ca", "Ti", "S", 
                 "Cl", "K", "Li", "Mn", "Ni", "Cr", "V", "W", "Zr", "F", "P"]
        solids = []
        smean = []
        ymax = -100.0
        iii   = np.where((Tg>Tmin) & (Tg<Tmax))[0]

        Ind = []

        compList = []
        molList = []

        for i in range(4+NELEM+NMOLE,4+NELEM+NMOLE+NDUST,1):
            solid = keyword[i]
            solids.append(solid[1:])
            smean.append(np.mean(dat[iii,i])) 
            ind = np.where(keyword == 'n'+solid[1:])[0]
            if (np.size(ind) == 0): continue
            ind = ind[0]
            yy = dat[:,ind] # log10 nsolid/n<H>
            ymax = np.max([ymax,np.max(yy[iii])])
            ymin = -99
            if ((yy[iii])[-1]>ymin) and (solid[1::] not in self.refractoryExcl):
                compList.append((yy[iii])[-1])
                molList.append(solid[1::])

        compList = np.array(compList)
        compList = 10**(np.array(compList))
        compList /= np.sum(compList)

        return molList, compList


    
    def appendGGCHEMData(self, mod):
        """
        Looks up the ggchem condensation sequence for the given model seed, and subsequently initializes the data into
        self.refractories.
        """
        
        fileName = "Static_Cond"+str(mod.seedStart)+".dat"
        molList, compList = self.determineComposition(self.refractoryPath, fileName)
        Ncomp = len(molList)
       
        
      
        if self.initRefr: # Initialize refractory data structures if this is the first function call.
            self.monomerSeeds = [0]*self.monoNum
            self.monomerSeeds[0] = mod.seedStart
            self.moleculeNames = list(molList)
            self.refractoryData = np.zeros((self.monoNum,len(self.moleculeNames)))
            self.refractoryData[0,:] = compList
            print(molList)
            self.initRefr = False
        else:
            N = self.refractoryData.shape[1] # molecule number

            self.monomerSeeds[self.modNum] = mod.seedStart
            

            for n in range(Ncomp):
                if molList[n] in self.moleculeNames:
                    molInd = self.moleculeNames.index(molList[n])
                    self.refractoryData[self.modNum,molInd] = compList[n]
                else:
                    print("Appending molecule:",molList[n])
                    self.moleculeNames.append(molList[n])
                    self.refractoryData = np.append(self.refractoryData, np.zeros((self.monoNum,1)), axis=1)
                    self.refractoryData[self.modNum,-1] = compList[n]
          