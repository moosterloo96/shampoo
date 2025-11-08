import numpy as np
import pandas as pd

class InitializeParameters:

    def loadPara(self, parameter_folder=None):
        if parameter_folder != None:
            path = parameter_folder
        else:
            path = ""
        paraDict = dict(pd.read_csv(path + "shampooInput.csv", header=None, usecols=(0, 1), comment="#", delimiter="\t",
                                    skipinitialspace=True).values)
        for item in paraDict.keys():
            if paraDict[item] == "True":
                paraDict[item] = True
            elif paraDict[item] == "False":
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
        self.para["mElements"] = 1e-3 * np.array([1.00797, 12.011, 14.0067, 15.9994, 32.06])
        # molar masses in kg/mol for the elements H, C, N, O and S

        # Unit conversion factors
        self.sTOyr = 3600 * 24 * 365.25
        self.sTOMyr = self.sTOyr * 1e6
        self.auTOm = 1.496e11

        # TODO: Move this to the disk class.
        # Adsorption energies and masses
        dataName = np.loadtxt(filepath + "/AdsorptionEnergies.in", dtype="str", comments="#", usecols=(0), encoding=None)
        dataNum = np.loadtxt(filepath + "/AdsorptionEnergies.in", dtype="float", comments="#", usecols=(1, 2),
                             encoding=None)
        N = len(dataName)
        for n in range(N):
            self.para["Eads" + dataName[n]] = dataNum[n, 0] * self.para["kB"]
            self.para["m" + dataName[n]] = dataNum[n, 1] * self.para["u"]

        if self.pisoBenchmark:
            self.para["EadsH2O"] = 5800 * self.para["kB"]

        # Ice formation properties
        if self.pisoBenchmark:
            self.para["NadsRef"] = 1e19
        else:
            self.para["NadsRef"] = float(self.paraDict["NadsRef"])

        # Molecule sticking properties; see He+ 2016.
        self.para["alphaStick"] = float(self.paraDict["alphaStick"])
        self.para["betaStick"] = float(self.paraDict["betaStick"])
        self.para["gammaStick"] = float(self.paraDict["gammaStick"])

        # Disk model parameters - Assigned from self.entities.parameter; we here do the dtype and unit conversions.
        # This is needed since the params-object from prodimopy has all data from parameter.out in the form of strings and cgs units.
        self.para["a_settle"] = float(self.disk.parameters["A_SETTLE"])  # the turbulent viscosity/dust settling parameters
        self.para["Mstar"] = float(self.disk.parameters["MSTAR"]) / 1e3  # mass of central star in kg
        self.para["Rtaper"] = float(self.disk.parameters["RTAPER"]) / 1e2  # tapering radius

        # Dust parameters
        self.para["nsize"] = len(self.disk.model.dust.asize)  # number of mass bins
        self.para["rho_mat"] = 2094  # Material density of the background dust # FIND THIS BACK IN PARAMETER.OUT!!!!

        # Scale height/flaring properties
        self.para["H0"] = float(self.disk.parameters["MCFOST_H0"]) * self.auTOm
        self.para["r0"] = float(self.disk.parameters["MCFOST_RREF"]) * self.auTOm
        self.para["beta_flare"] = float(self.disk.parameters["MCFOST_BETA"])

        # Inner and outer entities radii (in meters)
        self.para["r_inner"] = self.disk.model.x[0, 0] * self.auTOm
        self.para["r_outer"] = self.disk.model.x[-1, 0] * self.auTOm
        self.para["r_inner_cutoff"] = float(self.paraDict["r_inner_cutoff"])

        # Chemical parameters
        self.para["sticking"] = 1  # the sticking factor.
        self.para["sig_mol"] = 2e-19  # Molecular cross section, taken from Krijt+ 2018 / Okuzumi+ 2012
        self.para["ds"] = 1e-10  # Thickness of a single ice monolayer (see Oosterloo+ 2023)
        self.para["Ediff/Eads"] = 1 / 2  # Ratio between diffusion energy barrier and binding energy
        self.para["tauMin"] = 1 * self.sTOyr  # Minimum and maximum diffusion timescales.
        self.para[
            "tauMax"] = 1e7 * self.sTOyr  # Maximum value for MFP diffusion timescale (convert locally to sAgg timescale).
        self.para["tauMinInt"] = 1 * self.sTOyr  # Dummy variables to be updated
        self.para["tauMaxInt"] = 1e7 * self.sTOyr
        if self.pisoBenchmark:
            self.para["muH"] = 2.35
        else:
            self.para["muH"] = 2.3  # Informed from Krijt+ 2018
        self.oldNu = False  # Use the desorption attempt frequence from Tielens+ 198x

        # Radiative properties
        self.para[
            "FDraine"] = 1.921e12  # Draine flux of UV photons between 91.2 nm and 205 nm in m-2s-1 Taken from Woitke+ 2009

        # Coagulation properties
        self.grainSizes = self.disk.model.dust.asize * 1e-6  # convert to SI
        self.para["v_frag"] = float(self.paraDict[
                                        "v_frag"])  # Fragmentation velocity (value taken from Krijt & Ciesla 2016). 1 m/s might be better (Guttler et al. 2010; Birnstiel et al. 2010)
        self.para["del_v_frag"] = self.para["v_frag"] / 5
        self.para["x_frag"] = float(
            self.paraDict["x_frag"])  # the power law slope for collision fragments (value from Birnstiel+ 2010)