import numpy as np
import pandas as pd
from time import process_time
import os
import pickle

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

        self.auTOm = 1.496 * 1e11

    def init_pdfs(self):

        self.Pn = 1 / self.disk.model.x
        self.Pp = self.disk.model.rhod / np.sum(self.disk.model.rhod)

        R, Z = self.disk.model.x.shape
        self.zGrid = np.zeros((R, Z))
        self.rGrid = np.zeros((R, Z))

        for r in range(0, R):
            for z in range(0, Z):
                zj = self.disk.model.z[r, z]
                ri = self.disk.model.x[r, z]

                if z > 0 and z < (Z - 1):
                    zup = self.disk.model.z[r, z + 1] / self.disk.model.x[r, z + 1] - self.disk.model.z[r, z] / \
                          self.disk.model.x[r, z]
                    zdown = self.disk.model.z[r, z] / self.disk.model.x[r, z] - self.disk.model.z[r, z - 1] / \
                            self.disk.model.x[r, z - 1]

                elif z == 0:
                    zup = self.disk.model.z[r, 1] / self.disk.model.x[r, 1]  # may require an extra factor 2? Check!
                    zdown = 0
                else:
                    zup = self.disk.model.z[r, -1] / self.disk.model.x[r, -1] - self.disk.model.z[r, z] / \
                          self.disk.model.x[r, z]
                    zdown = self.disk.model.z[r, z] / self.disk.model.x[r, z] - self.disk.model.z[r, z - 1] / \
                            self.disk.model.x[r, z - 1]

                self.zGrid[r, z] = 2 * (zup + zdown)

                if r > 0 and r < (R - 1):
                    rup = self.disk.model.x[r + 1, z] - self.disk.model.x[r, z]
                    rdown = self.disk.model.x[r, z] - self.disk.model.x[r - 1, z]
                elif r == 0:
                    rup = self.disk.model.x[1, z]
                    rdown = 0.08
                else:
                    rup = self.disk.model.x[-1, z] - self.disk.model.x[r, z]
                    rdown = self.disk.model.x[r, z] - self.disk.model.x[r - 1, z]

                self.rGrid[r, z] = 2 * (rup + rdown)

    def initializeMonomerData(self):
        """
        Auxiliary function that (re-)initializes the main analysis data structure.
        """

        self.monomerData = {}

        if self.removeCorrupted != "noIce":
            self.keyList = ["n", "seed", "t", "r", "z", "nx", "nz", "zm", "sa", "exposed", "weights", "corruption",
                            "exitIndex"]
        else:
            self.keyList = ["n", "seed", "t", "r", "z", "nx", "nz", "weights"]

        self.keyList += self.environmentList + self.extremeList

        for key in self.keyList:
            self.monomerData[key] = [None] * self.monoNum
        if self.removeCorrupted != "noIce":
            for ice in self.disk.iceList:
                self.monomerData[ice] = [None] * self.monoNum

    def loadModels(self, loadPath="./Simulations/NonLocal1/", monoNum=100, read=False, cleaningIndex=7,
                   removeCorrupted="selective", environmentList=["Tg", "Td"], extremeList=["maxTd", "minTd"],
                   refractories=False, refractoryPath=None, refractoryExcl=[]):
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

        if loadPath == None:
            loadPath = self.homepath + "/Simulations/NonLocal2"
        if refractoryPath == None:
            self.refractoryPath = "/net/zach/data/users/moosterloo/PhD/Papers/ShampooGGCHEMPaper/GGCHEM/bulkoutput/"
        else:
            self.refractoryPath = refractoryPath

        self.monoNum = monoNum
        self.cleaningIndex = cleaningIndex
        self.removeCorrupted = removeCorrupted
        self.environmentList = environmentList
        self.extremeList = extremeList
        self.refractories = refractories
        self.refractoryExcl = refractoryExcl  # list with molecules excluded from analysis

        self.initializeMonomerData()
        if self.refractories:
            self.initRefr = True

        self.avgQuants = {}

        self.modNum = 0
        print("Loading ", monoNum, " models...")
        toc = process_time()
        for filename in os.listdir(loadPath[0:-1]):
            if (filename.endswith(".pickle")) and (self.modNum < self.monoNum):
                try:
                    mod = pickle.load(open(loadPath + filename, "rb"))

                    if self.refractories:
                        self.appendGGCHEMData(mod)
                    if np.any(mod.monomer.corruption == 1) and (self.removeCorrupted == "rigorous"):
                        self.monoNum -= 1
                    else:
                        self.appendModelData(mod)
                        self.modNum += 1

                    tic = process_time()
                    elapsed = tic - toc
                    estleft = elapsed / (self.modNum) * (monoNum - self.modNum)
                    print("Progress: {:.2f} %, Est. time: {:.2f} s".format((self.modNum) / monoNum * 100, estleft),
                          end="\r")
                except:
                    pass
        print("Concatenating monomers...")
        for item in self.monomerData.keys():
            self.monomerData[item] = np.concatenate(self.monomerData[item], axis=None)

        # See scipy ice evolution method for the meaning of the various indices. Ideally you want to clean above 3.
        if self.removeCorrupted != "noIce":
            print("Cleaned index statistics")
            for n in range(8):
                indexPerc = 100 * len((self.monomerData["exitIndex"])[self.monomerData["exitIndex"] == n]) / len(
                    self.monomerData["exitIndex"])
                print("Index ", n, "-   {:.5f} %".format(indexPerc))

        self.monomerData = pd.DataFrame.from_dict(self.monomerData)

        print("Loaded", self.modNum, "monomers")

    def appendModelData(self, mod):

        (self.monomerData["n"])[self.modNum] = np.ones(
            len(mod.monomer.t_sol)) * self.modNum  # label to track unique monomers
        (self.monomerData["seed"])[self.modNum] = np.ones(
            len(mod.monomer.t_sol)) * mod.seedStart  # seed to track unique monomers
        (self.monomerData["t"])[self.modNum] = mod.monomer.t_sol
        (self.monomerData["r"])[self.modNum] = mod.monomer.r_sol
        (self.monomerData["z"])[self.modNum] = abs(mod.monomer.z_sol)

        if self.removeCorrupted != "noIce":
            (self.monomerData["corruption"])[self.modNum] = mod.monomer.corruption
            (self.monomerData["exitIndex"])[self.modNum] = mod.monomer.exitTracker

            (self.monomerData["zm"])[self.modNum] = mod.monomer.zMon_sol
            (self.monomerData["sa"])[self.modNum] = mod.monomer.sAgg_sol

            (self.monomerData["exposed"])[self.modNum] = np.array(mod.monomer.sec_sol["exposed"])

        nx, nz = self.assignGridPoint(mod.monomer.r_sol, mod.monomer.z_sol)

        (self.monomerData["nx"])[self.modNum] = nx
        (self.monomerData["nz"])[self.modNum] = nz

        weights = self.calculateWeights(mod, nx, nz)
        (self.monomerData["weights"])[self.modNum] = weights

        # Deal with environment tracking
        for name in self.environmentList:
            quant = self.disk.interpol[name](mod.monomer.r_sol, abs(mod.monomer.z_sol) / mod.monomer.r_sol, grid=False)

            if (name in ["rhog", "rhod", "nd", "chiRT"] or (name[0:3] in ["gas", "ice", "tot"]) or (
                    name[0:5] in ["rhoda", "numda"]) or ("Abun" in name)):
                (self.monomerData[name])[self.modNum] = 10 ** quant
            else:
                (self.monomerData[name])[self.modNum] = quant

        # Adds maxTd and minTd, which track the max and min dust temperature so far.
        self.trackExtremes(mod)

        if self.removeCorrupted != "noIce":
            for ice in self.disk.iceList:
                # Do these flags follow a particle?
                iceList = mod.monomer.ice_sol[ice]

                mod.monomer.corruption[iceList < 0] = 1
                iceList[iceList < 0] = 0  # This might be an issue...

                mod.monomer.corruption[np.isinf(iceList) | np.isnan(iceList)] = 1
                iceList[np.isinf(iceList) | np.isnan(iceList)] = 0  # This probably solves itsself

                (self.monomerData[ice])[self.modNum] = iceList

            if self.removeCorrupted == "selective":
                cond = (self.monomerData["exitIndex"])[self.modNum] <= self.cleaningIndex
                for item in self.monomerData.keys():
                    # print(item, type((self.monomerData[item])[self.modNum]))
                    (self.monomerData[item])[self.modNum] = ((self.monomerData[item])[self.modNum])[cond]

    def assignGridPoint(self, rArr, zArr):
        """
        Discretizes the radial and vertical coordinates.
        """

        T = len(rArr)
        nxArr = np.zeros(T)
        nzArr = np.zeros(T)

        for t in range(T):
            r = rArr[t]
            z = zArr[t]

            nxArr[t] = np.argmin(abs(r - self.disk.model.x[:, 0]))
            nzArr[t] = np.argmin(abs(abs(z / r) - self.disk.model.z[-1, :] / self.disk.model.x[-1, :]))

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

        timesteps[0:-1] = mod.monomer.t_sol[1::] - mod.monomer.t_sol[0:-1]
        timesteps[-1] = 0  # we do not want to include the last timestep of each simulation

        # The second weight comes from differences in the mass pdf from the entities density structure and
        # the pdf used in the sampling.
        xind = int(nx[0])
        zind = int(nz[0])

        proprat = np.ones(T) * self.Pp[xind, zind] / (
                    self.Pn[xind, zind] * self.rGrid[xind, zind] * self.zGrid[xind, zind])

        # Calculate the total weight.
        weights = timesteps * proprat

        return weights

    def calcAvgQuant(self, quant):
        """
        Auxiliary function which calculates the weighted average grid when called on in a plotting script.
        """

        R = self.disk.model.nx
        Z = self.disk.model.nz

        avg_grid = np.zeros((R, Z))

        for r in range(R):
            for z in range(Z):
                print(r, z, end="\r")
                redData = self.monomerData[(self.monomerData["nx"] == r) & (self.monomerData["nz"] == z)]
                if quant == "zm/sa":
                    quantArr = redData["zm"].values / redData["sa"].values
                else:
                    quantArr = redData[quant].values
                weightsArr = redData["weights"].values
                avg_grid[r, z] = np.sum(quantArr * weightsArr) / np.sum(weightsArr)
        return avg_grid

    # Analysis functions related to the refractories.

    def trackExtremes(self, mod):

        T = len(mod.monomer.t_sol)

        for name in self.extremeList:
            # Define the arrays in the dataframe directly (check whether this is efficient)
            (self.monomerData[name])[self.modNum] = np.zeros(T)

        extreme_max_value = ((self.monomerData["Td"])[self.modNum])[0]
        extreme_min_value = ((self.monomerData["Td"])[self.modNum])[0]
        for t in range(0, T):
            current_value = ((self.monomerData["Td"])[self.modNum])[t]
            extreme_max_value = max(extreme_max_value, current_value)
            extreme_min_value = min(extreme_min_value, current_value)

            ((self.monomerData["maxTd"])[self.modNum])[t] = extreme_max_value
            ((self.monomerData["minTd"])[self.modNum])[t] = extreme_min_value

    def determineComposition(self, path, name):

        file = path + name

        data = open(file)
        dummy = data.readline()
        dimens = data.readline()
        dimens = np.array(dimens.split())
        NELEM = int(dimens[0])
        NMOLE = int(dimens[1])
        NDUST = int(dimens[2])
        NPOINT = int(dimens[3])
        header = data.readline()
        data.close()
        dat = np.loadtxt(file, skiprows=3)
        keyword = np.array(header.split())
        NPOINT = len(dat[0:])

        bar = 1.E+6  # 1 bar in dyn/cm2
        Tg = dat[:, 0]  # T [K]
        nHtot = dat[:, 1]  # n<H> [cm-3]
        lognH = np.log10(nHtot)
        press = dat[:, 2]  # p [dyn/cm2]
        Tmin = np.min(Tg)
        Tmax = np.max(Tg)

        iii = np.where((Tg > Tmin) & (Tg < Tmax))[0]

        idx = (np.abs(Tg[iii] - 600)).argmin()
        el = int(np.where(keyword == 'el')[0])

        # Look up all condensates at the end of the condensation sequence.

        with open('GGchem_sol.txt', 'w') as f:
            print('T', Tg, file=f)

        # names = list(keyword[el+1:el+1+NELEM])
        # names = ['CaMgSi2O6', 'Mg2SiO4', 'MgSiO3', 'FeS', 'FeS2', 'SiO2']
        names = ["H", "He", "C", "N", "O", "Na", "Mg", "Si", "Fe", "Al", "Ca", "Ti", "S",
                 "Cl", "K", "Li", "Mn", "Ni", "Cr", "V", "W", "Zr", "F", "P"]
        solids = []
        smean = []
        ymax = -100.0
        iii = np.where((Tg > Tmin) & (Tg < Tmax))[0]

        Ind = []

        compList = []
        molList = []

        for i in range(4 + NELEM + NMOLE, 4 + NELEM + NMOLE + NDUST, 1):
            solid = keyword[i]
            solids.append(solid[1:])
            smean.append(np.mean(dat[iii, i]))
            ind = np.where(keyword == 'n' + solid[1:])[0]
            if (np.size(ind) == 0): continue
            ind = ind[0]
            yy = dat[:, ind]  # log10 nsolid/n<H>
            ymax = np.max([ymax, np.max(yy[iii])])
            ymin = -99
            if ((yy[iii])[-1] > ymin) and (solid[1::] not in self.refractoryExcl):
                compList.append((yy[iii])[-1])
                molList.append(solid[1::])

        compList = np.array(compList)
        compList = 10 ** (np.array(compList))
        compList /= np.sum(compList)

        return molList, compList

    def appendGGCHEMData(self, mod):
        """
        Looks up the ggchem condensation sequence for the given model seed, and subsequently initializes the data into
        self.refractories.
        """

        fileName = "Static_Cond" + str(mod.seedStart) + ".dat"
        molList, compList = self.determineComposition(self.refractoryPath, fileName)
        Ncomp = len(molList)

        if self.initRefr:  # Initialize refractory data structures if this is the first function call.
            self.monomerSeeds = [0] * self.monoNum
            self.monomerSeeds[0] = mod.seedStart
            self.moleculeNames = list(molList)
            self.refractoryData = np.zeros((self.monoNum, len(self.moleculeNames)))
            self.refractoryData[0, :] = compList
            print(molList)
            self.initRefr = False
        else:
            N = self.refractoryData.shape[1]  # molecule number

            self.monomerSeeds[self.modNum] = mod.seedStart

            for n in range(Ncomp):
                if molList[n] in self.moleculeNames:
                    molInd = self.moleculeNames.index(molList[n])
                    self.refractoryData[self.modNum, molInd] = compList[n]
                else:
                    print("Appending molecule:", molList[n])
                    self.moleculeNames.append(molList[n])
                    self.refractoryData = np.append(self.refractoryData, np.zeros((self.monoNum, 1)), axis=1)
                    self.refractoryData[self.modNum, -1] = compList[n]
