import numpy as np
import prodimopy.read as pread
from scipy.interpolate import RectBivariateSpline

class Disk:

    # TODO: Add config for disk and refactor init.
    def __init__(self, species=["H2", "H2O", "CO"], folder="../BackgroundModels/ShampooCodingPaper/vFrag5",
                 modelName="ProDiMo.out", t_index="{:04d}".format(5), order=1, verbose=-1):

        self.verbose = verbose
        self.order = order
        self.diskFolder = folder
        if species != "all":
            self.species = species
            self.all = False
        else:
            self.species = (np.loadtxt(folder + "/AdsorptionEnergies.in", dtype="str", comments="#", usecols=(0),
                                       encoding=None)).tolist()
            self.all = True

        print(self.species)
        print(self.diskFolder)
        try:
            self.model = pread.read_prodimo(self.diskFolder, filename=modelName, td_fileIdx=t_index)
            self.model.dust = pread.read_dust(self.diskFolder)
        except:
            print(
                "Appropriate background entities model timestamp not found, falling back on default entities loading routine...")
            self.model = pread.read_prodimo(self.diskFolder, td_fileIdx=None)
            self.model.dust = pread.read_dust(self.diskFolder)

        self.parameters = self.model.params.mapping  # entities parameters from parameter.out are loaded in this dictionary. Used to assign to background model parameters in SHAMPOO.
        self.prepareInterpol()
        self.doInterpol()

    def prepareInterpol(self):
        # print("Starting interpolation")

        # tg, td, muH, rhog, rhod, ng, nd, pressure, soundspeed, nmol[:,:,self.model.spnames["234"]]
        # spnames: translation table from species to nspec

        # Old version
        # self.rVals = self.model.x.flatten()
        # self.zVals = self.model.z.flatten()/self.rVals
        # self.points = np.stack((self.rVals, self.zVals), axis=1)
        self.rVals = self.model.x[:, 0]
        self.zVals = (self.model.z / self.model.x)[0, :]

        self.data = {}

        # Prepare basic quantities
        self.data["Tg"] = self.model.tg
        self.data["Td"] = self.model.td

        # Note that we actually calculate log10 of these quantities.
        self.data["rhog"] = np.log10(1e3 * self.model.rhog)
        self.data["rhod"] = np.log10(1e3 * self.model.rhod)
        self.data["nd"] = np.log10(1e6 * self.model.nd)

        # self.pressure = LinearNDInterpolator(zipped, self.model.pressure.flatten()) #### CONVERT TO SI
        self.data["soundspeed"] = 1e3 * self.model.soundspeed
        self.data["pressure"] = 1e-1 * self.model.pressure

        self.data["chiRT"] = np.log10(self.model.chiRT)

        # Initialize chemical species abundances
        self.iceList = []
        self.elementDict = {}  # A dictionary containing all ice molecules considered, and the number of each element present.
        # currently we only consider H,C,N,O,S. With the value of elementDict containing a np array with 5 elements representing the number of atoms of each element in a given molecule.

        N = len(self.species)
        n = 0
        if self.verbose > -1:
            print("Received", N, "species. Attempting to load background model number densities...")
        while n < N:
            spec = self.species[n]
            # print(spec)
            if (spec in self.model.spnames.keys()) and (spec + "#" in self.model.spnames.keys()) and (
                    "-" not in spec) and ("+" not in spec):
                indexGas = self.model.spnames[spec]
                self.data["gasAbun" + spec] = np.log10(1e6 * self.model.nmol[:, :, indexGas])
                self.iceList.append(spec)
                self.elementDict[spec] = self.elementCount(spec)

                indexIce = self.model.spnames[spec + "#"]
                self.data["iceAbun" + spec] = np.log10(1e6 * self.model.nmol[:, :, indexIce])
                self.data["totAbun" + spec] = np.log10(
                    1e6 * (self.model.nmol[:, :, indexGas] + self.model.nmol[:, :, indexIce]))
                # print("totAbun"+spec, self.data["totAbun"+spec])
                # print((self.data["gasAbun"+spec])[10,0], (self.data["iceAbun"+spec])[10,0],(self.data["totAbun"+spec])[10,0])
                n += 1
            else:
                print("Omitting species : " + spec)
                (self.species).remove(spec)
                N = len(self.species)

        if self.verbose > -1:
            print("Sucessfully loaded", N, "species.")
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

        while l < L:
            if name[l] in elementlist:
                if l < L - 1:
                    if name[l + 1].isnumeric():
                        elements[elementlist.index(name[l])] += int(name[l + 1])
                    elif name[l + 1] in ["a", "i", "e"]:
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

        print(" " * (longestName + 2), "H", "C", "N", "O", "S")
        for spec in self.species:
            print(spec, " " * (longestName - len(spec)) + ":", (self.elementDict[spec])[0], (self.elementDict[spec])[1],
                  (self.elementDict[spec])[2], (self.elementDict[spec])[3], (self.elementDict[spec])[4])

    def doInterpol(self):
        self.interpol = {}

        ##print("Doing interpolation")
        for name in self.data.keys():
            self.interpol[name] = RectBivariateSpline(self.rVals, self.zVals, self.data[name], kx=self.order,
                                                      ky=self.order)
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
        dataName = np.loadtxt(filepath + "/AdsorptionEnergies.in", dtype="str", comments="#", usecols=(0),
                              encoding=None)
        dataNum = np.loadtxt(filepath + "/AdsorptionEnergies.in", dtype="float", comments="#", usecols=(1, 2),
                             encoding=None)

        uTOkg = 1.660539066e-27
        auTOm = 1.496e11

        N = len(dataName)
        for n in range(N):
            masses["m" + dataName[n]] = dataNum[n, 1] * uTOkg

        rhod = 10 ** (self.interpol["rhod"](rEva, zEva / rEva, grid=False))

        if species == None:

            absice = 0
            N = len(self.iceList)
            for n in range(N):
                diff = masses["m" + self.iceList[n]] * 10 ** (
                    self.interpol[label + "Abun" + self.iceList[n]](rEva, zEva / rEva, grid=False))
                absice += diff
            iceAbun = absice / rhod
        else:
            specice = masses["m" + species] * 10 ** (
                self.interpol[label + "Abun" + species](rEva, zEva / rEva, grid=False))

            iceAbun = specice / rhod

        return iceAbun