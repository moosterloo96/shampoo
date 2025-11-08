from time import process_time
from src.model.entities.disk import Disk

class MainInitializer:

    # def initialize(self, disk=None, species=["H2O", "CO", "CO2", "CH4", "NH3", "H2S"], disk_folder="../ShampooBackground",
    #              parameter_folder=None, verbose=None, migration=None, diffusion=None, collisions=None, trackice=None,
    #              debug=False, save_csv=[False, None], phi=None, breakIce=False, legacyIce=False, colEq=False,
    #              storage_style=0, readsorption=False, supverbose=False, qTest=False, qTestAds=False, qTestMig=False,
    #              activeML=True):
    #     """
    #     entities: Instance of Disk which contains the output from ProDiMo. If None a new instance is created.
    #     monomer: Instance of Monomer. If None a new instance is created.
    #     species: The molecules which are considered in calculating the composition.
    #     disk_folder: The folder which contains the ProDiMo output of this model.
    #
    #     verbose:
    #         1 - only warings
    #         2 - warnings and initialization/integration notifications
    #
    #     migration: switch for radial and vertical migration of the monomers.
    #     diffusion: if migration is true, this switch determines whether we calculate pure aerodynamic drag or also turbulent diffusion.
    #     collisions: switch for the coagulation module
    #     trackice: switch for the condensation/evaportation of ices
    #     debug: debug mode; currently unused
    #     save_csv: if save_csv[0]==True we regularize the output and write it to a .csv. Data is taken at at the times (in kyr) given in array-like save_csv[1].
    #     """
    def initialize(self, disk, config):

        # TODO: Add value validation
        # TODO: Do cleanup and sorting of stuff between <name>shampooInput.csv, and config.json
        # TODO: Rename .csv
        # TODO: Add functionality to pickle and unpickle a disk model such that we don't have to re-load and re-interpolate.
        # TODO: Refactor into separate methods.
        # TODO: DOC.

        if config["supverbose"]:
            self.supverbose = True
        else:
            self.supverbose = False

        if not self.supverbose:
            self.print_titlecard()

            print("Attempting to load parameters from folder: ", config["parameter_folder"])
        self.paraDict = self.loadPara(parameter_folder=config["parameter_folder"])

        if not self.supverbose:
            print("Attempt succesful!")

        if self.supverbose:
            self.verbose = -1000
        else:
            if config["verbose"] == None:
                self.verbose = int(self.paraDict["verbose"])
            else:
                self.verbose = config["verbose"]

        ### Debugging levers
        self.breakIce = config["breakIce"]
        self.legacyIce = config["legacyIce"]
        self.readsorption = config["readsorption"]
        self.qTest = config["qTest"]
        self.qTestAds = config["qTestAds"]
        self.qTestMig = config["qTestMig"]

        ### Setting the parameters for the ice integration
        self.integrator = ("LSODA", "LSODA")  # 0 is exposed, 1 is unexposed
        self.atol = (1e-6, 1e-6)  # These are the default values in the integrators.
        self.rtol = (1e-3, 1e-3)  # Default values. Used much lower values in planetbox model.
        self.floorval = 1e-22  # the floor value below which we set the derivatives 0. Default is 1e-22
        self.iceScaleFact = 1e5  # offset factor for numerical scaling in units of mMon.

        # Warning flags. Makes sure we print warnings only once.
        self.printIntWarning = False

        ### Functionality levers!
        # For dynamics...
        if config["migration"]  == None:
            self.migration = bool(self.paraDict["migration"])  # DYNAMICS
        else:
            self.migration = config["migration"]

        # ...collisions...
        if config["collisions"] == None:
            self.collisions = bool(self.paraDict["collisions"])  # KINETICS
        else:
            self.collisions = config["collisions"]

        # and ice evolution!
        if config["trackice"] == None:
            self.trackice = bool(self.paraDict["trackice"])
        else:
            self.trackice = config["trackice"]

        try:
            self.colEq = bool(self.paraDict["col_eq"])
        except:
            self.colEq = config["colEq"]
        try:
            self.activeML = bool(self.paraDict["activeML"])
        except:
            self.activeML = config["activeML"]

        self.printStatistics = bool(self.paraDict["print_stat"])
        if self.verbose > -1:
            tic = process_time()
            print("Initializing model")

        self.diskFolder = config["disk_folder"]
        self.debug = config["debug"]
        self.save_csv = config["save_csv"][0]
        self.eval_csv = config["save_csv"][1]

        self.pisoBenchmark = bool(self.paraDict["piso"])
        self.cieslaBenchmark = bool(self.paraDict["ciesla"])

        if disk == None:
            self.disk = Disk(species=config["species"], folder=self.diskFolder)
        else:
            self.disk = disk
            self.diskFolder = disk.diskFolder

        if self.trackice:
            self.iceNum = len(self.disk.iceList)
        else:
            self.iceNum = 0

        ### Storage management
        try:
            self.store = int(self.paraDict["storage_style"])
        except:
            self.store = config["storage_style"]  # Do we want to keep track of the local environment of the monomer/home aggregate?

        if self.store == 0:
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
            self.testTrueDelP = True  # Use simplification on the calculation of pressure gradient if false

        self.constTimestep = [bool(self.paraDict["const_delt"]), float(self.paraDict["const_delt_val"])]

        self.deterministic = bool(self.paraDict["deterministic"])

        self.fixR = bool(self.paraDict["fixR"])
        self.fixZ = bool(self.paraDict["fixZ"])

        if not self.migration:
            self.fixR = True
            self.fixZ = True

        self.innerRdraw = float(self.paraDict[
                                    "rInnerDraw"])  # inner radial sampling distance in AU if monomer initial positions are randomly picked.
        try:
            self.outerRdraw = float(self.paraDict[
                                        "rOuterDraw"])  # outer radial sampling distance in AU if monomer initial positions are randomly picked.
        except:
            if self.verbose > 1:
                print("No outer drawing distance found, using tapering radius instead...")
            self.outerRdraw = self.para["Rtaper"] / self.auTOm

        self.xi = 1 / 3  # numerical parameter related to the random turbulent kicks

        self.feps = float(self.paraDict["rat_group"])  # mass ratio where we start to group collision partners
        self.mrat = float(self.paraDict["rat_erode"])  # mass ratio of colliders where erosion and fragmentation transition

        self.ftub = float(self.paraDict["ftub"])  # fraction of the drift/turbulent timescale
        self.fcol = float(self.paraDict["fcol"])  # fraction of the collision timescale
        self.fdyn = float(self.paraDict["fdyn"])
        self.fice = float(self.paraDict["fice"])

        self.phi = config["phi"]  # Auxiliary parameter which allows for easy changing of the porosity.

        self.diffType = "mcfost"
        self.desType = "new"  # Use old (Piso+ 2015) or new (Cuppen+ 2017) desorption formalism.
        self.flag = False  # The warning flag if our mass change in ice becomes too large.
        self.delta_t_floor = float(self.paraDict["delta_t_floor"])  # Floor to variable timestep in yr.

        # print(vars(self))

        if self.verbose > -1:
            toc = process_time()
            print("Finished model initialization in {:.2f} CPU s".format(toc - tic))
            if self.verbose > 0:
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