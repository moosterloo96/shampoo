import numpy as np

from src.model.entities.aggregate import HomeAggregate

class Monomer:

    def __init__(self, model, r0, z0, home_aggregate=None, size=0.05e-6):
        """
            Initializes a monomer. For convenience, r0 and z0 are in AU.
        """

        if home_aggregate == None:
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
        self.prop["sMon"] = size  # in m
        try:
            self.prop["zCrit"] = model.paraDict["zCrit"] * size
        except:
            self.prop[
                "zCrit"] = 2 * size  # critical depth inside a home aggregate where the monomer is no longer exposed
        if model.pisoBenchmark:
            self.prop["rho_mat"] = 2000
        else:
            self.prop["rho_mat"] = float(
                model.paraDict["rhoMon"])  # monomer material density in kg/m3. Informed by ProDiMo
        self.prop["mMon"] = 4 / 3 * np.pi * self.prop["rho_mat"] * self.prop["sMon"] ** 3
        self.prop["yield"] = float(model.paraDict[
                                       "yield"])  # Yield rate for UV photodesorption. Experimental values for all major molecules are within 1 order of magnitude of this value (Oberg 2009).
        self.prop["Nads"] = float(
            model.paraDict["Nads"])  # Number of adsorption sites per square meter (Piso+ 2015; Hollenbach+ 2009)
        self.prop["Nact"] = int(
            model.paraDict["Nact"])  # Number based on the range of values 2-4 stated in Cuppen+ 2017.
