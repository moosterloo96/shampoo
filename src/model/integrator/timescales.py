import numpy as np


class Timescales:

    def tauMz(self, t, r, z, Hp=None, size=None):
        "Returns the vertical settling timescale in seconds. Traversed distance is one particle scale height"

        vz = abs(self.velocity_z(t, r, z, size=size))

        if Hp == None:
            Hp = self.calculateScaleHeight(r, method="mcfost", kind="gas", size=size)

        tauMz = Hp / vz

        return tauMz


    def tauMr(self, t, r, z, Hp=None, size=None):
        "Returns the radial drift timescale in seconds. Traversed distance is one particle scale height, unless PisoBenchmark"
    
        vr = abs(self.velocity_r(t, r, z, size=size))
        if Hp == None:
            Hp = self.calculateScaleHeight(r, method="mcfost", kind="gas", size=size)

        if self.pisoBenchmark:
            Hp = r

        tauMr = Hp / vr

        return tauMr


    def tauTub(self, t, r, z, Hg=None):
        "Returns the turbulent stirring timescale in seconds."

        Hg = self.calculateScaleHeight(r, method="mcfost", kind="gas")
        viscosity = self.nuTur(t, r, z, midplane=True)

        tauTub = Hg ** 2 / viscosity

        return tauTub


    def tauCol(self, t, r, z, physical=False):
        """
        Returns the shortest collision timescale in seconds.
        """

        if physical:
            tauCol = 1 / np.sum(self.colRates)
        else:
            tauCol = 1 / np.sum(self.effColRates)
        # print(self.effColRates)
        # print(np.sum(self.effColRates))

        # print(tauCol/self.sTOyr)
        # if minimum:
        #    tauCol = np.min(tauCol)

        return tauCol


    def tauAds(self, t, r, z, size, species):
        """
        Returns the adsorption timescale in seconds.
        """

        tauAds = (self.monomer.ice_sol[species]) / (4 * np.pi * (size ** 2) * self.rateAdsorption(t, r, z, species))

        return tauAds[0]


    def tauDes(self, t, r, z, size, species):
        """
        Returns the total desorption timescale in seconds.
        """

        rateDes = self.rateDesorptionThermal(t, r, z, species, fx=1) + self.rateDesorptionPhoto(t, r, z, species, fx=1)
        tauDes = (self.monomer.ice_sol[species]) / (4 * np.pi * (size ** 2) * rateDes)

        return tauDes[0]


    def tauDiff(self, t, r, z, size, species):
        """
        Calculates the diffusion timescale in a given aggregate for rms-displacement "size" for species "species"
        at position (r,z). All units in SI.
        """

        mDiffusivity = self.moleculeDiffusivity(t, r, z, species)

        tauDiff = (size ** 2) / (6 * mDiffusivity)

        return tauDiff


    def tauTds(self, t, r, z, size=None):
        """
        Calculates the thermal desorption timescale in accordance with Piso+ 2015 in seconds.

        !!! Do not use outside of PisoBenchmark !!!
        """

        if size == None:
            size = self.monomer.prop["sMon"]

        spec = self.rateDesorptionThermal(t, r, z, "H2O", dustT=None, init=False)

        tauTds = size * self.monomer.prop["rho_mat"] / (3 * spec)

        return tauTds


    def determineDeltaT(self, t_in, r_in, z_in):
        r, z, t = self.unpackVars(r_in, z_in, t_in)

        tauList = [1e6 * self.sTOyr]

        if self.collisions:  # assumes that the effective collision rates are already calculated.
            tauCol = self.tauCol(t, r, z)
            tauList.append(tauCol)

        if self.migration:
            tauMz = self.tauMz(t, r, z)
            tauMr = self.tauMr(t, r, z)
            tauList.append(self.fdyn * tauMz)
            tauList.append(self.fdyn * tauMr)
            if (t_in == 0) and (self.verbose > 0):
                print("Migration timescale for a grain of size {:.2f} m is {:.2e} yr".format(self.monomer.prop["sMon"],
                                                                                             tauMr / self.sTOyr))

            if self.diffusion:
                tauTub = self.tauTub(t, r, z)
                tauList.append(self.ftub * tauTub)

        # print("")
        # print(np.array(tauList)/self.sTOyr)
        # Note that the timescale is chosen to be shorter if this is required by the ice formation algorithm.
        if (self.collisions or self.migration):
            tauMin = min(tauList)
            deltaT = max([self.delta_t_floor * self.sTOyr, tauMin])
        else:
            # If do not do collisions or migration we set the global timestep to 1/10th the local orbital period.
            deltaT = 0.1 * 2 * np.pi / (self.Omega(t, r, z))

        return deltaT