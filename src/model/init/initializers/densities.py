import numpy as np
from scipy.interpolate import RectBivariateSpline

class InitializeDensities:

    def initDensities(self, prodimo=False):
        """
        Prepares the spatially interpolated number and mass densities for all mass bins considered in the model.
        """

        r = self.disk.model.x
        z = self.disk.model.z / self.disk.model.x

        M = len(self.grainSizes)
        R = self.disk.model.nx
        Z = self.disk.model.nz

        for m in range(M):
            if self.verbose > -99:
                print("Calculating size " + str(m + 1) + "/" + str(M), end="\r")
            size = self.grainSizes[m]  # convert from micron to SI

            self.probeEnvironment(r, z * r, 0, inside_loop=False)
            Hp = self.calculateScaleHeight(r * self.auTOm, method="mcfost", kind="dust", size=size) / (r * self.auTOm)
            expArg = -(z ** 2) / (2 * Hp ** 2)

            sigmaa = self.disk.model.dust.sigmaa[m, :] * 1e4 / 1e3  # convert from g/cm2 to kg/m2
            sig2D = np.zeros((R, Z))
            for zNo in range(Z):
                sig2D[:, zNo] = sigmaa

            N = 2 * sig2D / (
                        np.sqrt(2 * np.pi) * Hp * r * self.auTOm)  # We multiply as sigmaa is only 1/2 of the total surface
            # density
            mda = N * np.exp(expArg)

            ma = self.para["rho_mat"] * 4 / 3 * np.pi * (size) ** 3
            nda = mda / ma

            rhoName = "rhoda" + str(m)
            numName = "numda" + str(m)

            # Make sure we add a floor
            minval = 1e-20
            nda[nda < minval] = minval
            mda[mda < minval] = minval * 1e-30

            self.disk.data[rhoName] = np.log10(mda)
            self.disk.data[numName] = np.log10(nda)
            self.disk.interpol[rhoName] = RectBivariateSpline(self.disk.rVals, self.disk.zVals, self.disk.data[rhoName],
                                                              kx=self.disk.order, ky=self.disk.order)
            self.disk.interpol[numName] = RectBivariateSpline(self.disk.rVals, self.disk.zVals, self.disk.data[numName],
                                                              kx=self.disk.order, ky=self.disk.order)
            self.environment = None