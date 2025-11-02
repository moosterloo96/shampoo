import numpy as np


class CalculateDiffusivityGrid:

    def calculateDiffusivity(self, rVals, zVals, size=None):
        """
        Auxiliary function for calculating a grid of diffusivities outside the "environment"-framework.
        """

        # Evaluate the soundspeed--------
        soundspeed = self.disk.data["soundspeed"]
        soundspeed0 = np.zeros(soundspeed.shape)
        for i in range(0, len(soundspeed[0, :])):
            soundspeed0[:, i] = soundspeed[:, 0]
        # And gas density
        gasDensity = self.disk.data["rhog"]

        # Calculate omega--------
        d = np.sqrt(rVals ** 2 + zVals ** 2)
        omega = np.sqrt(self.para["G"] * self.para["Mstar"] / d ** 3)

        # Calculate the mean free path
        den = np.sqrt(2) * gasDensity / (self.para["muH"] * self.para["u"]) * self.para["sig_mol"]

        mfp = 1 / den

        # Calculate the stokesNumber--------

        # gasDensity = 10**(self.rhoD(abs(z)/self.auTOm))
        if size == None:
            size = self.monomer.homeAggregate.prop["sAgg"]
            density = self.monomer.homeAggregate.prop["rhoAgg"]
        else:
            density = 2094

        cond = size > 9 * mfp / 4

        num = 4 * density * size ** 2
        den = 9 * gasDensity * soundspeed * mfp
        stokesSt = np.sqrt(np.pi / 8) * num / den * omega
        epsteiSt = np.sqrt(np.pi / 8) * size * density / (soundspeed * gasDensity) * omega
        stokesNumber = np.where(cond, stokesSt, epsteiSt)
        # Note that we here implicitly calculate the pressure scale height as cs/omega
        # NOTE THAT THIS IS WRONG
        # num = self.para["a_settle"]*soundspeed0**2
        # den = omega

        gDiffusivity = self.para["a_settle"] * soundspeed * self.calculateScaleHeight(rVals, method="mcfost",
                                                                                      kind="gas", size=None)

        pDiffusivity = gDiffusivity / (1 + stokesNumber ** 2)

        self.diffusivity = pDiffusivity

        return pDiffusivity
