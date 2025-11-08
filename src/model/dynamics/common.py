import numpy as np


class CommonDynamicsMethods:

    def ReynoldsT(self, t, r, z, midplane=False):
        """
        Calculates the turbulence Reynolds number as the ratio between the turbulent and molecular viscosity.
        """

        nuT = self.nuTur(t, r, z, midplane=midplane)
        nuM = self.nuMol(t, r, z)

        reynoldsNumber = nuT / nuM

        return reynoldsNumber

    def ReynoldsP(self, t, r, z, midplane=False, size=None):

        if size is None:
            size = self.monomer.homeAggregate.prop["sAgg"]

        num = 4 * size * self.velocity_r(t, r, z, size=size)
        den = self.thermalGasVelocity(t, r, z, species="H2") * self.meanFreePath(t, r, z)

        reynoldsNumber = num / den

        return reynoldsNumber

    def Stokes(self, t, r, z, size=None, rhoMat=2094, midplane=False):

        if size is None:
            size = self.monomer.homeAggregate.prop["sAgg"]
            rhoMat = self.monomer.homeAggregate.prop["rhoAgg"]

        if midplane:
            soundspeed = self.evaluateQuant("soundspeed", r, 0 * r)
            gasDensity = self.evaluateQuant("rhog", r, 0 * r)
        else:
            soundspeed = self.environment["soundspeed"]
            gasDensity = self.environment["rhog"]

        # soundspeed = self.para["HD"]*self.Omega(t,r,z)

        # gasDensity = 10**(self.rhoD(abs(z)/self.auTOm))
        omega = self.Omega(t, r, z)

        mfp = self.meanFreePath(t, r, z)

        cond = size > 9 * mfp / 4

        try:
            if cond:
                if self.verbose > 0:
                    print("Home aggregate is in the Stokes regime")
                num = 4 * rhoMat * size ** 2
                den = 9 * gasDensity * soundspeed * mfp
                stokesNumber = np.sqrt(np.pi / 8) * num / den * omega
            else:
                stokesNumber = np.sqrt(np.pi / 8) * size * rhoMat / (soundspeed * gasDensity) * omega
        except:
            num = 4 * rhoMat * size ** 2
            den = 9 * gasDensity * soundspeed * mfp
            stokesSt = np.sqrt(np.pi / 8) * num / den * omega
            epsteiSt = np.sqrt(np.pi / 8) * size * rhoMat / (soundspeed * gasDensity) * omega
            stokesNumber = np.where(cond, stokesSt, epsteiSt)

        return stokesNumber

    def Schmidt(self, t, r, z):

        schmidtNumber = 1 + self.Stokes(t, r, z) ** 2

        return schmidtNumber

    def meanFreePath(self, t, r, z):
        gasDensity = self.environment["rhog"]

        den = np.sqrt(2) * gasDensity / (self.para["muH"] * self.para["u"]) * self.para["sig_mol"]

        mfp = 1 / den

        return mfp

    def Omega(self, t, r, z):
        d = np.sqrt(r ** 2 + z ** 2)

        omega = np.sqrt(self.para["G"] * self.para["Mstar"] / d ** 3)

        return omega

    def pressureGradient(self, t, r, z):

        if self.constDelP:
            eta = 1e-3
        elif self.testTrueDelP:
            soundspeed = self.environment["soundspeed"]
            gradient = self.environment["drhogdr"]
            gasDensity = self.environment["rhog"]

            factor1 = 1 / (2 * r * gasDensity)
            factor2 = (soundspeed / self.Omega(t, r, z)) ** 2

            eta = -factor1 * factor2 * gradient
        elif self.pisoBenchmark:  # Use calculation from Piso+ 2015
            soundspeed = self.environment["soundspeed"]

            num = soundspeed ** 2
            den = 2 * (r * self.Omega(t, r, z)) ** 2

            eta = num / den

        else:

            soundspeed = self.environment["soundspeed"]
            eta = (soundspeed / (r * self.Omega(t, r, z))) ** 2  # CHECK WHERE THIS COMES FROM (Armitage 2010?)

        return eta

    def thermalGasVelocity(self, t, r, z, species="H2", gasT=None):
        if type(gasT) == type(None):
            gasTemperature = self.environment["Tg"]
        else:
            gasTemperature = gasT

        num = self.para["kB"] * gasTemperature
        den = 2 * np.pi * self.para["m" + species]

        velocity = np.sqrt(num / den)

        return velocity

    def nuTur(self, t, r, z, midplane=False):
        """
        Calculates the turbulent viscosity as nuT = alpha*cs**2/omega
        """
        if midplane:
            soundspeed = self.environment["soundspeed0"]
        else:
            soundspeed = self.environment["soundspeed"]

        # soundspeed = self.environment["soundspeed"]
        omega = self.Omega(t, r, z)

        nuT = self.para["a_settle"] * soundspeed * self.calculateScaleHeight(r, method="mcfost", kind="gas", size=None)

        return nuT

    def nuMol(self, t, r, z):
        """
        Calculates the molecular viscosity using the expression givin in Krijt+ 2018
        """

        vThermal = np.sqrt(8 / np.pi) * self.environment["soundspeed"]

        mfp = self.meanFreePath(t, r, z)

        nuM = vThermal * mfp / 2

        return nuM

    def gasDiffusivity(self, t, r, z):
        """
        Calculated according to Krijt+ 2018 and Ciesla 2010
        """

        # we evaluate the sound speed at the midplane

        soundspeed = self.environment["soundspeed"]  # maybe do this at midplane?

        # stokesNumber = self.Stokes(t,r,z)############################################################################
        # scaleHeight = self.scaleHeight(r, method="pressure")

        # omega = self.Omega(t, r, z)

        # Note that we here implicitly calculate the pressure scale height as cs/omega
        num = self.para["a_settle"] * soundspeed * self.calculateScaleHeight(r, method="mcfost", kind="gas")
        # den = omega#*(1+stokesNumber) this is wrong

        gDiffusivity = num

        return gDiffusivity

    def particleDiffusivity(self, t, r, z):

        gDiffusivity = self.gasDiffusivity(t, r, z)
        stokesNumber = self.Stokes(t, r, z)

        pDiffusivity = gDiffusivity / (1 + stokesNumber ** 2)

        return pDiffusivity

    def velocity_r(self, t, r, z, size=None, stokes=None):

        if stokes == None:
            stokesNumber = self.Stokes(t, r, z, size)
        else:
            stokesNumber = stokes
        v_r = -2 * self.pressureGradient(t, r, z) * r * self.Omega(t, r, z) * stokesNumber / (1 + stokesNumber ** 2)

        return v_r

    def velocity_z(self, t, r, z, size=None, stokes=None):

        if stokes == None:
            stokesNumber = self.Stokes(t, r, z, size)
        else:
            stokesNumber = stokes
        v_z = -self.Omega(t, r, z) * z * stokesNumber
        # print(self.Omega(t,r,z), z/self.auTOm, self.Stokes(t,r,z, size))
        return v_z

    def velocity_eff_r(self, t, r, z):

        v_r = self.velocity_r(t, r, z)
        term1 = self.environment["dDdr"]

        gasDensity = self.environment["rhog"]

        term2a = self.particleDiffusivity(t, r, z) / gasDensity

        term2b = self.environment["drhogdr"]

        v_eff_r = v_r + term1 + term2a * term2b
        # if t==0:
        #	print("vr", v_r, "v_eff_r", v_eff_r)

        return v_eff_r

    def velocity_eff_z(self, t, r, z):

        v_z = self.velocity_z(t, r, z)
        term1 = self.environment["dDdz"]  # Krijt 2018: term1 = 0

        gasDensity = self.environment["rhog"]

        term2a = self.particleDiffusivity(t, r, z) / gasDensity
        # And again...
        term2b = self.environment["drhogdz"]

        v_eff_z = v_z + term1 + term2a * term2b
        # if t==0:
        # print("vz", v_z, "v_eff_z", v_eff_z, term1, term2a, term2b)

        return v_eff_z