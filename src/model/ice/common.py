import numpy as np


class CommonIceEvolutionMethods:

    def stickingFactor(self, t, r, z, iceName, dustT=None):
        if dustT == None:
            dustT = self.environment["Td"]

        tanhArg = self.para["betaStick"] * (dustT - self.para["gammaStick"] * self.para["Eads" + iceName] / self.para["kB"])
        sticking = self.para["alphaStick"] * (1 - np.tanh(tanhArg))

        return sticking


    def rateAdsorption(self, t, r, z, iceName, init=False, dustT=None, gasT=None, iceList=None):
        """
        Calculates the surface-specific adsorption rate. (kg/m2/s)
        """

        if self.pisoBenchmark:
            rateAds = 0
        else:

            nSpecies = self.environment["gasAbun" + iceName]

            # if iceName =="NH3":
            #    nSpecies /= 1e2

            if dustT == None:
                dustT = self.environment["Td"]
            vthermal = self.thermalGasVelocity(t, r, z, species=iceName, gasT=gasT)

            sticking = self.stickingFactor(t, r, z, iceName, dustT=dustT)

            rateAds = nSpecies * vthermal * sticking * self.para["m" + iceName]

        return rateAds


    def iceFraction(self, t, r, z, iceName, iceList=None):
        """
        Auxiliary function for the desorption rates to calculate the number fraction of ice species iceName in the ice mantle
        (# of iceName molecules per total amount of ice molecules).
        """
        # print(iceList)
        if self.legacyIce:
            if (self.monomer.iceTot_sol)[-1] > 0:
                if (self.monomer.ice_sol[iceName])[-1] > 0:
                    iceTot = 0
                    for n in range(self.iceNum):
                        iceTot += (self.monomer.ice_sol[self.disk.iceList[n]])[-1] / self.para["m" + self.disk.iceList[n]]
                    fx = ((self.monomer.ice_sol[iceName])[-1] / self.para["m" + iceName]) / iceTot
                else:
                    fx = 1
            else:
                fx = 1

        else:
            iceTot = 0  # count the total number of ice molecules
            for n in range(self.iceNum):
                iceTot += iceList[self.disk.iceList[n]] / self.para["m" + self.disk.iceList[n]]
            fx = iceList[iceName] / self.para["m" + iceName] / iceTot

        return fx


    def iceSpots(self, t, r, z, iceList=None):
        """
        Auxiliary function which calculates the total number of ice molecules present on the monomer.
        """

        nice = 0
        if self.legacyIce:
            for n in range(self.iceNum):
                nice += (self.monomer.ice_sol[self.disk.iceList[n]])[-1] / self.para["m" + self.disk.iceList[n]]
        else:
            for n in range(self.iceNum):
                nice += iceList[self.disk.iceList[n]] / self.para["m" + self.disk.iceList[n]]

        return nice


    def vibrationFrequency(self, t, r, z, iceName):
        nu = None

        if self.oldNu:
            fact1 = 1.6e11 * np.sqrt(
                self.para["Eads" + iceName] * self.para["mp"] / (self.para["kB"] * self.para["m" + iceName]))
            fact2 = (self.monomer.prop["Nads"] / self.para["NadsRef"]) ** (0.5)
            nu = fact1 * fact2
        else:
            # The default expression we always want to use.
            num = 2 * self.monomer.prop["Nads"] * self.para["Eads" + iceName]
            den = self.para["m" + iceName] * np.pi ** 2
            nu = np.sqrt(num / den)

        return nu


    def rateDesorptionThermal(self, t, r, z, iceName, dustT=None, init=False, fx=None, iceList=None):
        """
        Calculates the surface-specific thermal desorption rate. (kg/m2/s)
        """

        if dustT == None:
            dustTemperature = self.environment["Td"]
        else:
            dustTemperature = dustT

        if self.pisoBenchmark:
            fact1 = 1.6e11 * np.sqrt(
                self.para["Eads" + iceName] * self.para["mp"] / (self.para["kB"] * self.para["m" + iceName]))
            fact2 = self.monomer.prop["Nads"] * self.para["m" + iceName]

            rateDesT = fact1 * fact2 * np.exp(-self.para["Eads" + iceName] / (self.para["kB"] * dustTemperature))
        elif self.activeML:

            nu = self.vibrationFrequency(t, r, z, iceName)

            kdes = nu * np.exp(-self.para["Eads" + iceName] / (self.para["kB"] * dustTemperature))

            # We calculate the total number of active spots for the whole monomer
            nact = self.monomer.prop["Nact"] * self.monomer.prop["Nads"] * 4 * np.pi * (self.monomer.prop["sMon"]) ** 2

            # And compare this to the number of occupied spots on the monomer
            nice = self.iceSpots(t, r, z, iceList=iceList)

            if nice < nact:
                # In this case there are more active spots than ice molecules on the monomer, and thus desorption
                # is limited by the amount of molecules.
                if self.legacyIce:
                    iceAmount = (self.monomer.ice_sol[iceName])[-1]
                else:
                    iceAmount = iceList[iceName]

                # Divide by surface area of monomer to calculate # of spots per m2
                # corfactNum = self.environment["rhod"]
                corfactDen = self.para["m" + iceName] * 4 * np.pi * (self.monomer.prop["sMon"]) ** 2

                nspots = iceAmount / corfactDen

            else:
                # In this case desorption is limited by the number of active spots.
                if fx == None:
                    fracIce = self.iceFraction(t, r, z, iceName, iceList=iceList)
                else:
                    fracIce = fx

                nspots = self.monomer.prop["Nact"] * self.monomer.prop["Nads"] * fracIce

            rateDesT = kdes * nspots * self.para["m" + iceName]

        else:

            nu = self.vibrationFrequency(t, r, z, iceName)

            kdes = nu * np.exp(-self.para["Eads" + iceName] / (self.para["kB"] * dustTemperature))

            if fx == None:
                fracIce = self.iceFraction(t, r, z, iceName, iceList=iceList)
            else:
                fracIce = fx

            nspots = self.monomer.prop["Nact"] * self.monomer.prop["Nads"] * fracIce

            rateDesT = kdes * nspots * self.para["m" + iceName]

        return rateDesT


    def aggregateMFP(self, t, r, z):
        mfp = 4 * self.monomer.prop["sMon"] / (3 * self.monomer.homeAggregate.prop["phi"])

        return mfp


    def moleculeDiffusivity(self, t, r, z, iceName):
        # Call on necessary environment properties the diffusivity
        dustTemperature = self.environment["Td"]

        # And intermediate quantities
        mfp = self.aggregateMFP(t, r, z)
        sticking = self.stickingFactor(t, r, z, iceName, dustT=dustTemperature)
        nu0 = self.vibrationFrequency(t, r, z, iceName)

        numTot = nu0 * mfp ** 2

        expArg = self.para["Eads" + iceName] / (self.para["kB"] * dustTemperature)
        term1 = (mfp / self.para["ds"]) * np.exp(self.para["Ediff/Eads"] * expArg)
        term2 = sticking * np.exp(expArg)

        denTot = term1 + term2

        diffusivity = numTot / denTot

        return diffusivity


    def calcQFactor(self, t, r, z, tau, iceName=None):
        """
        Calculates the q factor in the intermediate tau_diff regime.
        """

        qFactor = (np.log10(tau) - np.log10(self.para["tauMinInt"])) / (
                    np.log10(self.para["tauMaxInt"]) - np.log10(self.para["tauMinInt"]))

        # if self.environment["iceAbun"+iceName]>0.5*self.environment["gasAbun"+iceName]:
        #    qFactor = 1
        # else:
        #    qFactor = 0

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
        tau = self.tauDiff(t, r, z, size, iceName)  # note that this is the aggregate diffusion timescale

        mfp = self.aggregateMFP(t, r, z)

        self.para["tauMinInt"] = self.para["tauMin"] * (size / mfp) ** 2
        self.para["tauMaxInt"] = self.para["tauMax"] * (size / mfp) ** 2  # convert from mfp to agg timescale
        # print(tau, self.para["tauMinInt"],self.para["tauMaxInt"])
        if tau > self.para["tauMaxInt"]:
            # In this case, diffusion goes very slow.
            # print("q =",0)
            qFactor = 1
            rateDesI = 0
        elif tau < self.para["tauMinInt"]:
            # In this case, diffusion goes very fast.
            # print("q =",1)
            qFactor = 0
            rateDesI = self.rateDesorptionThermal(t, r, z, iceName, dustT=None, iceList=iceList)
        else:
            # Otherwise we're in the transitional regime where 0<q<1.
            qFactor = self.calcQFactor(t, r, z, tau, iceName)
            # print("q =",qFactor)
            rateDesI = (1 - qFactor) * self.rateDesorptionThermal(t, r, z, iceName, dustT=None, iceList=iceList)

        return rateDesI, qFactor


    def rateDesorptionPhoto(self, t, r, z, iceName, init=False, fx=None, iceList=None):
        """
        Calculates the surface-specific UV photon desorption rate. (kg/m2/s)
        """

        if self.pisoBenchmark:
            rateDesP = 0
        else:
            chiRT = self.environment["chiRT"]
            if fx == None:
                fracIce = self.iceFraction(t, r, z, iceName, iceList=iceList)
            else:
                fracIce = fx

            # 1st order vs 2nd order desorption regime. Volume vs surface. See Woitke+ 2009; active layer concept.
            # yield: Leiden group has done better yield-measurements for H2O. (Start at Öberg+ 2009; Arasa+ 2010).
            # Semenov & Kamp: yield may not be that important.
            rateDesP = fracIce * self.monomer.prop["yield"] * self.para["FDraine"] * chiRT * self.para["m" + iceName]

        return rateDesP


    def calcIceTot(self):
        iceTot = 0
        for n in range(self.iceNum):
            iceTot += (self.monomer.ice_sol[self.disk.iceList[n]])[-1]

        return iceTot