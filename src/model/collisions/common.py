import numpy as np


class CommonCollisionsMethods:

    def vRelBrown(self, t, r, z):
        """
        Calculates difference in Brownian motion.
        """

        mLarge = self.para["mLarge"]
        mSmall = self.para["mSmall"]

        num = 8 * self.para["kB"] * self.environment["Td"] * (mLarge + mSmall)
        den = np.pi * mLarge * mSmall

        vBM = np.sqrt(num / den)

        return vBM


    def vRelDrift(self, t, r, z):
        """
        Calculates differential drift relative velocity.
        """

        v1 = self.velocity_r(t, r, z, size=self.para["sLarge"], stokes=self.vrel_Stlarge)
        v2 = self.velocity_r(t, r, z, size=self.para["sSmall"], stokes=self.vrel_Stsmall)

        vRD = abs(v1 - v2)

        return vRD, v1, v2


    def vRelSettling(self, t, r, z):
        """
        Calculates differential settling relative velocity.
        """

        v1 = self.velocity_z(t, r, z, size=self.para["sLarge"], stokes=self.vrel_Stlarge)
        v2 = self.velocity_z(t, r, z, size=self.para["sSmall"], stokes=self.vrel_Stsmall)

        vZD = abs(v1 - v2)

        return vZD


    def vRelAzimuthal(self, t, r, z, v1r, v2r):
        """
        Calculates the relative velocity contribution due to differences in azimuthal velocity.
        """

        # v1r = self.velocity_r(t, r, z, size=self.para["sLarge"], stokes=self.vrel_Stlarge)
        # v2r = self.velocity_r(t, r, z, size=self.para["sSmall"], stokes=self.vrel_Stsmall)

        v1 = v1r / (2 * self.vrel_Stlarge)
        v2 = v2r / (2 * self.vrel_Stsmall)

        vAD = abs(v1 - v2)

        return vAD


    def vRelTurbulent(self, t, r, z):
        """
        Calculates the relative velocity difference due to turbulence using the approximations presented
        by Ormel & Cuzzi 2007
        """

        # pRe = model.ReynoldsP(t, r, z, size=sagg)

        # if pRe>=1:
        #    vTM = 0
        # else:
        omega = self.Omega(t, r, z)
        Re = self.ReynoldsT(t, r, z, midplane=True)

        tstop = self.vrel_Stlarge / omega
        omg = 1 / omega  # ==equal to tL when following Ormel & Cuzzi 2007.
        teta = omg / np.sqrt(Re)  # tL = 1/omega

        if tstop <= teta:
            Q = (Re) ** 0.25 * abs(self.vrel_Stlarge - self.vrel_Stsmall)
        elif tstop >= omg:
            Q = np.sqrt(1 / (1 + self.vrel_Stlarge) + 1 / (1 + self.vrel_Stsmall))
        else:
            Q = 1.55 * np.sqrt(self.vrel_Stlarge)

        vTM = np.sqrt(self.para["a_settle"]) * self.environment["soundspeed0"] * Q

        return vTM


    def calcTotVRel(self, t, r, z, doPrint=False):
        """
        Calculate relative velocities for a home aggregate of a certain size s1 with a collision partner of size s2.
        """

        sagg = self.monomer.homeAggregate.prop["sAgg"]
        scol = self.monomer.homeAggregate.prop["sCol"]
        # magg = self.monomer.homeAggregate.prop["mAgg"]
        # mcol = self.monomer.homeAggregate.prop["mCol"]

        if scol > sagg:
            denSmall = 2094  # self.monomer.homeAggregate.prop["rhoAgg"]
            denLarge = 2094
            self.para["sLarge"] = scol
            self.para["sSmall"] = sagg
            # self.para["mLarge"] = #mcol
            # self.para["mSmall"] = #magg

        else:
            denSmall = 2094
            denLarge = 2094  # self.monomer.homeAggregate.prop["rhoAgg"]
            self.para["sLarge"] = sagg
            self.para["sSmall"] = scol
            # self.para["mLarge"] = #magg
            # self.para["mSmall"] = #mcol

        self.para["mSmall"] = 4 / 3 * np.pi * denSmall * self.para["sSmall"] ** 3  # magg
        self.para["mLarge"] = 4 / 3 * np.pi * denLarge * self.para["sLarge"] ** 3  # mcol

        # self.probeEnvironment(r/self.auTOm, z/self.auTOm, t/(1e3*self.sTOyr), inside_loop=False)
        if doPrint:
            print("Properties passed:")
            print("sagg: ", sagg, "scol:", scol)

        self.vrel_Stlarge = self.Stokes(t, r, z, size=self.para["sLarge"], rhoMat=denLarge)
        self.vrel_Stsmall = self.Stokes(t, r, z, size=self.para["sSmall"], rhoMat=denSmall)

        v1 = self.vRelBrown(t, r, z)
        v2, vr1, vr2 = self.vRelDrift(t, r, z)
        v3 = self.vRelSettling(t, r, z)
        v4 = self.vRelAzimuthal(t, r, z, vr1, vr2)
        v5 = self.vRelTurbulent(t, r, z)

        v_tot = np.sqrt(v1 ** 2 + v2 ** 2 + v3 ** 2 + v4 ** 2 + v5 ** 2)

        if doPrint:
            print("relative velocity: ", v_tot, "m/s")

        if (self.debug) and (self.monomer.homeAggregate.prop["sAgg"] > self.grainSizes[-1]):
            print("Relative velocity report:")
            print("sagg: ", sagg, "scol:", scol)
            print("Brownian:", v1)
            print("Radial:", v2)
            print("Settling:", v3)
            print("Azimuthal:", v4)
            print("Turbulence:", v5)
            print("Total: ", v_tot)
            print("Largest:", self.para["sLarge"], "Smallest:", self.para["sSmall"])
            print("t=", t, "r=", r, "z=", z)

        return v_tot

        # Collision rates


    def sigCol(self, t, r, z, s=None):
        """
        Calculates the collisional cross-section between the home aggregate and a particle of size s.
        """

        if s is None:
            s = self.monomer.homeAggregate.prop["sCol"]

        sig_col = np.pi * (self.monomer.homeAggregate.prop["sAgg"] + s) ** 2

        return sig_col


    def collisionRate(self, t, r, z, s=None):
        if s is None:
            s = self.monomer.homeAggregate.prop["sCol"]

        name = "numda" + str(np.argmin(abs(self.grainSizes - s)))
        # print(r,z)

        # tocn = process_time()
        nDen = self.evaluateQuant(name, r, z)
        # ticn = process_time()
        # print("nDen: {:.2e} s".format(ticn-tocn))

        # tocv = process_time()
        vRel = self.calcTotVRel(t, r, z)
        # ticv = process_time()
        # print("vrel: {:.2e} s".format(ticv-tocv))

        sig = self.sigCol(t, r, z)

        if self.verbose > 3:
            print("nDen = {:.2e} /m3".format(nDen), "vRel = {:.2e} m/s".format(vRel), "sig = {:.2e} m2".format(sig))

        colrate = nDen * vRel * sig
        # print("Ccol:",colrate,"   nDen:",nDen,"   vRel:",vRel,"   sig",sig )
        return colrate


    def calcColRates(self, t_now, r_now, z_now, size=None):
        """
        Calculates the collision rates for the home aggregate at its given position. Is a key function to be evaluated
        every time step.
        """

        r, z, t = self.unpackVars(r_now, z_now, t_now)
        sizes = self.grainSizes

        if size == None:
            sAgg = self.monomer.homeAggregate.prop["sAgg"]
        else:
            sAgg = size
            # self.monomer.homeAggregate.prop["sAgg"] = size
            # self.monomer.homeAggregate.prop["sMon"] = size
        # print("Home aggregate size:", sAgg)

        # calculate the unaltered collision rates
        self.colRates = np.zeros(self.para["nsize"])

        for s in range(self.para["nsize"]):
            self.monomer.homeAggregate.prop["sCol"] = sizes[s]
            # self.monomer.homeAggregate.prop["mCol"] = 4/3*np.pi*self.monomer.homeAggregate.prop["sCol"]**3*self.monomer.homeAggregate.prop["rhoAgg"]
            self.colRates[s] = self.collisionRate(t, r, z)

        # determine whether effective cross section is needed

        volFact = (sizes / sAgg) ** 3

        effTrue = volFact / self.feps
        effFalse = np.ones(self.para["nsize"])

        self.effectiveFact = np.where(volFact <= self.feps, effTrue, effFalse)

        self.effColRates = self.effectiveFact * self.colRates