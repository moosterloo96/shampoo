import concurrent.futures
import traceback

import numpy as np
from scipy.integrate import solve_ivp


class IceEvolutionOperations:

    def calcElements(self):
        """
        This method returns the respective masses of H, C, N, O & S in kg present in the monomer ice mantle.

        This method should only be used after the main integrateMonomer loop.
        """

        # diffMass *= self.monomer.prop["mMon"] # gain of molecule X in kg
        # Not needed as we are now tracing the ice mass in kg.

        self.monomer.ele_sol = np.zeros((len(self.monomer.ice_sol[self.disk.iceList[0]]), 5))

        # self.monomer.ice_sol[self.entities.iceList[n]]

        for n in range(self.iceNum):
            iceName = self.disk.iceList[n]

            # convert to moles
            conv = self.para["m" + iceName] / self.para[
                "u"] * 1e-3  # m+Icename is in kg, so conv represents kg/mol for molecule X

            massMol = self.monomer.ice_sol[iceName] / conv  # mass of molecule X in moles

            molElement = self.disk.elementDict[
                iceName]  # gives the number of moles of each element in one mole of molecule X
            # calculate how many moles there of every element due to the presence of molecule X.

            kgElement = molElement * self.para["mElements"]  # Convert from moles to kg.
            # print(np.outer(massMol kgElement).shape)
            self.monomer.ele_sol += np.outer(massMol,
                                             kgElement)  # gives the mass change in moles for each element. Multiply with molar mass to find gain in kg.

        return self.monomer.ele_sol


    def iceMassChange(self, t, r, z, iceName, iceList=None):
        """
        Calculates the gain/loss of a certain ice species in kg/s for the whole monomer ice mantle.
        """

        surface = 4 * np.pi * self.monomer.prop["sMon"] ** 2
        cross = 2 * np.pi * self.monomer.prop["sMon"] ** 2
        if self.qTest:
            if self.qTestAds:
                self.monomer.exposed = True
            else:
                self.monomer.exposed = False

        if self.monomer.exposed:
            # If the monomer is exposed we do adsorption, thermal desorption and photodesorption.
            ads = self.rateAdsorption(t, r, z, iceName, iceList=iceList)
            tds = self.rateDesorptionThermal(t, r, z, iceName, dustT=None, iceList=iceList)
            pds = self.rateDesorptionPhoto(t, r, z, iceName, iceList=iceList)

            # prodimo uses surface, we use cross section
            diffMass = cross * ads - surface * tds - cross * pds

            # extrafact = self.monomer.prop["mMon"]*self.environment["nd"]/self.environment["rhod"]
            #
            # diffMass *= 1#extrafact#############################

            qFactor = np.nan
        else:

            if self.readsorption:
                # if this lever is true we assume all thermally desorbed ice is readsorbed.
                diffMass = 0
                tds = 0
                pds = 0
                ads = 0
                qFactor = 1
            elif self.readsorption == None:
                tds, qFactor = self.rateDesorptionInternal(t, r, z, iceName, iceList=iceList)  ########## TO DO: iceList
                # Note that in this case the q-factor is defined in the above method.
                diffMass = -surface * tds
                pds = 0
                ads = 0
            else:
                ## Otherwise we only do thermal desorption.
                tds = self.rateDesorptionThermal(t, r, z, iceName, dustT=None, iceList=iceList)
                diffMass = -surface * tds
                # Otherwise nothing happens
                # diffMass = 0
                # tds = 0
                pds = 0
                ads = 0
                qFactor = 0

        if self.legacyIce:
            return diffMass, ads, tds, pds, qFactor  # In the old formalism we can return the rates
        else:
            # Otherwise we store them in a class variable.
            # if self.t_track==0.:
            #    iceInd = self.entities.iceList.index(iceName)
            #    self.rateTrack[iceInd, 0] = cross*ads
            #    self.rateTrack[iceInd, 1] = surface*tds
            #    self.rateTrack[iceInd, 2] = cross*pds
            #    print("Clause triggered at t={:.2f}".format(t_in/(self.sTOyr*1e3)))

            return diffMass


    def doIceEvolutionMyOwn(self, r_in, z_in, t_in):
        r, z, t = self.unpackVars(r_in, z_in, t_in)

        n = 0
        diffMassPrelim = {}
        adsPrelim = {}
        tdsPrelim = {}
        pdsPrelim = {}
        qPrelim = {}

        deltaTfloor = self.delta_t_floor * self.sTOyr

        zeroAccumulateFlag = {}

        # ------------------------------------------------------------------------

        while n < self.iceNum:

            # print("Calculating for "+self.entities.iceList[n]+" ice")
            diffMass, adsPrelim[self.disk.iceList[n]], tdsPrelim[self.disk.iceList[n]], pdsPrelim[self.disk.iceList[n]], \
            qPrelim[self.disk.iceList[n]] = self.iceMassChange(t, r, z, self.disk.iceList[n])
            diffMassPrelim[self.disk.iceList[n]] = diffMass * self.delta_t

            # Calculate the ratio, make sure to catch errors where we have 0/0.
            diffZeroBool = diffMassPrelim[self.disk.iceList[n]] == 0
            icesZeroBool = (self.monomer.ice_sol[self.disk.iceList[n]])[-1] == 0.

            if icesZeroBool:
                ratio = 1
            elif icesZeroBool and diffZeroBool:
                ratio = 0
            else:
                ratio = diffMassPrelim[self.disk.iceList[n]] / (self.monomer.ice_sol[self.disk.iceList[n]])[-1]

            ratioBool = ratio > self.fice
            # is the relative mass change for species n exceeded?

            floorBool = self.delta_t > deltaTfloor
            # have we not yet reached the minimum allowed timestep?

            bareBool = (self.monomer.ice_sol[self.disk.iceList[n]])[-1] < 1e-50
            # Was the monomer not just iceless?
            # if bareBool:
            # print("ding")
            notConstTimeBool = not self.constTimestep[0]
            # do we not use constant timestep?

            n += 1

            if (ratioBool) and (floorBool) and (notConstTimeBool) and (not bareBool):
                self.delta_t /= 10
                # Then we adjust the timestep if needed; note that we do this AFTER the shortest timescale has been chosen.

                if self.delta_t < deltaTfloor:
                    self.delta_t = deltaTfloor
                    zeroAccumulateFlag[self.disk.iceList[n - 1]] = True
                    n = 0
                else:
                    zeroAccumulateFlag[self.disk.iceList[n - 1]] = False
                    n = 0

            elif (ratioBool) and (bareBool):
                # This is a warning condition for when we accumulate too much ice too quickly.
                # We accept the change right away and calculate the desorption rate with the new changes later.
                # In this way we prevent unnecesary decreases in timestep down to the floor of delta_t.
                zeroAccumulateFlag[self.disk.iceList[n - 1]] = True
            else:
                # print(n-1, self.entities.iceList[n-1])
                zeroAccumulateFlag[self.disk.iceList[n - 1]] = False

                # ------------------------------------------------------------------------

        for n in range(self.iceNum):
            # First make the preliminary ads/tds/pds rates final.
            (self.monomer.ice_sol["ads" + self.disk.iceList[n]]).append(adsPrelim[self.disk.iceList[n]])
            (self.monomer.ice_sol["tds" + self.disk.iceList[n]]).append(tdsPrelim[self.disk.iceList[n]])
            (self.monomer.ice_sol["pds" + self.disk.iceList[n]]).append(pdsPrelim[self.disk.iceList[n]])
            (self.monomer.ice_sol["qFactor" + self.disk.iceList[n]]).append(qPrelim[self.disk.iceList[n]])

            iceNew = (self.monomer.ice_sol[self.disk.iceList[n]])[-1] + diffMassPrelim[self.disk.iceList[n]]

            self.monomer.ice_sol[self.disk.iceList[n]].append(max([0, iceNew]))  # at the end we add the new ice solution

        # Once we are done with dealing with the various ices, we calculate the new total amount of ice on the monomer.
        iceTot = self.calcIceTot()
        self.monomer.iceTot_sol.append(iceTot)

        # ------------------------------------------------------------------------

        diffMassPrelimZ = {}
        adsPrelimZ = {}
        tdsPrelimZ = {}
        pdsPrelimZ = {}
        qPrelimZ = {}

        for n in range(self.iceNum):
            # An extra loop to check if there were any zero-accumulations. If yes, check whether everything is
            # lost again next iteration to avoid oscilatory ice behaviour.

            if zeroAccumulateFlag[self.disk.iceList[n]]:
                diffMass, adsPrelimZ[self.disk.iceList[n]], tdsPrelimZ[self.disk.iceList[n]], pdsPrelimZ[
                    self.disk.iceList[n]], qPrelimZ[self.disk.iceList[n]] = self.iceMassChange(t, r, z,
                                                                                               self.disk.iceList[n])
                diffMassPrelimZ[self.disk.iceList[n]] = diffMass * self.delta_t

                if -diffMassPrelimZ[self.disk.iceList[n]] >= (self.monomer.ice_sol[self.disk.iceList[n]])[-1]:
                    iceNew = (self.monomer.ice_sol[self.disk.iceList[n]])[-1] + diffMassPrelimZ[self.disk.iceList[n]]
                    (self.monomer.ice_sol[self.disk.iceList[n]])[-1] = max([0, iceNew])

                    # print("Zeroclause triggered")

                    (self.monomer.ice_sol["ads" + self.disk.iceList[n]])[-1] = adsPrelimZ[self.disk.iceList[n]]
                    (self.monomer.ice_sol["tds" + self.disk.iceList[n]])[-1] = tdsPrelimZ[self.disk.iceList[n]]
                    (self.monomer.ice_sol["pds" + self.disk.iceList[n]])[-1] = pdsPrelimZ[self.disk.iceList[n]]
                    (self.monomer.ice_sol["qFactor" + self.disk.iceList[n]])[-1] = qPrelimZ[self.disk.iceList[n]]

        self.monomer.iceTot_sol[-1] = self.calcIceTot()

        if self.pisoBenchmark:
            self.monomer.prop["mMon"] = self.monomer.iceTot_sol[-1]
            self.monomer.prop["sMon"] = (3 * self.monomer.prop["mMon"] / (4 * np.pi * self.monomer.prop["rho_mat"])) ** (
                        1 / 3)
            self.monomer.homeAggregate.prop["mAgg"] = self.monomer.iceTot_sol[-1]
            self.monomer.homeAggregate.prop["sAgg"] = (3 * self.monomer.homeAggregate.prop["mAgg"] / (
                        4 * np.pi * self.monomer.homeAggregate.prop["rhoAgg"])) ** (1 / 3)


    def scipyAuxFunc(self, t_in, y_in, *args):
        """
        Auxilary function which calculates the derivatives for the ice changes.

        Units:
        Position: AU
        y_in: potatoes (ice mass)
        t_in: kyr (time)
        """

        r_in = args[0]
        z_in = args[1]
        t_in += args[2]

        r, z, t = self.unpackVars(r_in, z_in, t_in)

        dydt = np.zeros(len(y_in))

        iceList = {}

        for n in range(self.iceNum):
            y_in[n] = max(y_in[n], 0)
            iceList[self.disk.iceList[n]] = y_in[n] * self.numFact  # This is more consistent with ice bookkeeping in
            # self.iceMassChange. Maybe get rid of dictionaries altogether?
            # Also converts in iceList from potatoes to kg

        for n in range(self.iceNum):
            # note we have to convert from kg/s to potatoes/kyr
            dydt[n] = self.iceMassChange(t, r, z, self.disk.iceList[n], iceList=iceList) * (1e3 * self.sTOyr) / (
                self.numFact)

            ####### Failsave for when ice abundances get too low.
            if iceList[self.disk.iceList[n]] / self.monomer.prop["mMon"] < self.floorval and dydt[n] < 0:
                dydt[n] = 0

        return dydt


    def advanceIceMantle(self, t_start, t_stop, y0, position, integrator):
        self.printFlag = True

        t_eval = np.linspace(t_start, t_stop, 2)

        # Define the settings for the integrator.
        atol = self.atol[0] if self.monomer.exposed else self.atol[1]
        rtol = self.rtol[0] if self.monomer.exposed else self.atol[0]
        method = integrator

        # Define an auxiliary function to run the ivp.
        def run_ivp():
            return solve_ivp(self.scipyAuxFunc, (t_start, t_stop), y0, t_eval=t_eval, args=position, method=method,
                        atol=atol, rtol=rtol)

        executor = concurrent.futures.ThreadPoolExecutor()
        future = executor.submit(run_ivp)
        try:
            sol = future.result(timeout=1.5)
        except concurrent.futures.TimeoutError as e:
            future.cancel()
            executor.shutdown(wait=False)
            raise TimeoutError("Integration did not converge.") from e

        if sol["status"] == 0:
            success = sol["success"]
        else:
            success = False

        return sol, success

    def doIceEvolutionSciPy(self, r_in, z_in, t_in):
        """
        New code which solves the time evolution of the monomer ice using pre-existing ODE-integration routines.


        Classification of exit index:
        0 - Integrator worked as intended
        1 - Monomer is in the inner entities numerical regime, no integration peformed.
        2 - Integrator worked as intended in this timestep, but there have been timeouts in the past.
        3 - Integrator worked as intended in this timestep, but there have been manual resets in the past.
        ---- Default cutoff ----
        4 - Integrator worked as intended in this timestep, but there have been convergence errors in the past.
        5 - Timeout error.
        6 - Convergence error, ice mantle reset due to high local UV radiation.
        7 - Convergence error.
        """

        # abunArr = np.array([self.environment["gasAbun"+self.entities.iceList[n]] for n in range(self.iceNum)])
        # abunFact = np.mean(abunArr)/np.max(abunArr)

        # originally we had just the monomer mass
        self.numFact = self.monomer.iceTot_sol[-1] * self.iceScaleFact * self.monomer.prop["mMon"]  # kg/potatoes
        y0 = np.array([(self.monomer.ice_sol[self.disk.iceList[n]])[-1] / self.numFact for n in range(self.iceNum)])
        # such that y0 is in potatoes

        if self.migration or self.collisions:
            delt = self.delta_t / (1e3 * self.sTOyr)
        else:
            delt = self.t_stop

        t_start = 0
        t_stop = float(delt) # delt is np.float64
        # print(t_start, t_stop, t_in)

        integratorList = ["LSODA", "Radau", "BDF"]
        # LSODA should be most flexible, but can be worth trying others in case of failure because
        # Radau & BDF are written for stiff systems.
        I = len(integratorList)
        i = 0

        success = False
        self.monomer.exitIndex = 7  # Exit index 7 is the worst, it means an unhandled convergence error.

        if r_in > 1:
            while ((not success) and (i < I)):
                try:
                    sol, success = self.advanceIceMantle(t_start, t_stop, y0, (r_in, z_in, t_in),
                                                         integrator=integratorList[i])
                except TimeoutError as e:
                    print(f"Integration timed out: {e}")
                    # If integrators take too long, we keep the ice mantle from previous timestep.
                    self.monomer.exitIndex = 5  # For now we have a timeout of the routines.
                    # Success remains false in this case.
                    sol = {}

                    sol["y"] = self.floorval * self.monomer.prop["mMon"] / self.numFact * np.ones((self.iceNum, 2))
                    sol["y"][:, 0] = y0
                    sol["y"][:, 1] = y0
                    sol["t"] = np.linspace(t_start, t_stop, 2)
                i += 1

            # An extra clause to reset ice mantles strongly affected by UV radiation (sometimes causes numerical issues).
            if (not success) and (self.environment["chiRT"] > 1) and (self.monomer.exposed):
                self.monomer.exitIndex = 6
                sol = {}
                sol["y"] = self.floorval * self.monomer.prop["mMon"] / self.numFact * np.ones((self.iceNum, 2))
                sol["t"] = np.linspace(t_start, t_stop, 2)

            # Make sure we have the correct exit index if errors occured in the past.
            if success:
                if self.monomer.corrupt:
                    self.monomer.exitIndex = np.max(self.monomer.exitTracker) - 3
                else:
                    self.monomer.exitIndex = 0
        else:
            self.monomer.exitIndex = 1
            success = True
            sol = {}
            sol["y"] = self.floorval * self.monomer.prop["mMon"] / self.numFact * np.ones((self.iceNum, 2))
            sol["t"] = np.linspace(t_start, t_stop, 2)

        if (i - 1) > 0:
            print("Default ice routine failed, monomer exited index ", self.monomer.exitIndex, " with backup routine ",
                  integratorList[i - 1], ". Success: ", success)

        solution = sol["y"] * self.numFact

        limit = self.floorval * self.monomer.prop["mMon"]

        solution[solution < limit] = limit

        if self.migration or self.collisions:
            for n in range(self.iceNum):
                self.monomer.ice_sol[self.disk.iceList[n]].append(solution[n, -1])
        else:
            self.monomer.t_sol = sol["t"]
            for n in range(self.iceNum):
                self.monomer.ice_sol[self.disk.iceList[n]] = solution[n, :]

        # Once we are done with dealing with the various ices, we calculate the new total amount of ice on the monomer.
        iceTot = self.calcIceTot()
        self.monomer.iceTot_sol.append(iceTot)

        # Update corruption tracker
        if (not success):
            self.monomer.corrupt = True

        # print(self.monomer.t_sol, self.monomer.ice_sol["H2O"])


    def doIceEvolution(self, r_in, z_in, t_in):
        """
        At each timestep, we run this algorithm to update the abundances in the ice.
        """

        # print("b -",self.monomer.exposed)
        if self.legacyIce:
            self.doIceEvolutionMyOwn(r_in, z_in, t_in)
        else:
            self.doIceEvolutionSciPy(r_in, z_in, t_in)
        # print("a -",self.monomer.exposed)