from time import process_time

import numpy as np
from scipy.stats import loguniform

from src.model.entities.monomer import Monomer


class IntegratorOperations():
    def probeTimescales(self, size, species):
        """
        Calculates the timescales for a monomer of given size (in m) and species (label). t,r,z are in kyr/AU
        """

        R = len(self.disk.model.x[:, 0])
        Z = len(self.disk.model.z[0, :])

        tauSol = np.zeros((R, Z, 6))

        self.initGradients(size=size)  ##############! Repeat whenever size changes!!!
        self.initDensities()

        for rInd in range(R):
            for zInd in range(Z):
                print("Calculating... progress: {:.0f}/{:.0f} points".format((rInd * Z + zInd + 1), R * Z), end="\r")
                r_in = self.disk.model.x[rInd, zInd]
                z_in = self.disk.model.z[rInd, zInd]
                r, z, t = self.unpackVars(r_in, z_in, 0)
                self.monomer = Monomer(self, r, z, size=size)

                self.initIces(0, r_in, z_in)
                self.probeEnvironment(r_in, z_in, 0)
                self.calcColRates(0, r_in, z_in)

                tauSol[rInd, zInd, 0] = self.tauMr(t, r, z, Hp=None, size=size)
                tauSol[rInd, zInd, 1] = self.tauMz(t, r, z, Hp=None, size=size)
                tauSol[rInd, zInd, 2] = self.tauTub(t, r, z, Hg=None)
                tauSol[rInd, zInd, 3] = self.tauCol(t, r, z, physical=False)
                tauSol[rInd, zInd, 4] = self.tauAds(t, r, z, size, species)
                tauSol[rInd, zInd, 5] = self.tauDes(t, r, z, size, species)

        return tauSol


    def drawRandomPos(self):
        """
        If integrateMonomer receives randomize=True, we draw our monomer based on the 2D density distribution of the background model.
        """

        # r0 = 101
        #
        # while r0>(self.para["Rtaper"]/self.auTOm):
        #    r0 = (self.innerRdraw/np.random.power(self.para["epsilon"]))

        r0 = loguniform.rvs(self.innerRdraw, self.outerRdraw)
        z0 = r0 * (0.2 * np.random.rand() - 0.1)

        return r0, z0


    def unpackVars(self, r_in, z_in, t_in):
        """
        Auxiliary function in the dynamics module to convert the position in space and time from AU/kyr to SI.
        """

        r = r_in * self.auTOm  # convert from AU to SI
        z = z_in * self.auTOm  # convert from AU to SI
        t = t_in * 1e3 * self.sTOyr  # convert from kyr to SI

        return r, z, t


    def integrateMonomer(self, size=None, t_stop_in=None, t_start=0, r0=None, z0=None, randomize=None, discretePos=None,
                         timescaleMode=False, seed=None):
        """
        Evolves a monomer over time.

        size        - Monomer size in m
        t_stop_in   - Final integration time in yr
        t_start     - Starting integration time. Would not set this to anything different than zero.
        r0          - Initial radial position in AU.
        z0          - Initial vertical position in AU.
        """

        if not self.supverbose:
            print(" ")
            print(50 * "-")
            print(" ")
            print("Initializing monomer... ")
            print(" ")
            print(50 * "-")
            print(" ")

        # Input management

        if randomize == None:
            randomize = bool(self.paraDict["randomize"])

        if randomize:
            r0, z0 = self.drawRandomPos()
            self.paraDict["r0"] = r0
            self.paraDict["z0"] = z0
        elif (r0 == None) or (z0 == None):
            r0 = float(self.paraDict["r0"])
            z0 = float(self.paraDict["z0"])

        if not self.supverbose:
            print("Set initial monomer position to: r0 = {:.2f} AU and z0/r0 = {:.2f}".format(r0, z0 / r0))

        if size == None:
            size = float(self.paraDict["sMon"])

        if t_stop_in == None:
            self.t_stop = float(self.paraDict["t_stop"])
        else:
            self.t_stop = t_stop_in / 1e3  # convert to kyr

        # Choose random number setup
        # --------------------------
        if self.deterministic:
            if seed == None:
                try:
                    self.seedStart = int(self.paraDict["seed"])
                    self.seedNo = int(self.paraDict["seed"])
                except:
                    self.seedStart = np.random.randint(0, high=(2 ** 32 - 1), size=None, dtype=int)
                    self.seedNo = self.seedStart
            else:
                self.seedStart = seed
                self.seedNo = seed

        else:
            self.seedStart = np.random.randint(0, high=(2 ** 32 - 1), size=None, dtype=int)
            self.seedNo = self.seedStart

        if not self.supverbose:
            print("Monomer seed is: " + str(self.seedStart))
        np.random.seed(self.seedNo)
        # Initialize monomer
        # ------------------
        self.monomer = Monomer(self, r0, z0, size=size)
        # self.monomer.homeAggregate.prop["sAgg"] = size
        # self.monomer.prop["sMon"] = size

        self.monomer.r_sol = [self.monomer.initR]
        self.monomer.z_sol = [self.monomer.initZ]
        self.monomer.t_sol = [0]
        self.monomer.sAgg_sol = [self.monomer.homeAggregate.prop["sAgg"]]
        self.monomer.zMon_sol = [0]
        self.monomer.sec_sol = {}
        # We here also define a disk_interface array which attempts to keep track of whether data points have
        # become "corrupt" due to convergence failures of the integration routines.
        self.monomer.corrupt = False
        self.monomer.corruption = [int(self.monomer.corrupt)]

        self.monomer.exitIndex = 0
        self.monomer.exitTracker = [0]

        tn = 0
        ticTot = process_time()

        self.initGradients()  ##############! Repeat whenever size changes!!!

        if self.collisions:  ### calculate the size-dependent grain sizes
            self.initDensities()
            self.monomer.sec_sol["interaction"] = [0]
            verbose_string = "exposed"
            if self.colEq:  ### Update sAgg and zMon.
                self.initAggregate(0, r0, z0)
                self.monomer.sAgg_sol = [self.monomer.homeAggregate.prop["sAgg"]]
                self.monomer.zMon_sol = [self.monomer.prop["zMon"]]
                self.monomer.sec_sol["exposed"] = [int(self.monomer.exposed)]
                if int(self.monomer.exposed) == 0:
                    verbose_string = "unexposed"
            else:
                self.monomer.sec_sol["exposed"] = [1]
        if not self.supverbose:
            print("Monomer placed in aggregate:")
            print("sAgg:" + 10 * " " + "{:.2e} m".format(self.monomer.homeAggregate.prop["sAgg"]))
            print("zMon:" + 10 * " " + "{:.2e} m".format(self.monomer.prop["zMon"]))
            if self.monomer.exposed:
                verbose_string = "exposed"
            else:
                verbose_string = "unexposed"
            print("State:" + 10 * " " + verbose_string)

        # if self.trackice: # define the data structures for ice budgets.
        self.initIces(0, r0, z0)

        if not self.supverbose:
            if self.trackice:
                print("Initialized ice budget.")
            else:
                print("Skipped ice initialization.")

            print(50 * "-")
            print(" ")
            print("Finished monomer initialization, commencing time evolution...")
            print(" ")
            print(50 * "-")
            print(" ")

        probeEnvironmentDat = []
        collisionRateDat = []
        moveGrainDat = []
        doCollisionDat = []
        doIceEvolutionDat = []

        self.delta_t = self.constTimestep[1] * self.sTOyr

        # The main integration time loop
        # -------------------------------------------------------------------

        while (tn < self.t_stop):

            tocTot = process_time()
            progress = tn / self.t_stop
            if progress > 0:
                self.left = (tocTot - ticTot) / progress * (1 - progress)
            else:
                self.left = 0
            progress *= 100
            if not self.supverbose:
                print(
                    "Integration progress: {:.2f} % complete, estimated time left: {:.2f} s, current step size: {:.1e} yr".format(
                        progress, self.left, self.delta_t / self.sTOyr) + " " * 8)

            n = len(self.monomer.t_sol)
            if self.verbose > 2:
                print("Performing timestep " + str(n) + " at t = {:.2e} yr".format(self.monomer.t_sol[-1] * 1e3))

            r_now = self.monomer.r_sol[-1]
            z_now = self.monomer.z_sol[-1]
            t_now = self.monomer.t_sol[-1]
            #             if len(self.monomer.r_sol)<10:
            #                 print("Initialized sizes:", self.monomer.prop["sMon"], self.monomer.homeAggregate.prop["sAgg"])
            tic = process_time()
            self.probeEnvironment(r_now, z_now, t_now)  # Calculate interpolated quantities
            toc = process_time()
            probeEnvironmentDat.append(toc - tic)

            # Calculate collision rates
            if self.collisions:
                tic = process_time()
                self.calcColRates(t_now, r_now, z_now)
                toc = process_time()
                collisionRateDat.append(toc - tic)
            else:
                collisionRateDat.append(0)

            # Set the timestep
            # ------------------------
            if self.migration or self.collisions:
                if self.constTimestep[0]:
                    self.delta_t = self.constTimestep[1] * self.sTOyr
                else:
                    self.delta_t = self.determineDeltaT(t_now, r_now, z_now)
            else:
                self.delta_t = self.t_stop * 1e3 * self.sTOyr

            # Store the environment here
            # ------------------------
            if self.store == 0:
                self.storeEnvironment(r_now, z_now, t_now)

            # print(self.delta_t/(self.sTOyr))
            # delta_t is in seconds, because we only use it inside the routine.

            # Do ice formation
            if self.trackice:
                tic = process_time()
                self.doIceEvolution(r_now, z_now,
                                    t_now)  # Note that ice formation may decrease the stepsize if it turns out that ice formation goes very fast.
                toc = process_time()
                doIceEvolutionDat.append(toc - tic)
            else:
                doIceEvolutionDat.append(0)

            # Do collisions before migration
            if self.collisions:
                tic = process_time()
                self.doCollisions(r_now, z_now, t_now)
                if self.sizeChanged:
                    self.initGradients()  # once done, re-calculate the gradients because grain diffusion slopes have changed. Only do this if
                    # collision occured; otherwise the home aggregate size remains the same.
                toc = process_time()
                doCollisionDat.append(toc - tic)
                if self.store == 0:
                    if t_now == 0:
                        try:
                            self.monomer.vRel_sol = [self.vRel]
                        except:
                            self.vRel = 0
                            self.monomer.vRel_sol = [self.vRel]

                    else:
                        self.monomer.vRel_sol.append(self.vRel)
                self.monomer.sAgg_sol.append(self.monomer.homeAggregate.prop["sAgg"])
                self.monomer.zMon_sol.append(self.monomer.prop["zMon"])
            else:
                self.monomer.sAgg_sol.append(self.monomer.homeAggregate.prop["sAgg"])
                self.monomer.zMon_sol.append(0)
                if self.store == 0:
                    if t_now == 0:
                        self.monomer.vRel_sol = [0]
                    else:
                        self.monomer.vRel_sol.append(0)

            # Move the grain if we have set self.migration==True. We do this last; because we can treat all other processes locally.
            if self.migration:
                tic = process_time()

                r_new, z_new = self.moveGrain(r_now, z_now, t_now, n)
                # Move the grain to its new position

                toc = process_time()
                moveGrainDat.append(toc - tic)

                self.monomer.r_sol.append(r_new)
                self.monomer.z_sol.append(z_new)
            else:
                self.monomer.r_sol.append(r_now)
                self.monomer.z_sol.append(z_now)
                moveGrainDat.append(0)

            # Update corruption array
            self.monomer.corruption.append(int(self.monomer.corrupt))
            self.monomer.exitTracker.append(self.monomer.exitIndex)

            # Advance timestep
            t_new = self.monomer.t_sol[-1] + self.delta_t / (1e3 * self.sTOyr)  # Advance timestep

            tn = t_new

            if self.migration or self.collisions:
                self.monomer.t_sol.append(t_new)

        T = len(self.monomer.t_sol)
        if self.store == 0:
            self.storeEnvironment(self.monomer.r_sol[-1], self.monomer.z_sol[-1], self.monomer.t_sol[-1])
            try:
                self.monomer.vRel_sol.append(self.vRel)
            except:
                self.vRel = 0
                self.monomer.vRel_sol.append(self.vRel)

        tocTot = process_time()
        if self.verbose > -1:
            print("")
            print("Integration complete, elapsed CPU time is {:.2f} s".format(tocTot - ticTot))
            if self.printStatistics:
                print("")
                print("Function statistics:")
                print("-" * 50)
                print("self.calcColRates")
                tcollisionRateTot = np.sum(np.array(collisionRateDat))
                print("Total call time : {:.2f} CPUs".format(tcollisionRateTot))
                print("Average time per call : {:.2e} CPUs".format(tcollisionRateTot / T))
                print("")
                print("self.probeEnvironment")
                tProbeEnvironmentTot = np.sum(np.array(probeEnvironmentDat))
                print("Total call time : {:.2f} CPUs".format(tProbeEnvironmentTot))
                print("Average time per call : {:.2e} CPUs".format(tProbeEnvironmentTot / T))
                print("")
                print("self.doIceEvolution")
                tDoIceEvolutionTot = np.sum(np.array(doIceEvolutionDat))
                print("Total call time : {:.2f} CPUs".format(tDoIceEvolutionTot))
                print("Average time per call : {:.2e} CPUs".format(tDoIceEvolutionTot / T))
                print("")
                print("self.moveGrain ")
                tMoveGrainTot = np.sum(np.array(moveGrainDat))
                print("Total call time : {:.2f} CPUs".format(tMoveGrainTot))
                print("Average time per call : {:.2e} CPUs".format(tMoveGrainTot / T))
                print("")
                print("self.doCollisions")
                tdoCollisionTot = np.sum(np.array(doCollisionDat))
                print("Total call time : {:.2f} CPUs".format(tdoCollisionTot))
                print("Average time per call : {:.2e} CPUs".format(tdoCollisionTot / T))
                print("-" * 50)

        if self.migration or self.collisions:
            self.monomer.r_sol = np.array(self.monomer.r_sol)
            self.monomer.z_sol = np.array(self.monomer.z_sol)
            self.monomer.t_sol = np.array(self.monomer.t_sol)
            self.monomer.sAgg_sol = np.array(self.monomer.sAgg_sol)
            self.monomer.zMon_sol = np.array(self.monomer.zMon_sol)
            self.monomer.corruption = np.array(self.monomer.corruption)
            self.monomer.exitTracker = np.array(self.monomer.exitTracker)
        else:
            self.monomer.r_sol = self.monomer.r_sol[-1] * np.ones(T)
            self.monomer.z_sol = self.monomer.z_sol[-1] * np.ones(T)
            self.monomer.sAgg_sol = self.monomer.sAgg_sol[-1] * np.ones(T)
            self.monomer.zMon_sol = self.monomer.zMon_sol[-1] * np.ones(T)
            self.monomer.corruption = self.monomer.corruption[-1] * np.ones(T)
            self.monomer.exitTracker = self.monomer.exitTracker[-1] * np.ones(T)

        if self.store == 0:
            if self.collisions:
                self.monomer.vRel_sol = np.array(self.monomer.vRel_sol)
            for item in self.monomer.sec_sol.keys():
                self.monomer.sec_sol[item] = np.array(self.monomer.sec_sol[item])

        if self.trackice:
            # Calculate element ratios.
            # iceMassSum = np.zeros(len(self.monomer.t_sol))
            for n in range(self.iceNum):
                self.monomer.ice_sol[self.disk.iceList[n]] = np.array(self.monomer.ice_sol[self.disk.iceList[n]])
                # iceMassSum += self.monomer.ice_sol[self.entities.iceList[n]]
            self.monomer.iceTot_sol = np.array(self.monomer.iceTot_sol)
            if self.store == 0:
                self.monomer.ele_sol = self.calcElements()
                # Abundance ratio (number densities, not mass ratio)
                # 1.00797, 12.011, 14.0067, 15.9994, 32.06
                self.monomer.sec_sol["ratC/O"] = self.monomer.ele_sol[:, 1] / self.monomer.ele_sol[:, 3] * (
                            15.9994 / 12.011)
                self.monomer.sec_sol["ratN/O"] = self.monomer.ele_sol[:, 2] / self.monomer.ele_sol[:, 3] * (
                            15.9994 / 14.0067)
                self.monomer.sec_sol["ratS/O"] = self.monomer.ele_sol[:, 4] / self.monomer.ele_sol[:, 3] * (15.9994 / 32.06)

            self.integrationTime = tocTot - ticTot
            self.corruptionAmount = np.sum(self.monomer.corruption * self.monomer.t_sol) / np.sum(self.monomer.t_sol)
