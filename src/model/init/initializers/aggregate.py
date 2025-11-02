import numpy as np

class InitializeAggregate:

    def initAggregate(self, t_in, r_in, z_in):
        self.redo_sAgg(t_in, r_in, z_in)
        self.redo_zMon(t_in, r_in, z_in)

    def redo_sAgg(self, t_in, r_in, z_in):
        """
        Auxiliary function to simulate collisional equilibrium. Draws the home aggregate size based off the background dust mass
        distribution.

        r_in - Initial radial monomer position in AU
        z_in - Initial vertical monomer position in AU
        """

        M = len(self.grainSizes)
        compDensity = np.zeros(M)

        for m in range(M):
            rhoName = "rhoda" + str(m)
            compDensity[m] = self.evaluateQuant(rhoName, r_in * self.auTOm, z_in * self.auTOm)

        probs = compDensity / (np.sum(compDensity))

        sAgg = np.random.choice(self.grainSizes, size=None, replace=True, p=probs)
        self.monomer.homeAggregate.prop["sAgg"] = sAgg
        self.monomer.homeAggregate.prop["mAgg"] = 4 / 3 * np.pi * self.monomer.homeAggregate.prop["rhoAgg"] * sAgg ** 3


    def redo_zMon(self, t_in, r_in, z_in):
        """
        Auxiliary function for initAggregate, redetermines the initial monomer depth, which usually is set to 0. Is called
        after the new initial home aggregate size has been calculated by redo_sAgg. Also determines whether the monomer
        starts exposed.

        r_in - Initial radial monomer position in AU
        z_in - Initial vertical monomer position in AU
        """

        self.monomer.prop["zMon"] = self.determineRandomPos()

        if self.monomer.prop["zMon"] <= self.monomer.prop["zCrit"]:
            self.monomer.exposed = True
        else:
            pExp = self.determinePExp()
            if np.random.rand() <= pExp:
                self.monomer.exposed = True
            else:
                self.monomer.exposed = False

            self.seedNo += 1
            np.random.seed(self.seedNo)