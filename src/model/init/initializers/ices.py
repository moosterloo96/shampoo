

class InitializeIces:

    def initIces(self, t_in, r_in, z_in):
        """
        Initiates the data structures in which we store the ice abundances of the monomer.
        """
        self.monomer.ice_sol = {}
        self.monomer.atom_sol = {}

        self.probeEnvironment(r_in, z_in, t_in, inside_loop=False)
        r, z, t = self.unpackVars(r_in, z_in, t_in)  # Convert input from AU and kyr to SI units.

        #### Inform initial ice mass from background model!!!

        # Calculate initial individual and total ice budget on the monomer.
        iceTot = 0
        for n in range(self.iceNum):
            rhoDust = self.environment["rhod"]  # Gives the dust mass density (kg/m3)
            rhoIce = self.environment["iceAbun" + self.disk.iceList[n]] * self.para["m" + self.disk.iceList[n]]
            mIce = rhoIce / rhoDust * self.monomer.prop["mMon"]  # Ice mass in kg
            # print(self.entities.iceList[n], "rhodust:",rhoDust, "rhoice:", rhoIce, "ratio:", mIce)########################################################
            if self.pisoBenchmark:
                if self.disk.iceList[n] == "H2O":
                    mIce = self.monomer.prop["mMon"]
                else:
                    mIce = 0

            self.monomer.ice_sol[self.disk.iceList[n]] = [mIce]
            iceTot += mIce

        self.monomer.iceTot_sol = [iceTot]  # In kg!

        # Calculate the desorption/adsorption rates.

        if self.store == 0 and self.legacyIce:
            for n in range(self.iceNum):
                self.monomer.ice_sol["ads" + self.disk.iceList[n]] = [
                    self.rateAdsorption(t, r, z, self.disk.iceList[n], init=True)]
                self.monomer.ice_sol["tds" + self.disk.iceList[n]] = [
                    self.rateDesorptionThermal(t, r, z, self.disk.iceList[n], init=True)]
                self.monomer.ice_sol["pds" + self.disk.iceList[n]] = [
                    self.rateDesorptionPhoto(t, r, z, self.disk.iceList[n], init=True)]
                self.monomer.ice_sol["qFactor" + self.disk.iceList[n]] = [np.nan]
        # Calculate the initial element budgets
        # for n in range(self.iceNum):

        # We also calculate the change in element mass contained in the ice
        # the element gain/loss rate in monomer mass/s
        # iceMass = (self.monomer.ice_sol[self.entities.iceList[n]])[-1]
        # eleNew = self.calcElementChange(iceMass, self.entities.iceList[n])

        # if n==0:
        #    self.monomer.ele_sol = np.array([eleNew])
        # else:
        #    self.monomer.ele_sol[-1,:] += np.where(eleNew<0, np.zeros(5), eleNew)
