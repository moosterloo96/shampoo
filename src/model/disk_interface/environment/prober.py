class Prober:

    def probeEnvironment(self, r_now, z_now, t_now, inside_loop=True):
        """
        Infers the local entities properties to use in a given timestep. Input in AU and kyr.
        """

        if t_now == 0:
            self.environment = {}

        r_in, z_in, t_in = self.unpackVars(r_now, z_now, t_now)

        # For now we calculate the gas/dust density, temperature and soundspeed.

        for cond in ["rhog", "rhod", "Tg", "Td", "nd", "soundspeed", "chiRT", "nd"]:
            self.environment[cond] = self.evaluateQuant(cond, r_in, z_in)

        if self.diffusion or self.collisions:
            for cond in ["dDdz", "dDdr", "drhogdz", "drhogdr"]:
                self.environment[cond] = self.evaluateQuant(cond, r_in, z_in)

        self.environment["soundspeed0"] = self.evaluateQuant("soundspeed", r_in, 0 / r_in)

        if self.trackice:
            for spec in self.disk.iceList:
                self.environment["gasAbun" + spec] = self.evaluateQuant("gasAbun" + spec, r_in, z_in)
                self.environment["iceAbun" + spec] = self.evaluateQuant("iceAbun" + spec, r_in, z_in)
                self.environment["totAbun" + spec] = self.evaluateQuant("totAbun" + spec, r_in, z_in)


    def storeEnvironment(self, r_now, z_now, t_now):
        """
        Stores the environment in dedicated arrays.
        """
        # print(self.environment)
        r_in, z_in, t_in = self.unpackVars(r_now, z_now, t_now)
        if t_in == 0:
            for cond in self.trackListEnv:
                self.monomer.sec_sol[cond] = [self.environment[cond]]
            for cond in self.trackListFun:
                if cond == "St":
                    self.monomer.sec_sol[cond] = [self.Stokes(t_in, r_in, z_in)]
                elif cond == "delta_t":
                    self.monomer.sec_sol[cond] = [self.delta_t]
            if self.trackice:
                for species in self.disk.iceList:
                    self.monomer.sec_sol["gasAbun" + species] = [self.environment["gasAbun" + species]]
                    self.monomer.sec_sol["iceAbun" + species] = [self.environment["iceAbun" + species]]
        else:
            for cond in self.trackListEnv:
                self.monomer.sec_sol[cond].append(self.environment[cond])
            for cond in self.trackListFun:
                if cond == "St":
                    self.monomer.sec_sol[cond].append(self.Stokes(t_in, r_in, z_in))
                elif cond == "delta_t":
                    self.monomer.sec_sol[cond].append((self.monomer.t_sol[-1] - self.monomer.t_sol[-2]) * self.sTOyr * 1e3)
            if self.trackice:
                for species in self.disk.iceList:
                    self.monomer.sec_sol["gasAbun" + species].append(self.environment["gasAbun" + species])
                    self.monomer.sec_sol["iceAbun" + species].append(self.environment["iceAbun" + species])