import numpy as np


class DynamicsOperations:

    def moveGrain(self, r_old, z_old, t, n):
        """
        Calculates the displacement of the home aggregate/monomer
        """

        #         succes=False

        #         failcount = 0
        #         maxFail = 20

        #         while (succes==False)and(failcount<maxFail):

        # We here take abs(z) as we assume our entities to be symmetric around the midplane.

        randr, randz = 2 * np.random.rand(2) - 1

        self.seedNo += 1
        np.random.seed(self.seedNo)

        if self.qTest and (not self.qTestMig):
            del_r = -100 * self.delta_t / (100 * 1e3 * self.sTOyr)
        elif (self.fixR) or (self.breakIce):
            del_r = 0
        else:
            del_r = self.r_der(r_old, z_old, t, randr)

        if self.qTest and (not self.qTestMig):
            del_z = 0
        elif (self.fixZ) or (self.breakIce):
            del_z = 0
        else:
            del_z = self.z_der(r_old, z_old, t, randz)

        r_new = r_old + del_r

        if (r_new < self.para["r_inner_cutoff"]):
            print("The monomer has drifted interior to the inner entities wall.")
            self.delta_t = self.t_stop * 1e3 * self.sTOyr
            r_new = self.para["r_inner_cutoff"]
        elif (r_new > 0.999 * self.para["r_outer"] / self.auTOm):
            print("The monomer has drifted out of the entities.")
            self.delta_t = self.t_stop * 1e3 * self.sTOyr
            r_new = self.para["r_outer"] / self.auTOm

        z_new = z_old + del_z

        if self.breakIce:
            r_new = 20 - t
            z_new = 0.01 * r_new

        if abs(z_new / r_new) > 0.5:
            print("The monomer has drifted above abs(z/r)=0.5.")
            z_new = 0.5 * r_new

        return r_new, z_new

    def z_der(self, r_in, z_in, t_in, rand):

        r, z, t = self.unpackVars(r_in, z_in, t_in)

        if self.diffusion == True:
            randZ = rand * np.sqrt(2 / self.xi * self.particleDiffusivity(t, r, z) * self.delta_t)
            z_der = self.velocity_eff_z(t, r, z) * self.delta_t + randZ
            # print("randz = ", randZ)
            # print("v_eff_z_final = ", self.velocity_eff_z(t,r,z)*self.delta_t)

        elif self.diffusion == False:
            z_der = self.velocity_z(t, r, z) * self.delta_t

        z_der /= self.auTOm  # convert from m to AU

        if self.fixZ:
            z_der *= 0
        # print("z_der",z_der)
        return z_der

    def r_der(self, r_in, z_in, t_in, rand):

        r, z, t = self.unpackVars(r_in, z_in, t_in)

        if self.diffusion == True:
            randR = rand * np.sqrt(2 / self.xi * self.particleDiffusivity(t, r, z) * self.delta_t)
            r_der = self.velocity_eff_r(t, r, z) * self.delta_t + randR

        elif self.diffusion == False:
            r_der = self.velocity_r(t, r, z) * self.delta_t
        ################################################################################ Comment the thing below out; it is meant for debugging purposes
        #         rate = 25
        #         if t_in<.5:
        #             r_der = -rate*self.delta_t
        #         else:
        #             r_der = rate*self.delta_t
        r_der /= self.auTOm  # convert from m to AU

        if self.fixR:
            r_der *= 0
        # print("r_der",r_der)
        return r_der