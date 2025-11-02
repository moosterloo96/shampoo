from time import process_time

import numpy as np


class Evaluator:

    def evaluateQuant(self, name, r, z):
        # r & z are in m.
        if self.verbose > 2:
            print("Interpolating " + name)

        tic = process_time()

        rEva = r / self.auTOm

        # if (self.cieslaBenchmark and (name!="rhog")):
        # zEva = 0*r
        # else:
        zEva = abs(z / r)

        if (name in ["rhog", "rhod", "nd", "chiRT"] or (name[0:3] in ["gas", "ice", "tot"]) or (
                name[0:5] in ["rhoda", "numda"]) or ("Abun" in name)):

            quant = 10 ** (self.disk.interpol[name](rEva, zEva, grid=False))
            if (self.pisoBenchmark) and (name == "rhog"):
                H = 1 / self.Omega(0, r, 0 / r) * np.sqrt(
                    self.para["kB"] * 120 * (rEva) ** (-3 / 7) / (self.para["muH"] * self.para["mp"]))
                quant = 20000 * ((rEva) ** (-1)) / (H * np.sqrt(2 * np.pi))

        elif name in ["dDdz", "drhogdz"]:
            # print(type(z))######################
            if isinstance(z, (np.floating, float)):
                # print(self.entities.interpol[name](rEva, zEva, grid=False))##########################
                if z < 0:
                    quant = -self.disk.interpol[name](rEva, zEva, grid=False)
                else:
                    quant = self.disk.interpol[name](rEva, zEva, grid=False)
                # print(quant)##############
            else:
                quant = self.disk.interpol[name](rEva, zEva, grid=False)
                quant[zEva < 0] *= -1

        else:
            if isinstance(zEva, (np.floating, float)):
                quant = float(self.disk.interpol[name](rEva, zEva, grid=False))
            else:
                quant = self.disk.interpol[name](rEva, zEva, grid=False)
            if (self.pisoBenchmark) and (name in ["Tg", "Td"]):
                quant = quant / quant * 120 * (rEva) ** (-3 / 7)
            elif (self.pisoBenchmark) and (name in ["soundspeed", "soundspeed0"]):
                T = quant / quant * 120 * (rEva) ** (-3 / 7)
                quant = np.sqrt(self.para["kB"] * T / (self.para["muH"] * self.para["mp"]))
            elif (self.cieslaBenchmark) and (name == "soundspeed"):
                quant = self.disk.interpol["soundspeed"](rEva, 0 * zEva, grid=False)

        toc = process_time()
        if self.verbose > 2:
            print("Interpolation complete: Elapsed CPU time is {:.2f} s".format(toc - tic))

        return quant