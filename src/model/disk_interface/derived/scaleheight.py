import numpy as np


class ScaleHeight:

    def calculateScaleHeight(self, r, method="mcfost", kind="gas", size=None):
        """
        r: Distance from star in m.
        Calculates the scale height in m.
        """

        # Alternatively, we calculate the pressure scale height
        if method == "pressure":

            soundspeed = self.environment["soundspeed0"]

            omega = self.Omega(0, r, 0 / r)

            H = soundspeed / omega

        elif method == "mcfost":

            H = self.para["H0"] * (r / self.para["r0"]) ** self.para["beta_flare"]

        if (kind == "dust" or kind == "dustProdimo"):
            fact = np.sqrt(self.para["a_settle"] / (
                        np.sqrt(3) * self.Stokes(0, r, 0, size=size, midplane=True) + self.para[
                    "a_settle"]))  # Using the expression of Youdin & Lithwick 2007
            H *= fact
        # elif kind=="dustProdimo":
        #     fact = min([np.sqrt(self.para["a_settle"]/(self.Stokes(0, r, 0, size=size, midplane=True)*np.sqrt(3))),1]) # Using the expression of Dubrulle+ 1995
        #     H *= fact
        # if kind in ["dust","dustProdimo"]:
        #    print("Input r: ",r)
        #    print("Hp/Hg:", fact, "St:", self.Stokes(0, r, 0, size=size, midplane=True), "alpha:", self.para["a_settle"])
        return H