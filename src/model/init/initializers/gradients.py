from time import process_time
from scipy.interpolate import RectBivariateSpline


class InitializeGradients:

    def initGradients(self, size=None):
        """
        Initializes the gradients required for turbulent diffusion formalism.
        """

        toc = process_time()

        rVals, zVals = self.disk.model.x * self.auTOm, self.disk.model.z * self.auTOm
        quant = self.calculateDiffusivity(rVals, zVals, size=size)
        self.diffusivity = quant
        self.disk.data["dDdzSq"] = self.calculateDerivative(quant, "z")
        self.disk.data["dDdrSq"] = self.calculateDerivative(quant, "r")

        quant = self.disk.model.rhog * 1e3
        self.disk.data["drhogdzSq"] = self.calculateDerivative(quant, "z")
        self.disk.data["drhogdrSq"] = self.calculateDerivative(quant, "r")

        self.disk.data["dDdz"] = self.disk.data["dDdzSq"]
        self.disk.data["dDdr"] = self.disk.data["dDdrSq"]
        self.disk.data["drhogdz"] = self.disk.data["drhogdzSq"]
        self.disk.data["drhogdr"] = self.disk.data["drhogdrSq"]

        for name in ["dDdz", "dDdr", "drhogdz", "drhogdr"]:
            self.disk.interpol[name] = RectBivariateSpline(self.disk.rVals, self.disk.zVals, self.disk.data[name],
                                                           kx=self.disk.order, ky=self.disk.order)

        tic = process_time()