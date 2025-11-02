import numpy as np


class CalculateDerivativeGrid:

    def calculateDerivative(self, quant, type):
        """
        Calculates the spatial gradients in quant, which is of the same shape as the ProDiMo model output.
        """

        # dDp/dz

        rVals, zVals = self.disk.model.x*self.auTOm, self.disk.model.z*self.auTOm

        form = self.disk.model.x.shape
        if type=="z":

            dQdZ = np.zeros(form)

            # Convert to log-space for numerical stability.
            dQdZ1 = np.log(quant[:,0:-1])
            dQdZ2 = np.log(quant[:,1::])

            dQdZint = (dQdZ2-dQdZ1)/(zVals[:,1::]-zVals[:,0:-1])

            a1 = (zVals[:,1:-1]-zVals[:,0:-2])/(zVals[:,2::]-zVals[:,0:-2])
            a2 = (zVals[:,2::]-zVals[:,1:-1])/(zVals[:,2::]-zVals[:,0:-2])

            dQdZ[:,1:-1] = a1*dQdZint[:,0:-1] + a2*dQdZint[:,1::]

            return dQdZ*quant

        elif type=="r":
            dQdR = np.zeros(form)

            dQdR1 = np.log(quant[0:-1,:])
            dQdR2 = np.log(quant[1::,:])


            dQdRint = (dQdR2-dQdR1)/(rVals[1::,:]-rVals[0:-1,:])

            a1 = (rVals[1:-1,:]-rVals[0:-2,:])/(rVals[2::,:]-rVals[0:-2,:])
            a2 = (rVals[2::,:]-rVals[1:-1,:])/(rVals[2::,:]-rVals[0:-2,:])

            dQdR[1:-1,:] = a1*dQdRint[0:-1,:] + a2*dQdRint[1::,:]

            return dQdR*quant