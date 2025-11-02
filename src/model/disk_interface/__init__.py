from .derived.derivativegrid import *
from .derived.diffusivitygrid import *
from .derived.scaleheight import *
from .environment.evaluator import *
from .environment.prober import *

class DiskInterface(CalculateDerivativeGrid, CalculateDiffusivityGrid, ScaleHeight, Prober, Evaluator):

    def __init__(self):
        super().__init__()