from .common import CommonIceEvolutionMethods
from .operations import IceEvolutionOperations

class IceEvolutionMixin(CommonIceEvolutionMethods, IceEvolutionOperations):

    def __init__(self):
        super().__init__()