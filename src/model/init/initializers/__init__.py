from .aggregate import InitializeAggregate
from .densities import InitializeDensities
from .gradients import InitializeGradients
from .ices import InitializeIces
from .parameters import InitializeParameters
from .maininitializer import MainInitializer

class InitializersMixin(MainInitializer, InitializeAggregate, InitializeDensities, InitializeGradients, InitializeIces,
                  InitializeParameters):

    def __init__(self):
        super().__init__()