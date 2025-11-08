from .operations import IntegratorOperations
from .timescales import Timescales

class IntegratorMixin(IntegratorOperations, Timescales):

    def __init__(self):
        super().__init__()