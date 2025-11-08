from .common import CommonDynamicsMethods
from .operations import DynamicsOperations

class DynamicsMixin(CommonDynamicsMethods, DynamicsOperations):

    def __init__(self):
        super().__init__()