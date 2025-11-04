from .init import InitMixin
from .disk_interface import DiskInterface
from .collisions import CollisionsMixin
from .dynamics import DynamicsMixin
from .ice import IceEvolutionMixin
from .integrator import IntegratorMixin

class Model(InitMixin, DiskInterface, CollisionsMixin, DynamicsMixin, IceEvolutionMixin, IntegratorMixin):
    """
    The master class which stores all the information of the full simulation.
    """

    def __init__(self,
                 disk=None):
        super().__init__()
        config = self.load_config()

        self.initialize(disk=disk, config=config)
