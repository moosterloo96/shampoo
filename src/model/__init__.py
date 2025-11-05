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

    def __init__(self, disk=None, user_config=None):
        super().__init__()

        if user_config is None:
            config = self.load_config()
        else:
            config = self.complement_config(user_config)
        print(config)
        self.initialize(disk=disk, config=config)
