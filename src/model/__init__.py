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
                 disk=None,
                 species=["H2O", "CO", "CO2", "CH4", "NH3", "H2S"],
                 disk_folder="../ShampooBackground",
                 parameter_folder=None,
                 verbose=None,
                 migration=None,
                 diffusion=None,
                 collisions=None,
                 trackice=None,
                 debug=False,
                 save_csv=[False, None],
                 phi=None,
                 breakIce=False,
                 legacyIce=False,
                 colEq=False,
                 storage_style=0,
                 readsorption=False,
                 supverbose=False,
                 qTest=False,
                 qTestAds=False,
                 qTestMig=False,
                 activeML=True):
        super().__init__()
        #TODO: Add validator.

        self.initialize(disk=disk, species=species, disk_folder=disk_folder,
             parameter_folder=parameter_folder, verbose=verbose, migration=migration, diffusion=diffusion, collisions=collisions, trackice=trackice,
             debug=debug, save_csv=save_csv, phi=phi, breakIce=breakIce, legacyIce=legacyIce, colEq=colEq,
             storage_style=storage_style, readsorption=readsorption, supverbose=supverbose, qTest=qTest, qTestAds=qTestAds, qTestMig=qTestMig,
             activeML=activeML)