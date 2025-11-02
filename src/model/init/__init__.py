from .initializers import InitializersMixin
from .titlecard import Titlecard


class InitMixin(InitializersMixin, Titlecard):

    def __init__(self):
        super().__init__()