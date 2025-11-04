from .initializers import InitializersMixin
from .titlecard import Titlecard
from .loadconfig import LoadConfig


class InitMixin(InitializersMixin, Titlecard, LoadConfig):

    def __init__(self):
        super().__init__()