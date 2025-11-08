from .common import CommonCollisionsMethods
from .operations import CollisionOperations

class CollisionsMixin(CommonCollisionsMethods, CollisionOperations):

    def __init__(self):
        super().__init__()