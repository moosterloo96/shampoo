import pickle
from src.model.entities.disk import Disk

pathdict = {}

# Change this variable to the folder where your ProDiMo model output is located.
pathdict["disk"] = "./input/prodimo/vfrag1"

# Location and name of the input file.
pathdict["input"] = "./input/parameters/"

# Folder where the model data of monomer simulations are saved and where simulations are loaded from.
pathdict["simulation"] = "./output/monomers/"

# Folder where the figure files are saved.
pathdict["figures"] = "./output/figures/"

# Root project directory where the shampoo.py and this notebook are located.
pathdict["project"] = "./"



disk = Disk(species=["H2O", "CO", "CO2", "CH4", "CH3OH", "NH3", "H2S", "SO2", "OCS"], folder=pathdict["disk"], modelName="ProDiMo.out",
                    t_index="{:04d}".format(5))

pickle.dump(disk, open("input/disks/vfrag1_disk.pkl", "wb"))