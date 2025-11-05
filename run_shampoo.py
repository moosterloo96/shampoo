import pickle

from src.model import Model

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
pathdict["project"] = "."

name = "DemoModel"
disk = pickle.load(open("./input/disks/vfrag1_disk.pkl", "rb"))
config = {
            "parameter_folder": pathdict["input"] + name,
            "verbose": 0
        }
mod = Model(disk=disk, user_config=config)
mod.integrateMonomer()
pickle.dump(mod, open(pathdict["simulation"] + name + ".pkl", "wb"))