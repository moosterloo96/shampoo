import pickle
from pathlib import Path

from src.model import Model

if __name__ == "__main__":

    pathdict = {}

    # Root project directory where the shampoo.py and this notebook are located.
    pathdict["project"] = str(Path("..").resolve())

    # Change this variable to the folder where your ProDiMo model output is located.
    pathdict["disk"] = pathdict["project"] + "/input/prodimo/vfrag1"

    # Location and name of the input file.
    pathdict["input"] = pathdict["project"] + "/input/parameters/"

    # Folder where the model data of monomer simulations are saved and where simulations are loaded from.
    pathdict["simulation"] = pathdict["project"] + "/output/monomers/"

    # Folder where the figure files are saved.
    pathdict["figures"] = pathdict["project"] + "/output/figures/"

    name = "DemoModel"
    disk = pickle.load(pathdict["project"] + "/input/disks/vfrag1_disk.pkl", "rb")
    config = {
                "parameter_folder": pathdict["input"] + name,
                "verbose": 0
            }
    mod = Model(disk=disk, user_config=config)
    mod.integrateMonomer()
    pickle.dump(mod, open(pathdict["simulation"] + name + ".pkl", "wb"))