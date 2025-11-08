import pickle
from pathlib import Path

from src.model.entities.disk import Disk

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



disk = Disk(species=["H2O", "CO", "CO2", "CH4", "CH3OH", "NH3", "H2S", "SO2", "OCS"], folder=pathdict["disk"], modelName="ProDiMo.out",
                    t_index="{:04d}".format(5))

pickle.dump(disk, open(pathdict["project"] + "/input/disks/vfrag1_disk.pkl", "wb"))