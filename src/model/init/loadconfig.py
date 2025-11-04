import json

class LoadConfig:

    def load_config(self, path="../"):

        with open(path + "config.json") as f:
            config = json.load(f)

        return config