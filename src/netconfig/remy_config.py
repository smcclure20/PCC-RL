import sys
sys.path.insert(0, "/home/eecs/sarah/PCC-RL/src/netconfig")

from dna_pb2 import ConfigRange


def load_remy_config(filename):
    fd = open(filename, "rb")
    config_contents = fd.read()
    config = ConfigRange()
    config.ParseFromString(config_contents)
    return config