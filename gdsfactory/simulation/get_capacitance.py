import hashlib
import pathlib
import tempfile
from functools import partial
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Iterable, Sequence

import numpy as np
from pydantic import BaseModel

import gdsfactory as gf
from gdsfactory.name import clean_value
from gdsfactory.pdk import get_sparameters_path
from gdsfactory.typings import ComponentSpec
from gdsfactory.simulation.elmer.get_capacitance import run_capacitive_simulation_elmer
from gdsfactory.simulation.palace.get_capacitance import run_capacitive_simulation_palace




get_capacitance_elmer = partial(run_capacitive_simulation_elmer, tool="elmer")
get_capacitance_palace = partial(run_capacitive_simulation_palace, tool="palace")


if __name__ == "__main__":
    c = gf.components.mmi1x2()
    # p = get_sparameters_path_lumerical(c)
    # sp = np.load(p)
    # spd = dict(sp)
    # print(spd)
