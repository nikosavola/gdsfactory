from functools import partial
from pathlib import Path
from typing import Any, Mapping, Optional

import gdsfactory as gf
from gdsfactory.pdk import get_sparameters_path
# from gdsfactory.simulation.elmer.get_capacitance import run_scattering_simulation_elmer
from gdsfactory.simulation.palace.get_scattering import run_scattering_simulation_palace
from gdsfactory.typings import ComponentSpec, ElectrostaticResults


def get_scattering(
    component: ComponentSpec,
    simulator: str = "palace",
    simulator_params: Optional[Mapping[str, Any]] = None,
    simulation_folder: Optional[Path | str] = None,
    **kwargs,
) -> ElectrostaticResults:
    """Simulate component with a full-wave simulation and return scattering matrix.

    Args:
        component: component or component factory.
        simulator: Simulator to use. The choices are 'elmer' or 'palace'. Both require manual install.
            This changes the format of ``simulator_params``.
        simulator_params: Simulator-specific params as a dictionary. See template files for more details.
            Has reasonable defaults.
        simulation_folder: Directory for storing the simulation results. Default is a temporary directory.
        **kwargs: Simulation settings propagated to inner :func:`~run_capacitive_simulation_elmer` or
            :func:`~run_capacitive_simulation_palace` implementation.
    """
    simulation_folder = Path(simulation_folder or get_sparameters_path())
    component = gf.get_component(component)

    simulation_folder = (
        simulation_folder / component.function_name
        if hasattr(component, "function_name")
        else simulation_folder
    )
    simulation_folder.mkdir(exist_ok=True, parents=True)

    match simulator:
        # case "elmer":
        #     return run_scattering_simulation_elmer(
        #         component,
        #         simulation_folder=simulation_folder,
        #         simulator_params=simulator_params,
        #         **kwargs,
        #     )
        case "palace":
            return run_scattering_simulation_palace(
                component,
                simulation_folder=simulation_folder,
                simulator_params=simulator_params,
                **kwargs,
            )
        case _:
            raise UserWarning(f"{simulator=!r} not implemented!")

    # TODO do we need to infer path or be explicit?
    # component_hash = get_component_hash(component)
    # kwargs_hash = get_kwargs_hash(**kwargs)
    # simulation_hash = hashlib.md5((component_hash + kwargs_hash).encode()).hexdigest()

    # return dirpath / f"{component.name}_{simulation_hash}.npz"


# get_scattering_elmer = partial(get_capacitance, tool="elmer")
get_scattering_palace = partial(get_capacitance, tool="palace")


if __name__ == "__main__":
    # TODO example
    c = gf.components.mmi1x2()
    # p = get_sparameters_path_lumerical(c)
    # sp = np.load(p)
    # spd = dict(sp)
    # print(spd)
