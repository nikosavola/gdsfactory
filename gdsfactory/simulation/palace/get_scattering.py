from __future__ import annotations

import itertools
import json
import shutil
import subprocess
from math import inf
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Mapping

import gmsh
from numpy import isfinite
from pandas import read_csv

import gdsfactory as gf
from gdsfactory.components import interdigital_capacitor_enclosed
from gdsfactory.generic_tech import LAYER_STACK
from gdsfactory.technology import LayerStack
from gdsfactory.typings import DrivenFullWaveResults, MaterialSpec

DRIVE_JSON = "driven.json"
DRIVEN_TEMPLATE = Path(__file__).parent / DRIVE_JSON


def _generate_json(
    simulation_folder: Path,
    name: str,
    edge_signals: Sequence[Sequence[str]],
    internal_signals: Sequence[Sequence[str]],
    bodies: Dict[str, Dict[str, Any]],
    absorbing_surfaces: Sequence[str],
    layer_stack: LayerStack,
    material_spec: MaterialSpec,
    element_order: int,
    physical_name_to_dimtag_map: Dict[str, Tuple[int, int]],
    metal_surfaces: Sequence[str],
    background_tag: Optional[str] = None,
    simulator_params: Optional[Mapping[str, Any]] = None,
    driven_settings: Optional[Mapping[str, float | int | bool]] = None,
):
    """Generates a json file for full-wave Palace simulations."""
    # TODO: Generalise to merger with the Elmer implementations"""
    used_materials = {v.material for v in layer_stack.layers.values()} | (
        {background_tag} if background_tag else {}
    )
    used_materials = {
        k: material_spec[k]
        for k in used_materials
        if isfinite(material_spec[k].get("relative_permittivity", inf))
    }

    with open(DRIVEN_TEMPLATE, "r") as fp:
        palace_json_data = json.load(fp)

    material_to_attributes_map = {
        v["material"]: physical_name_to_dimtag_map[k][1] for k, v in bodies.items()
    }

    palace_json_data["Model"]["Mesh"] = f"{name}.msh"
    palace_json_data["Domains"]["Materials"] = [
        {
            "Attributes": [material_to_attributes_map.get(material, None)],
            "Permittivity": props["relative_permittivity"],
            "Permeability": props["relative_permeability"],
            "LossTan": props.get("loss_tangent", 0.),
            "Conductivity": props.get("conductivity", 0.),
        }
        for material, props in used_materials.items()
    ]
    #TODO list here attributes that contained LossTAN
    # palace_json_data["Domains"]["Postprocessing"]["Dielectric"] = [

    # ]

    # TODO 3d volumes as pec???, not needed for capacitance
    palace_json_data["Boundaries"]["PEC"] = {
        "Attributes": [physical_name_to_dimtag_map[layer][1] for layer in metal_surfaces] # TODO
    }
    port_i = 1
    if edge_signals:
        palace_json_data["Boundaries"]["WavePort"] = [
        {
            "Index": (port_i := port_i + 1),
            "Attributes": [
                physical_name_to_dimtag_map[signal][1] for signal in signal_group
            ],
            "Mode": 1,
            "Offset": 0.0,
            "Excitation": True
        } for signal_group in edge_signals
        ]
    if internal_signals:
        palace_json_data["Boundaries"]["LumpedPort"] = [
        {
            "Index": (port_i := port_i + 1),
            "Attributes": [
                physical_name_to_dimtag_map[signal][1] for signal in signal_group
            ],
            "Excitation": True,
            "R": 50,
        } for signal_group in internal_signals
        ]
    # Farfield surface
    palace_json_data["Boundaries"]["Absorbing"] = {
      "Attributes": [
                physical_name_to_dimtag_map[e][1] for e in absorbing_surfaces
            ], # TODO get farfield _None etc
      "Order": 1
    }
    # palace_json_data["Boundaries"]["Postprocessing"]["Dielectric"] =       [
    #     {
    #       "Index": 1,
    #       "Attributes": [3], # these two same as above if losstan
    #       "Side": "+Z",
    #       "Thickness": 2e-3, # need metadata for oxide layer thickness
    #       "PermittivitySA": 4.0,
    #       "LossTan": 1.0
    #     }
    #   ]

    palace_json_data["Solver"]["Order"] = element_order
    if driven_settings is not None:
        palace_json_data["Solver"]["Driven"] |= driven_settings
    if simulator_params is not None:
        palace_json_data["Solver"]["Linear"] |= simulator_params

    with open(simulation_folder / f"{name}.json", "w", encoding="utf-8") as fp:
        json.dump(palace_json_data, fp, indent=4)


def _palace(simulation_folder: Path, name: str, n_processes: int = 1):
    """Run simulations with Palace."""
    palace = shutil.which("palace")
    if palace is None:
        raise RuntimeError("palace not found. Make sure it is available in your PATH.")
    json_file = str(simulation_folder / f"{Path(name).stem}.json")
    with open(simulation_folder / f"{name}_palace.log", "w", encoding="utf-8") as fp:
        subprocess.run(
            [palace, json_file]
            if n_processes == 1
            else [palace, "-np", str(n_processes), json_file],
            cwd=simulation_folder,
            shell=False,
            stdout=fp,
            stderr=fp,
            check=True,
        )


def _read_palace_results(
    simulation_folder: Path,
    mesh_filename: str,
    n_processes: int,
    ports: Iterable[str],
    is_temporary: bool,
) -> DrivenFullWaveResults:
    """Fetch results from successful Palace simulations."""
    scattering_matrix = read_csv(
        simulation_folder / "postpro" / "TODO.csv", dtype=float
    )
    return DrivenFullWaveResults(
        scattering_matrix=scattering_matrix, # TODO convert to SDict from DataFrame
        **(
            {}
            if is_temporary
            else dict(
                mesh_location=simulation_folder / mesh_filename,
                field_file_location=simulation_folder
                / "postpro"
                / "paraview"
                / "driven"  # TODO
                / "driven.pvd",
            )
        ),
    )


def run_scattering_simulation_palace(
    component: gf.Component,
    element_order: int = 1,
    n_processes: int = 1,
    layer_stack: Optional[LayerStack] = None,
    material_spec: Optional[MaterialSpec] = None,
    simulation_folder: Optional[Path | str] = None,
    simulator_params: Optional[Mapping[str, Any]] = None,
    driven_settings: Optional[Mapping[str, float | int | bool]] = None,
    mesh_parameters: Optional[Dict[str, Any]] = None,
    mesh_file: Optional[Path | str] = None,
) -> DrivenFullWaveResults:
    """Run full-wave finite element method simulations using
    `Palace`_.
    Returns the field solution and resulting scattering matrix.

    .. note:: You should have `palace` in your PATH.

    Args:
        component: Simulation environment as a gdsfactory component.
        element_order:
            Order of polynomial basis functions.
            Higher is more accurate but takes more memory and time to run.
        n_processes: Number of processes to use for parallelization
        layer_stack:
            :class:`~LayerStack` defining defining what layers to include in the simulation
            and the material properties and thicknesses.
        material_spec:
            :class:`~MaterialSpec` defining material parameters for the ones used in ``layer_stack``.
        simulation_folder:
            Directory for storing the simulation results.
            Default is a temporary directory.
        simulator_params: Palace-specific parameters. This will be expanded to ``solver["Linear"]`` in
            the Palace config, see `Palace documentation <https://awslabs.github.io/palace/stable/config/solver/#solver[%22Linear%22]>`_
        driven_settings: Driven full-wave parameters in Palace. This will be expanded to ``solver["Driven"]`` in
            the Palace config, see `Palace documentation <https://awslabs.github.io/palace/stable/config/solver/#solver[%22Driven%22]>`_
        mesh_parameters:
            Keyword arguments to provide to :func:`~Component.to_gmsh`.
        mesh_file: Path to a ready mesh to use. Useful for reusing one mesh file.
            By default a mesh is generated according to ``mesh_parameters``.

    .. _Palace https://github.com/awslabs/palace
    """

    if layer_stack is None:
        layer_stack = LayerStack(
            layers={
                k: LAYER_STACK.layers[k]
                for k in (
                    "core",
                    "substrate",
                    "box",
                )
            }
        )
    if material_spec is None:
        material_spec: MaterialSpec = {
            "si": {"relative_permittivity": 11.45},
            "sio2": {"relative_permittivity": 1},
            "vacuum": {"relative_permittivity": 1},
        }

    temp_dir = TemporaryDirectory()
    simulation_folder = Path(simulation_folder or temp_dir.name)
    simulation_folder.mkdir(exist_ok=True, parents=True)

    filename = component.name + ".msh"
    if mesh_file:
        shutil.copyfile(str(mesh_file), str(simulation_folder / filename))
    else:
        component.to_gmsh(
            type="3D",
            filename=simulation_folder / filename,
            layer_stack=layer_stack,
            gmsh_version=2.2,  # see https://mfem.org/mesh-formats/#gmsh-mesh-formats
            **(mesh_parameters or {}),
        )

    # re-read the mesh
    gmsh.initialize(interruptible=False)
    gmsh.merge(str(simulation_folder / filename))
    mesh_surface_entities = {
        gmsh.model.getPhysicalName(*dimtag)
        for dimtag in gmsh.model.getPhysicalGroups(dim=2)
    }

    # Signals are converted to Boundaries
    ground_layers = {
        next(k for k, v in layer_stack.layers.items() if v.layer == port.layer)
        for port in component.get_ports()
    }  # ports allowed only on metal
    # TODO infer port delimiter from somewhere
    port_delimiter = "__"
    metal_surfaces = [
        e for e in mesh_surface_entities if any(ground in e for ground in ground_layers)
    ]
    # Group signal BCs by ports
    metal_signal_surfaces_grouped = [
        [e for e in metal_surfaces if port in e] for port in component.ports
    ]
    metal_ground_surfaces = set(metal_surfaces) - set(
        itertools.chain.from_iterable(metal_signal_surfaces_grouped)
    )

    ground_layers |= metal_ground_surfaces

    absorbing_surfaces = [] # TODO __NONE

    # dielectrics
    bodies = {
        k: {
            "material": v.material,
        }
        for k, v in layer_stack.layers.items()
        if port_delimiter not in k and k not in ground_layers
    }
    if background_tag := (mesh_parameters or {}).get("background_tag", "vacuum"):
        bodies = {**bodies, background_tag: {"material": background_tag}}

    # TODO refactor to not require this map, the same information could be transferred with the variables above
    physical_name_to_dimtag_map = {
        gmsh.model.getPhysicalName(*dimtag): dimtag
        for dimtag in gmsh.model.getPhysicalGroups()
    }
    gmsh.finalize()

    _generate_json(
        simulation_folder,
        component.name,
        metal_signal_surfaces_grouped, # edge
        metal_signal_surfaces_grouped, # internal
        bodies,
        absorbing_surfaces,
        layer_stack,
        material_spec,
        element_order,
        physical_name_to_dimtag_map,
        metal_surfaces,
        background_tag,
        simulator_params,
        driven_settings,
    )
    _palace(simulation_folder, filename, n_processes)
    results = _read_palace_results(
        simulation_folder,
        filename,
        n_processes,
        component.ports,
        is_temporary=str(simulation_folder) == temp_dir.name,
    )
    temp_dir.cleanup()
    return results


if __name__ == "__main__":
    import pyvista as pv

    from gdsfactory.generic_tech import LAYER
    from gdsfactory.technology.layer_stack import LayerLevel

    # Example LayerStack similar to doi:10.1103/PRXQuantum.4.010314
    layer_stack = LayerStack(
        layers=dict(
            substrate=LayerLevel(
                layer=LAYER.WAFER,
                thickness=500,
                zmin=0,
                material="Si",
                mesh_order=99,
            ),
            metal=LayerLevel(
                layer=LAYER.WG,
                thickness=200e-3,
                zmin=500,
                material="Nb",
                mesh_order=2,
            ),
        )
    )
    material_spec = {
        "Si": {"relative_permittivity": 11.45, "relative_permeability": 1},
        "Nb": {"relative_permittivity": inf},
        "vacuum": {"relative_permittivity": 1, "relative_permeability": 1},
    }

    # Test capacitor
    simulation_box = [[-200, -200], [200, 200]]
    c = gf.Component("capacitance_palace")
    cap = c << interdigital_capacitor_enclosed(
        metal_layer=LAYER.WG, gap_layer=LAYER.DEEPTRENCH, enclosure_box=simulation_box
    )
    c.add_ports(cap.ports)
    substrate = gf.components.bbox(bbox=simulation_box, layer=LAYER.WAFER)
    c << substrate
    c.flatten()

    results = run_scattering_simulation_palace(
        c,
        layer_stack=layer_stack,
        material_spec=material_spec,
        driven_settings={
            'MinFreq': 0.1,
            'MaxFreq': 5,
            'FreqStep': 2,
            # 'AdaptiveTol': 1e-5,
        },
        mesh_parameters=dict(
            background_tag="vacuum",
            background_padding=(0,) * 5 + (700,),
            portnames=c.ports,
            verbosity=1,
            default_characteristic_length=200,
            layer_portname_delimiter=(delimiter := "__"),
            resolutions={
                "bw": {
                    "resolution": 14,
                },
                "substrate": {
                    "resolution": 50,
                },
                "vacuum": {
                    "resolution": 120,
                },
                **{
                    f"bw{delimiter}{port}_vacuum": {
                        "resolution": 8,
                    }
                    for port in c.ports
                },
            },
        ),
    )
    print(results)

    if results.field_file_location:
        field = pv.read(results.field_file_location)
        field.slice_orthogonal().plot(scalars="E", cmap="turbo")
