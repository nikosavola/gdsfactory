from __future__ import annotations

import itertools
import shutil
import subprocess
from math import inf
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import gmsh
from jinja2 import Environment, FileSystemLoader
from numpy import isfinite
from pandas import read_csv

import gdsfactory as gf
from gdsfactory.components import interdigital_capacitor_enclosed
from gdsfactory.generic_tech import LAYER_STACK
from gdsfactory.technology import LayerStack
from gdsfactory.typings import ElectrostaticResults, MaterialSpec

ELECTROSTATIC_SIF = "electrostatic.sif"
ELECTROSTATIC_TEMPLATE = Path(__file__).parent / f"{ELECTROSTATIC_SIF}.j2"


def _generate_sif(
    simulation_folder: Path,
    name: str,
    signals: Sequence[Sequence[str]],
    bodies: Dict[str, Dict[str, Any]],
    ground_layers: Iterable[str],
    layer_stack: LayerStack,
    material_spec: MaterialSpec,
    element_order: int,
    background_tag: Optional[str] = None,
    simulator_params: Optional[Mapping[str, Any]] = None,
):
    # pylint: disable=unused-argument
    """Generates a sif file for Elmer simulations using Jinja2."""
    used_materials = {v.material for v in layer_stack.layers.values()} | (
        {background_tag} if background_tag else {}
    )
    used_materials = {
        k: material_spec[k]
        for k in used_materials
        if isfinite(material_spec[k].get("relative_permittivity", inf))
    }

    sif_template = Environment(
        loader=FileSystemLoader(ELECTROSTATIC_TEMPLATE.parent)
    ).get_template(ELECTROSTATIC_TEMPLATE.name)
    output = sif_template.render(**locals())
    with open(simulation_folder / f"{name}.sif", "w", encoding="utf-8") as fp:
        fp.write(output)


def _elmergrid(simulation_folder: Path, name: str, n_processes: int = 1):
    """Run ElmerGrid for converting gmsh mesh to Elmer format."""
    elmergrid = shutil.which("ElmerGrid")
    if elmergrid is None:
        raise RuntimeError(
            "ElmerGrid not found. Make sure it is available in your PATH."
        )
    with open(simulation_folder / f"{name}_ElmerGrid.log", "w", encoding="utf-8") as fp:
        subprocess.run(
            [elmergrid, "14", "2", name],
            cwd=simulation_folder,
            shell=False,
            stdout=fp,
            stderr=fp,
            check=True,
        )
        if n_processes > 1:
            subprocess.run(
                [
                    elmergrid,
                    "2",
                    "2",
                    f"{Path(name).stem}/",
                    "-metiskway",
                    str(n_processes),
                    "4",
                    "-removeunused",
                ],
                cwd=simulation_folder,
                shell=False,
                stdout=fp,
                stderr=fp,
                check=True,
            )


def _elmersolver(simulation_folder: Path, name: str, n_processes: int = 1):
    """Run simulations with ElmerFEM."""
    elmersolver = (
        shutil.which("ElmerSolver")
        if (no_mpi := n_processes == 1)
        else shutil.which("ElmerSolver_mpi")
    )
    if elmersolver is None:
        raise RuntimeError(
            ("ElmerSolver" if n_processes == 1 else "ElmerSolver_mpi")
            + " not found. Make sure it is available in your PATH."
        )
    sif_file = str(simulation_folder / f"{Path(name).stem}.sif")
    with open(
        simulation_folder / f"{name}_ElmerSolver.log", "w", encoding="utf-8"
    ) as fp:
        subprocess.run(
            [elmersolver, sif_file]
            if no_mpi
            else ["mpiexec", "-np", str(n_processes), elmersolver, sif_file],
            cwd=simulation_folder,
            shell=False,
            stdout=fp,
            stderr=fp,
            check=True,
        )


def _read_elmer_results(
    simulation_folder: Path,
    mesh_filename: str,
    n_processes: int,
    ports: Iterable[str],
    is_temporary: bool,
) -> ElectrostaticResults:
    """Fetch results from successful Elmer simulations."""
    raw_name = Path(mesh_filename).stem
    raw_capacitance_matrix = read_csv(
        simulation_folder / f"{raw_name}_capacitance.dat",
        sep=r"\s+",
        header=None,
        dtype=float,
    ).values
    return ElectrostaticResults(
        capacitance_matrix={
            (iname, jname): raw_capacitance_matrix[i][j]
            for (i, iname), (j, jname) in itertools.product(
                enumerate(ports), enumerate(ports)
            )
        },
        **(
            {}
            if is_temporary
            else dict(
                mesh_location=simulation_folder / mesh_filename,
                field_file_location=simulation_folder
                / raw_name
                / f'{raw_name}_t0001.{"pvtu" if n_processes > 1 else "vtu"}',
            )
        ),
    )


def run_capacitive_simulation_elmer(
    component: gf.Component,
    element_order: int = 1,
    n_processes: int = 1,
    layer_stack: Optional[LayerStack] = None,
    material_spec: Optional[MaterialSpec] = None,
    simulation_folder: Optional[Path | str] = None,
    simulator_params: Optional[Mapping[str, Any]] = None,
    mesh_parameters: Optional[Dict[str, Any]] = None,
    mesh_file: Optional[Path | str] = None,
) -> ElectrostaticResults:
    """Run electrostatic finite element method simulations using
    `Elmer`_.     Returns the field solution and resulting capacitance matrix.

    .. note:: You should have `ElmerGrid`, `ElmerSolver` and `ElmerSolver_mpi` and in your PATH.

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
        simulator_params: Elmer-specific parameters. See template file for more details.
        mesh_parameters:
            Keyword arguments to provide to :func:`~Component.to_gmsh`.
        mesh_file: Path to a ready mesh to use. Useful for reusing one mesh file.
            By default a mesh is generated according to ``mesh_parameters``.

    .. _Elmer https://github.com/ElmerCSC/elmerfem
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
            **(mesh_parameters or {}),
        )

    gmsh.initialize(interruptible=False)
    gmsh.merge(str(simulation_folder / filename))
    mesh_surface_entities = [
        gmsh.model.getPhysicalName(*dimtag)
        for dimtag in gmsh.model.getPhysicalGroups(dim=2)
    ]
    gmsh.finalize()

    # Signals are converted to Elmer Boundary Conditions
    ground_layers = {
        next(k for k, v in layer_stack.layers.items() if v.layer == port.layer)
        for port in component.get_ports()
    }  # ports allowed only on metal
    # TODO infer port delimiter from somewhere
    # TODO raise error for port delimiters not supported by Elmer MATC or find how to escape
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

    # dielectrics
    bodies = {
        k: {"material": v.material}
        for k, v in layer_stack.layers.items()
        if port_delimiter not in k and k not in ground_layers
    }
    if background_tag := (mesh_parameters or {}).get("background_tag", "vacuum"):
        bodies = {**bodies, background_tag: {"material": background_tag}}

    _generate_sif(
        simulation_folder,
        component.name,
        metal_signal_surfaces_grouped,
        bodies,
        ground_layers,
        layer_stack,
        material_spec,
        element_order,
        background_tag,
        simulator_params,
    )
    _elmergrid(simulation_folder, filename, n_processes)
    _elmersolver(simulation_folder, filename, n_processes)
    results = _read_elmer_results(
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
        "Si": {"relative_permittivity": 11.45},
        "Nb": {"relative_permittivity": inf},
        "vacuum": {"relative_permittivity": 1},
    }

    # Test capacitor
    simulation_box = [[-200, -200], [200, 200]]
    c = gf.Component("capacitance_elmer")
    cap = c << interdigital_capacitor_enclosed(
        metal_layer=LAYER.WG, gap_layer=LAYER.DEEPTRENCH, enclosure_box=simulation_box
    )
    c.add_ports(cap.ports)
    substrate = gf.components.bbox(bbox=simulation_box, layer=LAYER.WAFER)
    c << substrate
    c.flatten()
    results = run_capacitive_simulation_elmer(
        c,
        layer_stack=layer_stack,
        material_spec=material_spec,
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
                    f"bw{delimiter}{port}": {
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
        field.slice_orthogonal().plot(scalars="electric field", cmap="turbo")
