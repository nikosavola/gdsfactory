from __future__ import annotations

import shutil
import subprocess
from math import inf
from pathlib import Path
from typing import Dict, Any, List, Sequence, Optional

from numpy import isfinite
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel

import gdsfactory as gf
from gdsfactory.components import interdigital_capacitor_enclosed
from gdsfactory.generic_tech import LAYER_STACK
from gdsfactory.technology import LayerStack
from gdsfactory.typings import MaterialSpec

ELECTROSTATIC_SIF = "electrostatic.sif"
ELECTROSTATIC_TEMPLATE = Path(__file__).parent / f"{ELECTROSTATIC_SIF}.j2"


class ElectrostaticResults(BaseModel):
    """Results class for electrostatic simulations."""

    capacitance_matrix: List[List[float]]
    mesh_location: Path
    field_file_location: Optional[Path] = None


def _generate_sif(
    simulation_folder: Path,
    name: str,
    signals: Sequence[str],
    bodies: Dict[str, Dict[str, Any]],
    layer_stack: LayerStack,
    material_spec: MaterialSpec,
    element_order: int,
):
    """Generates a sif file for Elmer simulations using Jinja2."""
    used_materials = {v.material for _, v in layer_stack.layers.items()}
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
    with open(simulation_folder / f"{name}_ElmerGrid.log", encoding="utf-8") as fp:
        subprocess.run(
            [elmergrid, "14", "2", f"{name}.msh"],
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
                    f"{name}/",
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
            "ElmerSolver not found. Make sure it is available in your PATH."
        )
    sif_file = simulation_folder / f"{name}.sif"
    with open(simulation_folder / f"{name}_ElmerSolver.log", encoding="utf-8") as fp:
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
    simulation_folder: Path, mesh_filename: str
) -> ElectrostaticResults:
    """Fetch results from successful Elmer simulations."""
    # todo pandas or numpy ?
    return ElectrostaticResults(
        capacitance_matrix=[[1e-15]], mesh_location=simulation_folder / mesh_filename
    )  # debug


def run_capacitive_simulation(
    component: gf.Component,
    element_order: int = 1,
    n_processes: int = 1,
    layer_stack: Optional[LayerStack] = None,
    material_spec: Optional[MaterialSpec] = None,
    simulation_folder: Optional[Path | str] = None,
    mesh_parameters: Optional[Dict[str, Any]] = None,
) -> ElectrostaticResults:
    """Run electrostatic finite element method simulations using
    `Elmer`_.
    Returns the field solution and resulting capacitance matrix.

    Args:
        component: Simulation envitonment as a gdsfactory component.
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
        mesh_parameyers:
            Keyword arguments to provide to :func:`~Component.to_gmsh`.

    .. _Elmer https://github.com/ElmerCSC/elmerfem
    """

    if layer_stack is None:
        layer_stack = LayerStack(
            layers={
                k: LAYER_STACK.layers[k]
                for k in (
                    "core",  # metal
                    # "ground",
                    "substrate",
                    "box",
                )
            }
        )
    if material_spec is None:
        material_spec: MaterialSpec = {
            "si": {"relative_permittivity": 11.45},
            "sio2": {"relative_permittivity": 1},  # vacuum
            "vacuum": {"relative_permittivity": 1},  # vacuum
        }

    # TODO use TemporaryDirectory if simulation_folder not specified
    simulation_folder = Path(simulation_folder or (Path(__file__).parent / "temp"))
    simulation_folder.mkdir(exist_ok=True, parents=True)

    filename = component.name + ".msh"
    component.to_gmsh(
        type="3D",
        filename=simulation_folder / filename,
        layer_stack=layer_stack,
        **(mesh_parameters or {}),
    )
    # TODO read Physical groups from the mesh or netlist

    # Signals are converted to Elmer Boundary Conditions
    signals = ["core"]  # todo infer from mesh
    # non-signals are converted to Elmer Bodies
    # TODO infer also
    # metal without signal (port) should be grounded
    bodies = {"vacuum": {"material": "vacuum"}, "substrate": {"material": "si"}}

    _generate_sif(
        simulation_folder,
        component.name,
        signals,
        bodies,
        layer_stack,
        material_spec,
        element_order,
    )
    # _elmergrid(simulation_folder, filename, n_processes)
    # _elmersolver(simulation_folder, filename, n_processes)
    return _read_elmer_results(simulation_folder, filename)


if __name__ == "__main__":
    from gdsfactory.technology.layer_stack import LayerLevel
    from gdsfactory.generic_tech import LAYER

    # Example LayerStack values from doi:10.1103/PRXQuantum.4.010314 and
    layer_stack = LayerStack(
        layers=dict(
            substrate=LayerLevel(
                layer=LAYER.WAFER,
                thickness=500e-6,
                zmin=0,
                material="Si",
                mesh_order=99,
            ),
            vacuum=LayerLevel(
                layer=LAYER.WAFER,
                thickness=1000e-6,
                zmin=500e-6,
                material="vacuum",
                mesh_order=99,
            ),
            metal=LayerLevel(
                layer=LAYER.WG,
                thickness=200e-9,
                zmin=500e-6,
                material="Nb",
                mesh_order=2,
                width_to_z=0.5,
            ),
            # TODO vacuum in between core
        )
    )

    # Test mesh capacitor
    c = interdigital_capacitor_enclosed()

    # mesh.get_cells_type("triangle")
    # mesh = from_meshio(mesh)
    # mesh.draw().plot()
    # mesh
    results = run_capacitive_simulation(c)
    print(results)
