from __future__ import annotations

import numpy as np
import pytest

from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.port import Port
from gdsfactory.routing.manhattan import RouteWarning, round_corners, route_manhattan

TOLERANCE = 0.001
DEG2RAD = np.pi / 180
RAD2DEG = 1 / DEG2RAD

O2D = {0: "East", 180: "West", 90: "North", 270: "South"}


def test_manhattan() -> None:
    top_cell = Component()
    layer = (1, 0)

    inputs = [
        Port("in1", center=(10, 5), width=0.5, orientation=90, layer=layer),
        # Port("in2",center= (-10, 20), width=0.5, 0),
        # Port("in3",center= (10, 30), width=0.5, 0),
        # Port("in4",center= (-10, -5), width=0.5, 90),
        # Port("in5",center= (0, 0), width=0.5, 0),
        # Port("in6",center= (0, 0), width=0.5, 0),
    ]

    outputs = [
        Port("in1", center=(290, -60), width=0.5, orientation=180, layer=layer),
        # Port("in2", (-100, 20), 0.5, 0),
        # Port("in3", (100, -25), 0.5, 0),
        # Port("in4", (-150, -65), 0.5, 270),
        # Port("in5", (25, 3), 0.5, 180),
        # Port("in6", (0, 10), 0.5, 0),
    ]

    lengths = [349.974]

    for input_port, output_port, length in zip(inputs, outputs, lengths):
        # input_port = Port("input_port", (10,5), 0.5, 90)
        # output_port = Port("output_port", (90,-60), 0.5, 180)
        # bend = bend_circular(radius=5.0)

        route = route_manhattan(
            input_port=input_port,
            output_port=output_port,
            radius=5.0,
            auto_widen=True,
            width_wide=2,
            layer=layer
            # width=0.2,
        )

        top_cell.add(route.references)
        assert np.isclose(route.length, length), route.length


def test_manhattan_pass() -> None:
    waypoints = [
        [10.0, 0.0],
        [20.0, 0.0],
        [20.0, 12.0],
        [120.0, 12.0],
        [120.0, 80.0],
        [110.0, 80.0],
    ]
    route = round_corners(waypoints, radius=5)
    c = Component()
    c.add(route.references)


def test_manhattan_fail() -> None:
    waypoints = [
        [10.0, 0.0],
        [20.0, 0.0],
        [20.0, 12.0],
        [120.0, 12.0],
        [120.0, 80.0],
        [110.0, 80.0],
    ]
    with pytest.warns(RouteWarning):
        route = round_corners(waypoints, radius=10.0, with_point_markers=False)
    c = Component()
    c.add(route.references)


@cell
def _demo_manhattan_fail() -> Component:
    waypoints = [
        [10.0, 0.0],
        [20.0, 0.0],
        [20.0, 12.0],
        [120.0, 12.0],
        [120.0, 80.0],
        [110.0, 80.0],
    ]
    route = round_corners(waypoints, radius=10.0, with_point_markers=False)
    c = Component()
    c.add(route.references)
    return c


if __name__ == "__main__":
    # test_manhattan_fail()
    test_manhattan_pass()
