import gdsfactory as gf

c = gf.components.spiral_racetrack(min_radius=5, straight_length=10.0, spacings=[2, 2, 3, 3, 2, 2], cross_section='strip')
c.plot()