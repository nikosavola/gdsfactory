import gdsfactory as gf

c = gf.components.ring_single_bend_coupler(radius=5.0, gap=0.2, coupling_angle_coverage=180.0, length_y=0.6, cross_section_inner='strip', cross_section_outer='strip')
c.plot()