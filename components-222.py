import gdsfactory as gf

c = gf.components.via_stack_slab_m3(size=[11.0, 11.0], layers=['SLAB90', 'M1', 'M2', 'M3'])
c.plot()