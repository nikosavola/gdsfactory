import gdsfactory as gf

c = gf.components.via_stack_heater_m3(size=[11.0, 11.0], layers=['HEATER', 'M2', 'M3'])
c.plot()