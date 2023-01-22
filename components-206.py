import gdsfactory as gf

c = gf.components.terminator(length=50, cross_section_input='strip', tapered_width=0.2, doping_layers=['NPP'])
c.plot()