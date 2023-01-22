import gdsfactory as gf

c = gf.components.greek_cross(cross_struct_length=30.0, cross_struct_width=1.0, cross_struct_layers=['WG'], cross_implant_length=30.0, cross_implant_width=2.0, cross_implant_layers=['N'], contact_layers=['WG', 'NPP'], contact_offset=10, contact_buffer=10, pad_width=50)
c.plot()