import gdsfactory as gf

c = gf.components.mzi_lattice_mmi(coupler_widths=[None, None], coupler_widths_tapers=[1.0, 1.0], coupler_lengths_tapers=[10.0, 10.0], coupler_lengths_mmis=[5.5, 5.5], coupler_widths_mmis=[2.5, 2.5], coupler_gaps_mmis=[0.25, 0.25], taper_functions_mmis=[{'function': 'taper'}, {'function': 'taper'}], straight_functions_mmis=[{'function': 'straight'}, {'function': 'straight'}], cross_sections_mmis=['strip', 'strip'], delta_lengths=[10.0])
c.plot()