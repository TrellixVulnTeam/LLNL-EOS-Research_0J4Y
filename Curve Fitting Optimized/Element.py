import numpy as np
import mendeleev
import sympy

from Var import Var

mendeleev.H.isotopes[1].vdw_radius = 50.0
mendeleev.H.isotopes[2].vdw_radius = 40.0
mendeleev.H.isotopes[2].atomic_mass = 3.016


class Element:
    planck = 6.62607004e-34
    amu = 1.66054e-27
    density_data = list(np.power(10, Var.domain))
    planck_data = [planck] * len(Var.domain)

    def __init__(self, element_name):
        self.name = element_name
        if element_name == 'D':
            self.element = mendeleev.H.isotopes[1]
            self.element.vdw_radius = 50.0
        elif element_name == 'T':
            self.element = mendeleev.H.isotopes[2]
            self.element.vdw_radius = 40.0
            self.element.mass = 3.016
        else:
            self.element = getattr(mendeleev, element_name)
        self.atomic_radius, self.mass = self.element.vdw_radius, self.element.mass * Element.amu

        self.atomic_radius_data = [self.atomic_radius] * len(Var.domain)
        self.mass_data = [self.mass] * len(Var.domain)
        self.z_data = [self.element.atomic_number] * len(Var.domain)

        self.pressure_data = list(map(Var(np.loadtxt(f'Purg Data/{self.name}.purgv157_rho_Pccgs.dat.fix')), np.power(10, Var.domain)))
        assert len(self.pressure_data) > 10, 'No data read'
        try:
            self.tf_data = list(map(Var(np.loadtxt(f'TF Data/{self.name}_TFCC_rhoPccgs.dat')), np.power(10, Var.domain)))
        except OSError:
            raise AssertionError('No TF data available')


