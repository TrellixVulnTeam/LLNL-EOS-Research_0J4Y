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

        self.bulk_data = list(map(Var(np.loadtxt(f'Elements/{self.name}/Bulk_Test_{self.name}.dat')), np.power(10, Var.domain)))
        self.bulk_prime_data = list(map(Var(np.loadtxt(f'Elements/{self.name}/BulkPrime_Test_{self.name}.dat')), np.power(10, Var.domain)))
        self.pressure_data = list(map(Var(np.loadtxt(f'Elements/{self.name}/{self.name}.purgv157_rho_Pcgs.dat.fix')), np.power(10, Var.domain)))

    def __getitem__(self, var):
        if var == sympy.symbols('B'):
            return self.bulk_data
        elif var == sympy.symbols('Bp'):
            return self.bulk_prime_data
        elif var == sympy.symbols('P'):
            return self.pressure_data
        elif var == sympy.symbols('rho'):
            return Element.density_data
        elif var == sympy.symbols('r'):
            return self.atomic_radius_data
        elif var == sympy.symbols('m'):
            return self.mass_data
        elif var == sympy.symbols('h'):
            return Element.planck_data
        elif var == sympy.symbols('z'):
            return self.z_data
        else:
            return None
