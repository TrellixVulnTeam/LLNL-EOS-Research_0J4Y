
import numpy as np
import mendeleev
import math

from Core.Var import Var
from Core.Utils import Utils

mendeleev.H.isotopes[1].vdw_radius = 50.0
mendeleev.H.isotopes[2].vdw_radius = 40.0
mendeleev.H.isotopes[2].atomic_mass = 3.016


class Element:
    planck = 6.62607004e-34
    amu = 1.66054e-27
    H_mass = mendeleev.H.mass
    TF_data = np.asarray(np.loadtxt('Pressure Data/TF_Hydrogen_Pccgs.dat'), dtype=float)
    orbitals = [(1, 's'), (2, 's'), (2, 'p'), (3, 's'), (3, 'p'), (4, 's'), (3, 'd'), (4, 'p'), (5, 's'), (4, 'd'),
                (5, 'p'), (6, 's'), (4, 'f'), (5, 'd'), (6, 'p'), (7, 's')]
    orbital_dict = {orbital: [] for orbital in orbitals}

    def __init__(self, element_name):
        self.name = element_name
        if element_name == 'D':
            self.element = mendeleev.H.isotopes[1]
            self.element.vdw_radius = 50.0
            self.element.ec = mendeleev.H.ec
        elif element_name == 'T':
            self.element = mendeleev.H.isotopes[2]
            self.element.vdw_radius = 40.0
            self.element.mass = 3.016
            self.element.ec = mendeleev.H.ec
        else:
            self.element = getattr(mendeleev, element_name)
        self.atomic_radius, self.mass = self.element.vdw_radius, self.element.mass * Element.amu

        self.atomic_radius_data = [self.atomic_radius] * len(Var.domain)
        self.mass_data = [self.mass] * len(Var.domain)
        self.z_data = [self.element.atomic_number] * len(Var.domain)
        self.electron_configuration = self.element.ec
        for orbital in reversed(Element.orbitals):
            if orbital in self.electron_configuration._conf:
                self.orbital = orbital
                break
        Element.orbital_dict[self.orbital].append((self.name, self.element.atomic_number))
        Element.orbital_dict[self.orbital] = list(sorted(Element.orbital_dict[self.orbital], key=lambda arg: arg[1]))

        self.pressure_data = np.asarray(list(map(Var(np.loadtxt(f'Pressure Data/Purg Data/{self.name}.purgv157_rho_Pccgs.dat.fix')),
                                                 Var.domain)), dtype=float)
        # self.pressure_data = np.exp(Utils.SMA_smooth(np.log(self.pressure_data)))
        try:
            x_scale = Element.H_mass / (self.element.mass * self.element.atomic_number)
            y_scale = self.element.atomic_number ** (10 / 3)
            lower, upper = np.searchsorted(Element.TF_data[:, 0],
                                           (x_scale * 10 ** Var.domain[0], x_scale * 10 ** Var.domain[-1]), side='right')
            self.tf_data = Element.TF_data[lower:upper]
            self.tf_data = list(map(Var(np.vstack((self.tf_data[:, 0] / x_scale, self.tf_data[:, 1] * y_scale)).T), Var.domain))
            # self.tf_data = np.exp(Utils.SMA_smooth(np.log(self.tf_data)))
        except OSError:
            raise AssertionError('No TF data available')


