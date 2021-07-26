
import mendeleev
import numpy as np
from matplotlib import pyplot
from Engine import Engine

if __name__ == '__main__':
    element_list = ['H', 'Fe', 'Sn', 'Ta', 'Pb']
    data = {element: np.loadtxt(f'TF Data/{element}_TFCC_rhoPccgs.dat') for element in element_list}
    x_list = np.asarray([data['H'][i][0] for i in range(0, len(data['H']), 10)], dtype=float)
    x_list = np.log10(x_list)
    reversed_x_list = np.asarray(list(reversed(x_list)), dtype=float)
    y_dict = {element: np.asarray([data[element][i][1] for i in range(0, len(data[element]), 10)], dtype=float) for element in element_list}

    atomic_mass_dict = {element: getattr(mendeleev, element).mass for element in element_list}
    atomic_number_dict = {element: getattr(mendeleev, element).atomic_number for element in element_list}

    atomic_masses = np.asarray(list(atomic_mass_dict.values()), dtype=float)
    atomic_numbers = np.asarray(list(atomic_number_dict.values()), dtype=float)

    shifted_x_dict = {element: np.log10(atomic_mass_dict[element] * atomic_number_dict[element]) - reversed_x_list for element in element_list}
    shifted_y_dict = {element: np.asarray(list(reversed(y_dict[element])), dtype=float) for element in element_list}

    print({element: (shifted_x_dict[element][0], shifted_x_dict[element][-1]) for element in element_list})
    intersection = (max([domain[0] for domain in shifted_x_dict.values()]),
                    min([domain[-1] for domain in shifted_x_dict.values()]))
    print(intersection)


