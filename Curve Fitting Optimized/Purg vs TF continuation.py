import math
import numpy as np
from Core.Var import Var
import matplotlib
from matplotlib import pyplot
from Core.Engine import Engine
from Core.Element import Element
from Core.Utils import Utils

if __name__ == '__main__':
    matplotlib.rcParams['lines.linewidth'] = 0.7
    e = Engine()
    print(Element.orbital_dict)
    title = {'fontname': 'Times New Roman', 'size': 16}
    label = {'fontname': 'Times New Roman', 'size': 10}


    def set_labels(title_label, x_label, y_label, n=None):
        if n is not None:
            pyplot.figure(n)
        pyplot.title(title_label, **title)
        pyplot.xlabel(x_label, **label)
        pyplot.ylabel(y_label, **label)


    def color(m):
        k = 2 * np.pi * m / 200
        return (1 + np.asarray([np.sin(k), np.sin(k + 2 * np.pi / 3), np.sin(k + 4 * np.pi / 3)], dtype=float)) / 2

    # element_list = Engine.element_range('H', 'Se')
    # element_list = Engine.complete_list[:]
    element_list = ['H'] + ['Al', 'S', 'Cl', 'Ar']
    # ['N', 'O', 'F', 'Ne']
    # ['Al', 'S', 'Cl', 'Ar']
    # ['Ga', 'Ge', 'As', 'Se']
    '''element_list = []
    orbital_list = [(4, 'p'), (5, 'p'), (6, 'p')]
    for orbital in orbital_list:
        element_list = element_list + [arg[0] for arg in Element.orbital_dict[orbital]]
    if 'H' not in element_list:
        element_list = ['H'] + element_list'''

    purg_color_dict, tf_color_dict = dict(), dict()

    x_list = Var.domain
    reversed_x = np.asarray(list(reversed(x_list)), dtype=float)
    x_list = -reversed_x

    dx = (x_list[-1] - x_list[0]) / (len(x_list) - 1)

    x_dict = dict()
    purg_y_dict, tf_y_dict = dict(), dict()
    residual_dict = dict()
    # purg_dv_dict, tf_dv_dict = dict(), dict()
    residual_dv_dict = dict()

    atomic_mass_dict, atomic_number_dict = dict(), dict()
    orbital_dict = dict()

    for element in element_list:
        library_element = Engine.library[element]

        m, z = library_element.element.mass, library_element.element.atomic_number
        atomic_mass_dict[element], atomic_number_dict[element] = m, z
        orbital_dict[element] = library_element.orbital

        purg_color = color(m)
        tf_color = 0.7 * purg_color

        shifted_x = np.log10(m * z) - reversed_x
        purg_y = np.asarray(list(reversed(library_element.pressure_data)), dtype=float)
        tf_y = np.asarray(list(reversed(library_element.tf_data)), dtype=float)

        x_dict[element], purg_y_dict[element], tf_y_dict[element] = shifted_x, purg_y, tf_y
        # purg_dv_dict[element], tf_dv_dict[element] = purg_derivative, tf_derivative

        purg_color_dict[element], tf_color_dict[element] = purg_color, tf_color

        # residual_y = tf_y - purg_y
        residual_y = 1 - np.power(10, purg_y - tf_y)
        residual_dict[element] = residual_y
        residual_derivative = Utils.SMA_smooth(
            np.asarray([residual_y[i + 1] - residual_y[i] for i in range(len(x_list) - 1)], dtype=float) / dx,
            buffer=70)
        # residual_derivative = np.asarray([residual_y[i + 1] - residual_y[i] for i in range(len(x_list) - 1)], dtype=float) / dx
        residual_dv_dict[element] = residual_derivative

    atomic_numbers = np.asarray(list(atomic_number_dict.values()), dtype=float)
    atomic_masses = np.asarray(list(atomic_mass_dict.values()), dtype=float)

    figure1, figure2 = pyplot.figure(1), pyplot.figure(2)

    residual_ldv_dict = dict()

    for element in element_list:
        m, z = atomic_mass_dict[element], atomic_number_dict[element]
        y_list = residual_dict[element]
        dv_list = residual_dv_dict[element]

        pyplot.figure(1)
        pyplot.plot(x_list, y_list, color=purg_color_dict[element], label=element)
        k = len(x_list) - len(dv_list)
        lower, upper = math.floor(k / 2), -math.ceil(k / 2)
        pyplot.plot(x_list[lower:upper], dv_list, color=tf_color_dict[element])
        pyplot.legend(bbox_to_anchor=(1, 1), loc='upper left')

        ldv_list = dv_list / y_list[lower:upper]
        residual_ldv_dict[element] = ldv_list

        pyplot.figure(2)
        pyplot.plot(x_list[lower:upper], ldv_list, color=purg_color_dict[element], label=element)
        pyplot.legend(bbox_to_anchor=(1, 1), loc='upper left')

    pyplot.figure(1)
    # pyplot.yscale('log')
    pyplot.ylim(bottom=0)
    set_labels('Residual vs. Residual Derivative',
               r'$\log_{10}\left(\frac{1}{\rho}\right)$',
               r'$1 - \frac{P_{purgatorio}}{P_{TF}}$')
               # r'$\log_{10}\left(\frac{P_{TF}}{P_{purgatorio}}\right)$')
    figure1.savefig('Diagrams2/Residual Without Shift.png', bbox_inches='tight')

    pyplot.figure(2)
    set_labels('Residual Logarithmic Derivative',
               r'$\log_{10}\left(\frac{1}{\rho}\right)$',
               r'$\frac{\frac{d}{dx}r(x, E)}{r(x, E)}$')
    figure2.savefig('Diagrams2/Residual Logarithmic Derivative Without Shift.png', bbox_inches='tight')

    x_shift_dict, y_shift_dict = dict(), dict()
    x_shift_dict['H'], y_shift_dict['H'] = 0, 0

    di = 4
    for element in element_list[1:]:
        ldv_list = residual_ldv_dict[element]
        dldv_list = [ldv_list[i + di] - ldv_list[i] for i in range(len(ldv_list) - di)]
        k = np.argmin(np.abs(dldv_list))
        shift_index = k + math.floor((len(x_list) - len(ldv_list) + di) / 2)
        x_shift_dict[element] = -(x_list[shift_index] - x_list[0]) - np.log10(atomic_mass_dict[element])
        y_shift_dict[element] = ldv_list[k + math.floor(di / 2)]

    x_shifts = np.asarray(list(x_shift_dict.values()), dtype=float)
    y_shifts = np.asarray(list(y_shift_dict.values()), dtype=float)

    figure3 = pyplot.figure(3)
    pyplot.scatter(np.log10(atomic_numbers)[1:], x_shifts[1:])
    set_labels('X-shift vs. Atomic Number', r'$\log_{10}(z)$', r'$X-shift$')
    figure3.savefig('Diagrams2/LDV X-shift vs. Atomic Number.png', bbox_inches='tight')

    figure4 = pyplot.figure(4)
    pyplot.scatter(np.log(atomic_numbers)[1:], y_shifts[1:])
    set_labels('Y-shift vs. Atomic Number', r'$\ln(z)$', r'$Y-shift$')
    figure4.savefig('Diagrams2/LDV Y-shift vs. Atomic Number.png', bbox_inches='tight')

    x_scale = np.linalg.lstsq(np.vstack((np.log10(atomic_numbers[1:]), np.ones(len(atomic_numbers) - 1))).T, x_shifts[1:], rcond=None)[0][0]
    print(x_scale)
    y_scale = -np.linalg.lstsq(np.vstack((np.log(atomic_numbers[1:]), np.ones(len(atomic_numbers) - 1))).T, y_shifts[1:], rcond=None)[0][0]
    # y_scale = 0.71985

    figure5 = pyplot.figure(5)
    for element in element_list[1:]:
        m, z = atomic_mass_dict[element], atomic_number_dict[element]
        ldv_list = residual_ldv_dict[element]

        k = len(x_list) - len(ldv_list)
        lower, upper = math.floor(k / 2), -math.ceil(k / 2)

        pyplot.plot(x_list[lower:upper] + np.log10(m * z ** x_scale), ldv_list + np.log(z) * y_scale, color=purg_color_dict[element], label=element)
        pyplot.legend(bbox_to_anchor=(1, 1), loc='upper left')

    set_labels('Shifted Residual Logarithmic Derivative',
               r'$\log_{10}\left(\frac{m \cdot z^{' + str(x_scale) + r'}}{\rho}\right)$',
               r'$\frac{\frac{d}{dx}r(x, E)}{r(x, E)} + ' + str(y_scale) + r'\cdot \ln(z)$')
    figure5.savefig('Diagrams2/Residual Logarithmic Derivative with Theorized Shift', bbox_inches='tight')

    residual_x_dict, scaled_y_dict = dict(), dict()

    z_ref = 13 # min(atomic_numbers[1:])

    figure6, figure7 = pyplot.figure(6), pyplot.figure(7)
    for element in element_list[1:]:
        m, z = atomic_mass_dict[element], atomic_number_dict[element]

        shifted_x = x_list + np.log10(m * z ** x_scale)
        scaled_x = y_scale * shifted_x
        y_list = residual_dict[element]
        scaled_y = y_list * (z ** scaled_x)

        pyplot.figure(6)
        pyplot.plot(shifted_x, scaled_y, color=purg_color_dict[element], label=element)
        pyplot.legend(bbox_to_anchor=(1, 1), loc='upper left')

        z_norm = z / z_ref
        pyplot.figure(7)
        pyplot.plot(shifted_x, y_list * (z_norm ** scaled_x), color=purg_color_dict[element], label=element)
        pyplot.legend(bbox_to_anchor=(1, 1), loc='upper left')

    pyplot.figure(6)
    set_labels('Scaled Residual',
               r'$\log_{10}\left(\frac{m \cdot z^{' + str(x_scale) + r'}}{\rho}\right)$',
               r'$\left(1 - \frac{P_{purgatorio}}{P_{TF}}\right) \cdot z^{' + str(y_scale) + r'\cdot x}$')
               # r'$\log_{10}\left(\frac{P_{TF}}{P_{purgatorio}}\right) \cdot z^{' + str(y_scale) + r'\cdot x}$')
    figure6.savefig('Diagrams2/Scaled Residual with Theorized Shift', bbox_inches='tight')

    pyplot.figure(7)
    # pyplot.yscale('log')
    set_labels('Scaled Residual (Reference)',
               r'$\log_{10}\left(\frac{m \cdot z^{' + str(x_scale) + r'}}{\rho}\right)$',
               r'$\left(1 - \frac{P_{purgatorio}}{P_{TF}}\right) \cdot (z / z^*)^{' + str(y_scale) + r'\cdot x}$')
               # r'$\log_{10}\left(\frac{P_{TF}}{P_{purgatorio}}\right) \cdot (z / z^*)^{' + str(y_scale) + r'\cdot x}$')
    figure7.savefig('Diagrams2/Scaled Residual (Reference) with Theorized Shift', bbox_inches='tight')


    closed_figures = [2, 3, 4, 6]
    for n in closed_figures:
        pyplot.close(eval(f'figure{n}'))

    pyplot.tight_layout()

    pyplot.show()




