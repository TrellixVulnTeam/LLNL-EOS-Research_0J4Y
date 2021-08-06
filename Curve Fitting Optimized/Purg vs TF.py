
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

    print(len(Engine.complete_list))
    # element_list = Engine.element_range('Y', 'Cd')
    # element_list = Engine.complete_list[:6]
    element_list = ['H'] + ['Ga', 'Ge', 'As', 'Se']
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
    # purg_dv_dict, tf_dv_dict = dict(), dict()
    difference_dv_dict = dict()

    atomic_mass_dict, atomic_number_dict = dict(), dict()
    orbital_dict = dict()

    difference_dict = dict()

    figure1 = pyplot.figure(1)
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

        '''purg_derivative = Utils.SMA_smooth(
            np.asarray([purg_y[i + 1] - purg_y[i] for i in range(len(x_list) - 1)], dtype=float) / dx,
            buffer=100)
        tf_derivative = Utils.SMA_smooth(
            np.asarray([tf_y[i + 1] - tf_y[i] for i in range(len(x_list) - 1)], dtype=float) / dx,
            buffer=100)'''

        x_dict[element], purg_y_dict[element], tf_y_dict[element] = shifted_x, purg_y, tf_y
        # purg_dv_dict[element], tf_dv_dict[element] = purg_derivative, tf_derivative

        purg_color_dict[element], tf_color_dict[element] = purg_color, tf_color

        pyplot.plot(shifted_x, purg_y - (10 / 3) * np.log10(z), color=purg_color, label=element)
        pyplot.plot(shifted_x, tf_y - (10 / 3) * np.log10(z), color=tf_color)
        pyplot.legend(bbox_to_anchor=(1, 1), loc='upper left')

        diff_y = tf_y - purg_y
        difference_dict[element] = diff_y
        diff_derivative = Utils.SMA_smooth(
            np.asarray([diff_y[i + 1] - diff_y[i] for i in range(len(x_list) - 1)], dtype=float) / dx,
            buffer=40)
        # diff_derivative = np.asarray([diff_y[i + 1] - diff_y[i] for i in range(len(x_list) - 1)], dtype=float) / dx
        difference_dv_dict[element] = diff_derivative

    set_labels('Thomas Fermi vs. Purgatorio',
               r'$\log_{10}\left(\frac{mz}{\rho}\right)$',
               r'Dark: $\log_{10}\left(\frac{P_{TF}}{z^{10 / 3}}\right)$''\n'
               r'Light: $\log_{10}\left(\frac{P_{purgatorio}}{z^{10 / 3}}\right)$')
    figure1.savefig('Diagrams/Thomas Fermi vs. Purgatorio.png', bbox_inches='tight')

    shift_dict = dict()

    for element in element_list:
        index = -1
        k = float('inf')
        for i in range(len(x_list) - 200):
            x_range = difference_dict['H'][i:i + 200]
            y_range = difference_dict[element][:200]

            # x_range = difference_dict['H'][i:len(x_list)]
            # y_range = difference_dict[element][:len(x_list) - i]
            if (v := np.linalg.lstsq(np.vstack((x_range, )).T, y_range, rcond=None)[1]) < k:
                k = v
                index = i
        shift = x_list[index] - x_list[0]
        shift_dict[element] = shift

    figure2, figure3, figure4 = pyplot.figure(2), pyplot.figure(3), pyplot.figure(4)

    for element in element_list:
        m, z = atomic_mass_dict[element], atomic_number_dict[element]
        print(f'{element}: {np.log10(m)}, {np.log10(z)}')
        print(f'\tShift: {shift_dict[element]}')

        pyplot.figure(2)
        pyplot.plot(x_list, difference_dict[element], color=purg_color_dict[element], label=element)
        # k = len(x_list) - len(difference_dv_dict[element])
        # pyplot.plot(x_list[math.floor(k / 2):-math.ceil(k / 2)], difference_dv_dict[element], color=tf_color_dict[element])
        pyplot.legend(bbox_to_anchor=(1, 1), loc='upper left')

        pyplot.figure(3)
        pyplot.plot(x_list + shift_dict[element], difference_dict[element], color=purg_color_dict[element], label=element)
        pyplot.legend(bbox_to_anchor=(1, 1), loc='upper left')

        pyplot.figure(4)
        pyplot.plot(x_list + np.log10(m * z ** (-2 / 3)), difference_dict[element], color=purg_color_dict[element], label=element)
        pyplot.legend(bbox_to_anchor=(1, 1), loc='upper left')

    pyplot.figure(2)
    pyplot.ylim(bottom=0)
    set_labels('Residual Aligned',
               r'$\log_{10}\left(\frac{1}{\rho}\right)$',
               r'$\log_{10}\left(\frac{P_{TF}}{P_{purgatorio}}\right)$')
    figure2.savefig('Diagrams/Residual Without Shift.png', bbox_inches='tight')

    pyplot.figure(3)
    pyplot.ylim(bottom=0)
    set_labels('Residual with Optimized Shift',
               r'$\log_{10}\left(\frac{m}{\rho}\right) + shift$',
               r'$\log_{10}\left(\frac{P_{TF}}{P_{purgatorio}}\right)$')
    figure3.savefig('Diagrams/Residual vs. Optimized Shift.png', bbox_inches='tight')

    pyplot.figure(4)
    pyplot.ylim(bottom=0)
    set_labels('Residual with Theorized Shift',
               r'$\log_{10}\left(\frac{m}{\rho z^{2 / 3}}\right)$',
               r'$\log_{10}\left(\frac{P_{TF}}{P_{purgatorio}}\right)$')
    figure4.savefig('Diagrams/Residual vs. Theorized Shift.png', bbox_inches='tight')

    print(shift_dict)

    atomic_numbers = np.asarray(list(atomic_number_dict.values()), dtype=float)
    atomic_masses = np.asarray(list(atomic_mass_dict.values()), dtype=float)
    shifts = np.asarray(list(shift_dict.values()), dtype=float)

    figure5 = pyplot.figure(5)
    pyplot.scatter(np.log10(atomic_numbers), shifts - np.log10(atomic_masses), color='black', s=3)
    pyplot.ylim(top=0)

    set_labels('Shift vs. Atomic Number', r'$\log_{10}(z)$', r'$shift$')
    figure5.savefig('Diagrams/Shift vs. Atomic Number.png', bbox_inches='tight')

    coefficient_dict = dict()
    shift_index_dict = dict()

    for element in element_list:
        m, z = atomic_mass_dict[element], atomic_number_dict[element]
        shift = np.log10(m * z ** (-2 / 3))
        shift_index = np.searchsorted(x_list - x_list[0], shift, side='right')
        x_range = difference_dict['H'][shift_index:]
        y_range = difference_dict[element][:len(x_list) - shift_index]
        coefficient_dict[element] = np.linalg.lstsq(np.vstack((x_range, )).T, y_range, rcond=None)[0]

        shift_index_dict[element] = shift_index

    shift_index_dict['H'] = 0
        
    coefficients = np.asarray(list(coefficient_dict.values()), dtype=float)

    scale = np.linalg.lstsq(np.vstack((np.log10(atomic_numbers),)).T, coefficients - 1, rcond=None)[0][0][0]
    print(scale)

    figure6 = pyplot.figure(6)
    pyplot.scatter(np.log10(atomic_numbers), coefficients, color='black', s=3)
    xrange = np.arange(*pyplot.xlim())
    pyplot.plot(xrange, 1 + scale * xrange, color='blue', linewidth=1)
    pyplot.ylim(bottom=0)

    set_labels('Coefficient vs. Atomic Number', r'$\log_{10}(z)$',
               r'$c(E) = \frac{\log_{10}\left(\frac{P_{TF}(E)}{P_{purgatorio}(E)}\right)}'
               r'{\log_{10}\left(\frac{P_{TF}(H)}{P_{purgatorio}(H)}\right)}$')
    figure6.savefig('Diagrams/Coefficient vs. Atomic Number.png', bbox_inches='tight')

    scaled_y_dict = dict()

    figure7 = pyplot.figure(7)
    for element in element_list:
        m, z = atomic_mass_dict[element], atomic_number_dict[element]

        scaled_y = difference_dict[element] / (1 + np.log10(z) / 3)
        scaled_y_dict[element] = scaled_y

        pyplot.plot(x_list + np.log10(m * z ** (-2 / 3)), scaled_y, color=purg_color_dict[element], label=element)
        pyplot.legend(bbox_to_anchor=(1, 1), loc='upper left')
    pyplot.ylim(bottom=0)

    set_labels('Scaled Residual vs. Theorized Shift',
               r'$\log_{10}\left(\frac{m}{\rho z^{2 / 3}}\right)$',
               r'$\frac{\log_{10}\left(\frac{P_{TF}(E)}{P_{purgatorio}(E)}\right)}{c(E)}$')
    figure7.savefig('Diagrams/Scaled Residual vs. Theorized Shift.png', bbox_inches='tight')

    expanded_x_list = np.asarray([x_list[0] + dx * i for i in range(max(shift_index_dict.values()) + len(x_list))], dtype=float)

    collapsed_y = [[] for i in range(len(expanded_x_list))]
    for element in element_list:
        scaled_y_list = scaled_y_dict[element]
        shift_index = shift_index_dict[element]
        for i in range(len(x_list)):
            collapsed_y[i + shift_index].append(scaled_y_list[i])
    collapsed_y = np.asarray([np.mean(arg) for arg in collapsed_y], dtype=float)

    cut = 25
    expanded_x_list, collapsed_y = expanded_x_list[:-cut], collapsed_y[:-cut]

    figure8 = pyplot.figure(8)
    pyplot.plot(expanded_x_list, collapsed_y)
    pyplot.ylim(bottom=0)

    set_labels('Average Scaled Residual vs. Theorized Shift',
               r'$\log_{10}\left(\frac{m}{\rho z^{2 / 3}}\right)$',
               r'$Average\left(\frac{\log_{10}\left(\frac{P_{TF}(E)}{P_{purgatorio}(E)}\right)}{c(E)}\right)$')
    figure8.savefig('Diagrams/Averaged Scaled Residual vs. Theorized Shift.png', bbox_inches='tight')

    closed_figures = [1, 3, 4, 5, 6, 7, 8]
    for n in closed_figures:
        pyplot.close(eval(f'figure{n}'))

    pyplot.tight_layout()

    pyplot.show()




