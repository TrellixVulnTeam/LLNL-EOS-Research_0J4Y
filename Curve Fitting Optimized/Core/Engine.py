
import os
import numpy as np
from Core.Library import Library


class Engine:
    complete_list = None
    library = Library()

    def __init__(self):
        complete_list = []
        for directory in os.listdir('Pressure Data/Purg Data'):
            element = directory[:directory.index('.')]
            if hasattr(Engine.library[element], 'tf_data'):
                complete_list.append(element)
        complete_list = sorted(complete_list, key=lambda arg: Engine.library[arg].element.atomic_number)
        complete_list[:3] = ('H', 'D', 'T')
        Engine.complete_list = complete_list

    # SECTION: Element Retrieval =======================================================================================

    @classmethod
    def element_range(cls, element1, element2):
        i1, i2 = Engine.complete_list.index(element1), Engine.complete_list.index(element2)
        l = Engine.complete_list[min(i1, i2):max(i1, i2) + 1]
        if l[0] == 'H':
            return l
        else:
            return ['H'] + l

