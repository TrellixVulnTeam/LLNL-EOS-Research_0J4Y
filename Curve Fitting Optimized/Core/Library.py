import os

from Core.Element import Element


class Library:
    def __init__(self):
        for directory in os.listdir('Pressure Data/Purg Data'):
            directory = directory[:directory.index('.')]
            try:
                element = Element(directory)
                setattr(self, directory, element)
            except AssertionError:
                pass

    def __getitem__(self, element):
        try:
            return getattr(self, element)
        except AttributeError:
            return None
