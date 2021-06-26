import os

from Element import Element


class Library:
    def __init__(self):
        for directory in os.listdir('Elements'):
            if len(directory) <= 2:
                setattr(self, directory, Element(directory))

    def __getitem__(self, element):
        if element in os.listdir('Elements'):
            return getattr(self, element)
        else:
            return None
