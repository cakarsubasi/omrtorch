import numpy as np
from music21 import note as mnote

'''
Classes to convert bounding boxes to staff measure relative coordinates
# TODO: EVERYTHING
'''

class glyph:

    def __init__(self, bbox: np.array):
        self.x = None
        self.y = None
        self.size = None

        pass

    def toBox(self):

        pass


class Note(glyph):

    def __init__(self, bbox: np.array):
        super.__init__(bbox)

        pass


class Accidental(glyph):
    '''
    Modify the notes to the right of the measure,
    if to the left of all notes, modify every note
    '''
    def __init__(self, bbox: np.array):
        super.__init__(bbox)
        self.type = None

        pass


class Rest(glyph):
    '''
    Optional, rests do nothing
    '''
    def __init__(self, bbox: np.array):

        pass


class Clef(glyph):
    '''
    Determines the octave of the staff
    '''
    def __init__(self, bbox: np.array):

        pass



def getNote(clef: Clef, note: Note) -> mnote.Note:

    pass
