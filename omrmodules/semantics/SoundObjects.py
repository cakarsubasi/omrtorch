import numpy as np
from music21 import note as mnote
from functools import total_ordering

'''
Classes to convert bounding boxes to staff measure relative coordinates
# TODO: EVERYTHING
'''

__notes = ["noteheadFull", "noteheadHalf", "noteheadWhole"]
__notelengths = ["Whole", "Half", "Quarter", "8th", "16th"]
__accidentals = ["accidentalSharp", "accidentalFlat", "accidentalNatural"]
__clefs = ['gClef', 'fClef', 'cClef']


@total_ordering
class glyph:

    def __init__(self, bbox: np.array):

        self.x = (bbox[0] + bbox[2])/2
        self.y = (bbox[1] + bbox[3])/2
        # size is only needed to go back to bbox
        self.size = [bbox[2]-bbox[0], bbox[3]-bbox[1]]

    def bbox(self) -> np.array:
        '''
        return the bounding box of the glyph
        '''
        xmin: float = self.x - self.size[0]/2
        xmax: float = self.x + self.size[0]/2
        ymin: float = self.y - self.size[1]/2
        ymax: float = self.y + self.size[1]/2

        return np.asarray([xmin, ymin, xmax, ymax])

    def relativePos(self, bottom, gap) -> int:
        '''
        Get position relative to the lowest staffline
        Each staffline has a gap of 2.0
        Lowest staffline: 0.0
        Return the value rounded to the nearest int
        '''
        return round(2*(bottom - self.y)/gap)

    def __lt__(self, other):
        return self.x < other.x

    def __eq__(self, other):
        return self.x == other.x

    def toDict(self):
        '''
        serialization method
        # TODO: use a better format for Java parsing
        '''
        dictionary = {}
        dictionary['type'] = 'glyph'
        dictionary['objects'] = {'x': self.x,
                                 'y': self.y,
                                 'w': self.size[0].astype(float),
                                 'h': self.size[1].astype(float),
                                 }
        return dictionary


class Note(glyph):

    def __init__(self, bbox: np.array, length=1.0):
        super().__init__(bbox)
        self.type = 'Note'
        self.length = length

    def toDict(self):
        dictionary = super().toDict()
        dictionary['type'] = 'Note'
        dictionary['objects']['length'] = self.length
        return dictionary


class Accidental(glyph):
    '''
    Modify the notes to the right of the measure,
    if to the left of all notes, modify every note
    '''

    def __init__(self, bbox: np.array, AccidentalType="Natural"):
        super().__init__(bbox)
        self.type = AccidentalType

    def toDict(self):
        dictionary = super().toDict()
        dictionary['type'] = 'Accidental'
        dictionary['objects']['accidentalType'] = self.type
        return dictionary


class Rest(glyph):
    '''
    Optional, rests do nothing
    '''

    def __init__(self, bbox: np.array, length=1.0):
        super().__init__(bbox)
        self.length = length

    def toDict(self):
        dictionary = super().toDict()
        dictionary['type'] = 'Rest'
        dictionary['objects']['length'] = self.length
        return dictionary


class Clef(glyph):
    '''
    Determines the octave of the staff

    Clefs are surprisingly complicated, we are not going to bother with covering every possibility,
    just the three most common.
    '''

    def __init__(self, bbox: np.array, clefType='gClef'):
        super().__init__(bbox)
        self.type = clefType

    def toDict(self):
        dictionary = super().toDict()
        dictionary['type'] = 'Clef'
        dictionary['objects']['clefType'] = self.type
        return dictionary


def getNote(clef: str, relativepos: int) -> mnote.Note:
    '''
    Get music21 note based on Clef and relative position.
    '''
    name: str = ''

    if (clef == 'gClef'):
        name = {
            -4: 'a3',
            -3: 'b3',
            -2: 'c4',
            -1: 'd4',
            0: 'e4',
            1: 'f4',
            2: 'g4',
            3: 'a4',
            4: 'b4',
            5: "c5",
            6: "d5",
            7: "e5",
            8: "f5",
            9: "g5",
            10: "a5",
            11: "b5",
            12: "c6",
        }[relativepos]   
        pass

    elif (clef == 'fClef'):
        name = {
            -4 : "c2",
            -3 : "d2",
            -2 : "e2",
            -1 : "f2",
            0  : "g2",
            1  : "a2",
            2  : "b2",
            3  : "c3",
            4  : "d3",
            5  : "e3",
            6  : "f3",
            7  : "g3",
            8  : "a3",
            9  : "b3",
            10 : "c4",
            11 : "d4",
            12 : "e4",
        }[relativepos]  

    elif (clef == 'cClef'):
        # TODO: update this to be correct
        name = {
            -4  : "b2",
            -3  : "c3",
            -2  : "d3",
            -1  : "e3",
            0   : "f3",
            1   : "g3",
            2   : "a3",
            3   : "b3",
            4   : "c4",
            5   : "d4",
            6   : "e4",
            7   : "f4",
            8   : "g4",
            9   : "a4",
            10  : "b4",
            11  : "c5",
            12  : "d5",
        }[relativepos]  

    else:
        pass
    
    return mnote.Note(name)
