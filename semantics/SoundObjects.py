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

@total_ordering
class glyph:

    def __init__(self, bbox: np.array):

        self.x = (bbox[0] + bbox[2])/2
        self.y = (bbox[1] + bbox[3])/2
        # size is only needed to go back to bbox
        self.size = [bbox[2]-bbox[0], bbox[3]-bbox[1]]

        pass

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
        return round(2*(self.y - bottom)/gap)

    def __lt__(self, other):
        return self.x < other.x

    def __eq__(self, other):
        return self.x == other.x
        


class Note(glyph):

    def __init__(self, bbox: np.array, length=1.0):
        super().__init__(bbox)
        self.length = length
        pass


class Accidental(glyph):
    '''
    Modify the notes to the right of the measure,
    if to the left of all notes, modify every note
    '''
    def __init__(self, bbox: np.array, type="Natural"):
        super().__init__(bbox)
        self.type = type

        pass


class Rest(glyph):
    '''
    Optional, rests do nothing
    '''
    def __init__(self, bbox: np.array):
        super().__init__(bbox)
        pass


class Clef(glyph):
    '''
    Determines the octave of the staff

    Clefs are surprisingly complicated, we are not going to bother with covering every possibility,
    just the three most common.
    '''
    def __init__(self, bbox: np.array, type='g'):
        super().__init__(bbox)
        pass



def getNote(clef: Clef, relativepos: int) -> mnote.Note:

    pass
