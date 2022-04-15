from typing import Optional, Tuple
import numpy as np

class Staff():
    '''
    The idea is to have an array of staff objects and update each staff object
    and then suppress the objects that don't have enough pairs

    fields:
    measures: array of 1x4 ndarrays of normalized xmin, ymin, xmax, ymax values
    stats: dict of information regarding the 
    '''

    # Constant to decide whether to include or reject a measure
    TOLERANCE = 0.01

    def __init__(self, measureObj: np.array=None):
        self.measures = []
        self.stats = {}
        if measureObj is not None:
            self.measures.append(measureObj)
    
    def append(self, measureObj: np.array) -> bool:
        if len(self.measures) == 0:
            self.measures.append(measureObj)
            return True
        self._calculateStats()
        top = measureObj[1]
        bottom = measureObj[3]
        if np.abs(self.stats['top'] - top) > self.TOLERANCE or np.abs(self.stats['bottom'] - bottom) > self.TOLERANCE:
            return False
        else:
            self.measures.append(measureObj)
            return True

    def __lt__(self, o):
        self_center = self.getStats()['center']
        o_center = o.getStats()['center']

        return self_center < o_center
    
    def __str__(self):
        return str(self.getStats())
        
    def getStats(self):
        self._calculateStats()
        return self.stats

    def getStaffLines(self) -> np.array:
        '''
        Get the staff lines
        '''
        self._calculateStats()
        top = self.stats['top']
        bottom = self.stats['bottom']

        lines = np.linspace(top, bottom, 5)
        lines = np.hstack([lines[0], np.repeat(lines[1:4], 2), lines[4]])
        lines = np.reshape(lines, (4,2))
        return lines

    def getStaffLinesBBs(self) -> np.array:
        lines = self.getStaffLines()
        xmin = self.stats['left']
        xmax = self.stats['right']
        lines = np.insert(lines, 0, xmin, axis=1)
        lines = np.insert(lines, 2, xmax, axis=1)

        return lines

    def _calculateStats(self):
        measures = np.array(self.measures)

        self.stats['top'] = np.average(measures[:,1])
        self.stats['bottom'] = np.average(measures[:,3])
        self.stats['left'] = np.min(measures[:,0])
        self.stats['right'] = np.max(measures[:,2])
        self.stats['center'] = self.stats['top'] + (self.stats['bottom'] - self.stats['top'])/2
        self.stats['num'] = measures.shape[0]

class SystemStaff():
    '''
    Contain one or two staffs
    '''
    def __init__ (self, staff1: Staff, staff2: Optional[Staff]=None):
        self.staff1 = staff1
        
        if staff2 is not None:
            self.staff2 = staff2
        
    
    pass

class Measure():
    '''
    key: key signature
    '''

class Song():
    '''
    Get a tuple of system measures
    '''
    def __init__(self, systems: Tuple[SystemStaff], image: np.array, notes: np.array):
        self.systems = systems
        self.image = image
        self.notes = notes

    def assignNotes(self):
        pass

    pass

class Page():
    '''
    Factory for songs
    '''
    def __init__(self, image, measuredetections, objectdetections):
        '''
        :param image:

        
        '''
        self.image = image
        self.measuredetections = measuredetections
        self.objectdetections = objectdetections
        pass

    pass