from typing import Optional, Tuple
import torch
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

    def __init__(self, measureObj: np.array = None):
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
        lines = np.reshape(lines, (4, 2))
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

        self.stats['top'] = np.average(measures[:, 1])
        self.stats['bottom'] = np.average(measures[:, 3])
        self.stats['left'] = np.min(measures[:, 0])
        self.stats['right'] = np.max(measures[:, 2])
        self.stats['center'] = self.stats['top'] + \
            (self.stats['bottom'] - self.stats['top'])/2
        self.stats['num'] = measures.shape[0]


class SystemStaff():
    '''
    Contain one or two staffs
    '''

    def __init__(self, staff1: Staff, staff2: Optional[Staff] = None):
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
    Contains system staffs which contain one or more staff objects
    Keeps track of staff regions and assigns notes to these regions
    '''

    def __init__(self, systems: Tuple[SystemStaff], image: np.array, notes: np.array):
        self.systems = systems
        self.image = image
        self.notes = notes

    def assignNotes(self):
        pass

    def toStream(self):
        '''
        Generate a music21 stream from object'''

        pass

    pass


class SongFactory():
    '''
    Factory for songs
    '''

    MEASURE_THRESHOLD = 0.75

    def __init__(self, image, measuredetections, objectdetections):

        self.image = image
        self.height, self.width = image.shape[1:3]
        self.system_measures = None
        self.staff_measures = None
        self.staves = None
        self.boundaries = None

        # filter unreliable results
        best_boxes = torch.where(
            measuredetections['scores'] > self.MEASURE_THRESHOLD)

        detections_reduced = {}
        detections_reduced['boxes'] = measuredetections['boxes'][best_boxes]
        detections_reduced['labels'] = measuredetections['labels'][best_boxes]
        detections_reduced['scores'] = measuredetections['scores'][best_boxes]

        # get system measures and staff measures
        system_measures = detections_reduced['boxes'][torch.where(
            detections_reduced['labels'] == 1)]
        staff_measures = detections_reduced['boxes'][torch.where(
            detections_reduced['labels'] == 2)]

        # move tensors to CPU
        system_measures = system_measures.cpu().detach().numpy()
        staff_measures = staff_measures.cpu().detach().numpy()

        # normalize
        staff_measures[:, [1, 3]] = staff_measures[:, [1, 3]]/self.height
        staff_measures[:, [0, 2]] = staff_measures[:, [0, 2]]/self.width

        system_measures[:, [1, 3]] = system_measures[:, [1, 3]]/self.height
        system_measures[:, [0, 2]] = system_measures[:, [0, 2]]/self.width

        # sort by y axis (ascending)
        sort_order = np.argsort(staff_measures[:, 1])
        staff_measures = staff_measures[sort_order]

        sort_order = np.argsort(system_measures[:, 1])
        system_measures = system_measures[sort_order]

        self.system_measures = system_measures
        self.staff_measures = staff_measures

        # Set up staffs
        staves = []
        staves.append(Staff())
        for i in range(staff_measures.shape[0]):
            if staves[-1].append(staff_measures[i]) is True:
                pass
            else:
                nextstaff = Staff()
                staves.append(nextstaff)
                staves[-1].append(staff_measures[i])

        self.staves = staves

        # get staff boundaries
        boundaries = get_staff_boundaries(
            [staff.stats['center'] for staff in self.staves])
        self.boundaries = boundaries

        # TODO: system detection

        # TODO: process object detections
        self.objectdetections = objectdetections
        pass

    def visualize(self):
        # TODO
        pass


def get_staff_boundaries(measure_centers):
    '''
    returns an Nx2 ndarray 
    '''
    x1 = measure_centers
    x2 = np.roll(measure_centers, 1)
    gaps = ((x1+x2)/2)[1:]

    x3 = np.concatenate([[0], gaps])
    x4 = np.concatenate([gaps, [1]])

    return np.stack([x3, x4], axis=1)
