from typing import Optional, Tuple
import torch
import numpy as np
import music21




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
        # TODO
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


class Streamable():
    def __init__(self):

        pass

    def toStream():
        pass


class SystemStaff():
    '''
    Contains staffs and boundaries

    '''

    def __init__(self, staves: Tuple[Staff], boundaries: np.array, objects: dict):
        self.staves = staves
        self.boundaries = boundaries
        self.objects = objects

    def _objectify(self):
        '''
        Convert the object dictionary into 
        '''
        pass

    def bbox(self):
        # TODO handle multi staff systems
        return np.asarray([0, self.boundaries[0], 1, self.boundaries[1]])

    def toStream(self):
        # get staff position and staff gap
        pass


class Measure():
    '''
    measure handling ideas:
    There are two possibilities
    Measures are the same length on every staff or measures are different length in each stuff

    Latter case is more general.

    Assume that we detected a leftmost and rightmost measure. Use these as the min and max values.

    For each staff, check all of the "gaps"

    Add a syntethic measure if the gap is larger than expected
    ////
    Batch suppression?
    If there are two measures with close left and right boundaries, average them

    If there are two measures with close left OR right boundaries, there is ambiguity
    (the longer one is more likely to be the right detection due to some note bars being detected)

    This class should essentially be identical to SystemStaff except SystemStaff also has to handle staff measures
    '''
    pass


class Song():
    '''
    Contains system staffs which contain one or more staff objects
    Keeps track of staff regions and assigns notes to these regions
    '''

    def __init__(self, systems: Tuple[SystemStaff], image: np.array):
        self.systems = systems
        self.image = image

    def assignNotes(self):
        pass

    def toStream(self):
        '''
        Generate a music21 stream from object
        Get the SystemStaff streams and concat them
        '''

        pass

    pass


class SongFactory():
    '''
    Factory for songs
    '''

    MEASURE_THRESHOLD = 0.75
    #OBJECT_THRESHOLD = 1.0

    def __init__(self, image, measuredetections, objectdetections, label_dict=None):

        self.image = image
        self.height, self.width = image.shape[1:3]
        self.system_measures = None
        self.staff_measures = None
        self.staves = None
        self.boundaries = None
        self.objects: dict = {}

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

        object_boxes = objectdetections['boxes'].cpu().detach().numpy()
        object_labels = objectdetections['labels'].cpu().detach().numpy()
        # this is only for gt boxes, real detections will always have scores attached
        if 'scores' not in objectdetections:
            object_scores = np.ones(shape=object_labels.shape)
        else:
            object_scores = objectdetections['scores'].cpu().detach().numpy()

        # normalize
        object_boxes[:, [1, 3]] = object_boxes[:, [1, 3]]/self.height
        object_boxes[:, [0, 2]] = object_boxes[:, [0, 2]]/self.width

        # sort by y-axis (ascending)
        sort_order = np.argsort((object_boxes[:, 1]+object_boxes[:, 3]))

        object_boxes = object_boxes[sort_order]
        object_labels = object_labels[sort_order]
        object_scores = object_scores[sort_order]

        self.objects['boxes'] = object_boxes
        self.objects['labels'] = object_labels
        self.objects['scores'] = object_scores

        # generate SystemMeasures
        groups = []
        for boundary in self.boundaries:
            staff_dict = {}
            group = boundary[0] < self.objects['boxes'][:, 1]
            group = group * (self.objects['boxes'][:, 1] < boundary[1])
            for k, v in self.objects.items():
                staff_dict[k] = v[group]
            groups.append(staff_dict)

        # TODO if we extract measure boundaries precisely, we can also split to measures
        # need to do post process for measure boundaries
        
        systemStaffs = []

        for idx, group in enumerate(groups):
            systemStaffs.append(SystemStaff(self.staves[idx], boundaries[idx], group))
            
        self.song = Song(systemStaffs, self.image)

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

def get_measure_boundaries():
    # TODO
    pass

def detect_systems():
    '''
    Idea: check overlap of measures and system measures
    and return system measures as Tuple[Tuple[int]]

    ie, you have four staffs and they form two systems,
    return  [[0, 1], [2, 3]]
    '''
    pass


def denormalize_bboxes(bboxes, image):
    if len(bboxes.shape) == 1:
        bboxes = np.expand_dims(bboxes, 0)

    bboxes[:, [1, 3]] = bboxes[:, [1, 3]]*image.shape[1]
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]]*image.shape[2]

    return bboxes
    