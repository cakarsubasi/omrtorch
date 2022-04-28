from typing import Optional, Tuple
import torch
import numpy as np
import json
import music21
from . import SoundObjects
from functools import total_ordering

_notes = ["noteheadFull", "noteheadHalf", "noteheadWhole"]
_accidentals = ["accidentalSharp", "accidentalFlat", "accidentalNatural"]
_clefs = ['gCflef', 'fClef', 'cClef']

__pitch_objects__ = ['noteheadFull', 'noteheadHalf', 'noteheadWhole', 'accidentalSharp', 'accidentalFlat', 'accidentalNatural',
                     'gCflef', 'fClef', 'cClef']


@total_ordering
class Staff():
    '''
    The idea is to have an array of staff objects and update each staff object
    and then suppress the objects that don't have enough pairs

    fields:
    measures:  Nx4 ndarray of normalized xmin, ymin, xmax, ymax values
    stats: dict of information regarding the 
    '''

    def __init__(self, measureObj: np.array = None):
        self.measures = np.array([])
        self.stats = {
            'top': 0,
            'bottom': 0,
            'left': 0,
            'right': 0,
            'center': 0,
            'num': 0,
        }
        if measureObj is not None:
            self.measures = np.expand_dims(measureObj, axis=0)

    def append(self, measureObj: np.array) -> bool:
        if len(self.measures) == 0:
            self.measures = np.expand_dims(measureObj, axis=0)
            return True
        self._calculateStats()
        # accept a measure if its center is inside the top and bottom
        # of existing averaged measures
        center = (measureObj[1] + measureObj[3])/2
        if self.stats['top'] < center < self.stats['bottom']:
            self.measures = np.vstack([self.measures, measureObj])
            return True
        else:
            return False

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

        self.stats['top'] = np.average(measures[:, 1]).astype(float)
        self.stats['bottom'] = np.average(measures[:, 3]).astype(float)
        self.stats['left'] = np.min(measures[:, 0])
        self.stats['right'] = np.max(measures[:, 2])
        self.stats['center'] = self.stats['top'] + \
            (self.stats['bottom'] - self.stats['top'])/2
        self.stats['num'] = measures.shape[0]

    def toDict(self):
        # TODO
        dictionary = {}
        dictionary['type'] = 'staff'
        # Convert to primitive types
        dictionary['objects'] = {}
        dictionary['objects']['top'] = self.stats['top']
        dictionary['objects']['bottom'] = self.stats['bottom']
        return dictionary


class Streamable():
    def __init__(self):

        pass

    def toStream():
        pass


@total_ordering
class SystemStaff():
    '''
    Contains staffs and boundaries

    Generates Measures and populate them with objects TODO

    '''

    def __init__(self, staves: Tuple[Staff], yboundaries: np.array, objects: dict = None):
        '''
        staves: staffs that belong to this system
        yboundaries: ymin, ymax limits of the system
        objects: dictionary of object bboxes, classes, scores
        '''
        # TODO handle two staff systems
        self.staves = staves
        self.boundaries = yboundaries
        self.measure_boxes = staves[0].measures.copy()
        self.measure_boxes[:,1] = yboundaries[0]
        self.measure_boxes[:,3] = yboundaries[1]

        self.objects = _objectify(objects)
        self.objects.sort()

        # List of measures in order
        measures = []
        for measure in self.measure_boxes:
            objects = list(
                filter(lambda obj: measure[0] <= obj.x < measure[2], self.objects))
            measures.append(Measure(boundaries=measure, objects=objects))
        self.measures = measures

    def __lt__(self, other):
        return np.average(self.boundaries) < np.average(other.boundaries)

    def bbox(self):
        # TODO handle multi staff systems
        return np.asarray([0, self.boundaries[0], 1, self.boundaries[1]])

    def toDict(self):
        # TODO add boundaries and staffs
        dictionary = {}
        dictionary['type'] = 'StaffSystem'
        dictionary['objects'] = {}
        dictionary['objects']['measures'] = {idx: obj.toDict()
                                             for idx, obj in enumerate(self.measures)}
        dictionary['objects']['staffs'] = {
            idx: obj.toDict() for idx, obj in enumerate(np.array([staff for staff in self.staves]))}
        dictionary['objects']['boundaries'] = {
            'ymin': self.boundaries[0].astype(float),
            'ymax': self.boundaries[1].astype(float),
        }
        return dictionary
    # TODO

    def toStream(self):
        # get staff position and staff gap
        # self.staves[0].
        pass


@total_ordering
class Measure():
    '''
    This class should essentially be identical to SystemStaff except SystemStaff also has to handle staff measures
    '''

    def __init__(self, boundaries: np.array, objects=None):
        '''
        boundaries: 1x4 bbox coordinates
        objects
        '''
        self.boundaries = boundaries
        if objects is not None:
            self.objects = objects

    def bbox(self):
        return self.boundaries

    def __lt__(self, other):
        return np.average(self.boundaries[[0, 2]]) < np.average(other.boundaries[[0, 2]])

    def toDict(self):
        dictionary = {}
        dictionary['type'] = 'Measure'
        dictionary['objects'] = {idx: obj.toDict()
                                 for idx, obj in enumerate(self.objects)}
        return dictionary

    def toStream(self):

        pass


class Song():
    '''
    Contains system staffs which contain one or more staff objects
    Keeps track of staff regions and assigns notes to these regions
    '''

    def __init__(self, systems: Tuple[SystemStaff], image: np.array):
        self.systems = systems
        self.image = image

    def toDict(self):
        return {idx: system.__dict__ for idx, system in enumerate(self.__dict__['systems'])}

    def assignNotes(self):
        pass

    def toDict(self):
        # TODO, image metadata?
        dictionary = {}
        dictionary['type'] = 'Song'
        dictionary['objects'] = {idx: system.toDict()
                                 for idx, system in enumerate(self.systems)}
        return dictionary

    def toStream(self):
        '''
        Generate a music21 stream from object
        Get the SystemStaff streams and concat them
        '''

        pass

    def toJSON(self):
        '''
        Serialize the song into a JSON string
        # TODO
        '''

        return json.dumps(self.toDict())


class SongFactory():
    '''
    Factory for songs
    '''

    MEASURE_THRESHOLD = 0.75
    OBJECT_THRESHOLD = 0.0

    def __init__(self, image, measuredetections, objectdetections, label_list=None):

        self.image = image
        self.height, self.width = image.shape[1:3]
        self.system_measures = None
        self.staff_measures = None
        self.staves = None
        self.boundaries = None
        self.objects: dict = {}

        if label_list is None:
            label_list = __pitch_objects__

        ####
        # STAFF HANDLING
        ####
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

        staves = process_measures2(staves)
        # Measure processing
        for staff in staves:
            staff.measures = process_measures(staff.measures)

        self.staves = staves

        # get staff boundaries
        boundaries = get_staff_boundaries(
            [staff.stats['center'] for staff in self.staves])
        self.boundaries = boundaries

        # TODO: system detection

        ####
        # OBJECT HANDLING
        ####

        object_boxes = objectdetections['boxes'].cpu().detach().numpy()
        object_labels = objectdetections['labels'].cpu().detach().numpy()
        object_labels = np.asarray([label_list[idx-1]
                                   for idx in object_labels])

        # this is only for gt boxes, real detections will always have scores attached
        if 'scores' not in objectdetections:
            object_scores = np.ones(shape=object_labels.shape)
        else:
            object_scores = objectdetections['scores'].cpu().detach().numpy()

        # Filter unreliable results
        best_boxes = np.where(
            object_scores > self.OBJECT_THRESHOLD)

        object_boxes = object_boxes[best_boxes]
        object_labels = object_labels[best_boxes]
        object_scores = object_scores[best_boxes]

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
            systemStaffs.append(SystemStaff(
                [self.staves[idx]], boundaries[idx], group))

        self.song = Song(systemStaffs, self.image)


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


def detect_systems():
    '''
    Idea: check overlap of measures and system measures
    and return system measures as Tuple[Tuple[int]]

    ie, you have four staffs and they form two systems,
    return  [[0, 1], [2, 3]]

    # TODO
    '''
    pass


def denormalize_bboxes(bboxes, image):
    if len(bboxes.shape) == 1:
        bboxes = np.expand_dims(bboxes, 0)

    bboxes[:, [1, 3]] = bboxes[:, [1, 3]]*image.shape[1]
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]]*image.shape[2]

    return bboxes


def _objectify(objectsdict):
    '''
    Convert the object dictionary into our abstract classes
    '''
    objects = []
    for b, l, s in zip(objectsdict['boxes'], objectsdict['labels'], objectsdict['scores']):
        obj = None
        if l in _notes:
            # TODO: consider lengths (beyond scope potentially)
            obj = SoundObjects.Note(b)
        elif l in _accidentals:
            obj = SoundObjects.Accidental(b, l)
        elif l in _clefs:
            obj = SoundObjects.Clef(b, l)
        if obj is not None:
            objects.append(obj)
    objects.sort()
    return objects


def process_measures(measures, xmin: float = 0.0, xmax: float = 1.0):
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

    Takes a list of measures in the same staff

    Returns a "processed" list of measures

    New list has no gaps and overlaps and extends across the entire staff
    '''
    # TODO: batch suppression

    # TODO: detect large gaps

    # merge measures
    sort_order = np.argsort(measures[:, 0])
    measures = measures[sort_order]

    # xmin
    xmins = measures[:, 0]
    xmaxs = measures[:, 2]

    if len(xmins) < 2:
        return measures
    avgs = np.average([xmins[1:], xmaxs[:-1]], axis=0)

    measures[1:, 0], measures[:-1, 2] = avgs, avgs

    return measures

def process_measures2(staves: Tuple[Staff]):
    '''
    Detects large gaps in staves and fills them with synthetic measures
    '''
    grouped = [staff.measures for staff in staves]
    ungrouped = np.vstack(grouped)
    left_limit = np.min(ungrouped[:,0])
    right_limit = np.max(ungrouped[:,2])
    mingap = np.average(ungrouped[:,2] - ungrouped[:,0])/2
    for gid, group in enumerate(grouped):
        top = np.average(group[:,1])
        bottom = np.average(group[:,3])
        for idx, measure in enumerate(group):
            if idx == 0: # check left of the first detection
                if measure[0] > left_limit + mingap:
                    synth = np.array([left_limit, top, measure[0], bottom])
                    group = np.vstack([group, synth])
            if idx == len(group) - 1: # check right of the last detection
                if measure[2] < right_limit - mingap:
                    synth = np.array([measure[2], top, right_limit, bottom])
                    group = np.vstack([group, synth])
            else: # check between this detection and the next
                if group[idx+1][0] - measure[2] > mingap:
                    synth = np.array([measure[2], top, group[idx+1][0], bottom])
                    group = np.vstack([group, synth])
            grouped[gid] = group

    for idx in range(len(staves)):
        sort_order = np.argsort(grouped[idx][:,0])
        grouped[idx] = grouped[idx][sort_order]
        staves[idx].measures = grouped[idx]

    return staves
