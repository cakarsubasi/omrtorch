import numpy as np
import SystemObjects

if __name__ == "__main__":

    some_measures = np.array([[0.59070754, 0.37419364, 0.7296071 , 0.46695033],
       [0.05971682, 0.37450802, 0.25843462, 0.46590468],
       [0.420092  , 0.374791  , 0.58300257, 0.4671924 ],
       [0.25922823, 0.37479445, 0.41415334, 0.46614403],
       [0.7334418 , 0.37778202, 0.8635829 , 0.46550924]],)
    

    some_measures = SystemObjects.process_measures(some_measures)

    assert (np.all(some_measures[1:,0] == some_measures[:-1,2]))

    print("All assertions passed.")