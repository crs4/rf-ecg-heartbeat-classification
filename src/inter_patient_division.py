
import numpy as np

def create_train_test_sets(features, labels, sources):
    # Selecting test and train sets using literature defined inter-patient paradigm
    DS1 = {'101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122',
        '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230'}
    DS2 = {'100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210',
        '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234'}

    train_set = {
        'features': np.array([x for k, x in enumerate(features) if sources[k] in DS1]),
        'labels': np.array([x for k, x in enumerate(labels) if sources[k] in DS1]),
        'sources': np.array([x for k, x in enumerate(sources) if sources[k] in DS1])
    }
    test_set = {
        'features': np.array([x for k, x in enumerate(features) if sources[k] in DS2]),
        'labels': np.array([x for k, x in enumerate(labels) if sources[k] in DS2]),
        'sources': np.array([x for k, x in enumerate(sources) if sources[k] in DS2])
    }
    return train_set, test_set
    


    