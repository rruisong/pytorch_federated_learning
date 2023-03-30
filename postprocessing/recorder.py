import numpy as np
import matplotlib.pyplot as plt
import json
from json import JSONEncoder
import pickle

json_types = (list, dict, str, int, float, bool, type(None))


class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}


def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(dct['_python_object'].encode('latin-1'))
    return dct


class Recorder(object):
    def __init__(self):
        self.res_list = []
        self.res = {'server': {'iid_accuracy': [], 'train_loss': []},
                    'clients': {'iid_accuracy': [], 'train_loss': []}}

    def load(self, filename, label):
        """
        Load the result files
        :param filename: Name of the result file
        :param label: Label for the result file
        """
        with open(filename) as json_file:
            res = json.load(json_file, object_hook=as_python_object)
        self.res_list.append((res, label))

    def plot(self):
        """
        Plot the testing accuracy and training loss on number of epochs or communication rounds
        """
        fig, axes = plt.subplots(2)
        for i, (res, label) in enumerate(self.res_list):
            axes[0].plot(np.array(res['server']['iid_accuracy']), label=label, alpha=1, linewidth=2)
            axes[1].plot(np.array(res['server']['train_loss']), label=label, alpha=1, linewidth=2)

        for i, ax in enumerate(axes):
            ax.set_xlabel('# of Epochs', size=12)
            if i == 0:
                ax.set_ylabel('Testing Accuracy', size=12)
            if i == 1:
                ax.set_ylabel('Training Loss', size=12)
            ax.legend(prop={'size': 12})
            ax.tick_params(axis='both', labelsize=12)
            ax.grid()


