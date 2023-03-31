#!/usr/bin/env python
import os
import random
import json
import pickle
import argparse
import yaml
from json import JSONEncoder
from tqdm import tqdm

from fed_baselines.client_base import FedClient
from fed_baselines.client_fedprox import FedProxClient
from fed_baselines.client_scaffold import ScaffoldClient
from fed_baselines.client_fednova import FedNovaClient
from fed_baselines.server_base import FedServer
from fed_baselines.server_scaffold import ScaffoldServer
from fed_baselines.server_fednova import FedNovaServer

from postprocessing.recorder import Recorder
from preprocessing.baselines_dataloader import divide_data
from utils.models import *

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


def fed_args():
    """
    Arguments for running federated learning baselines
    :return: Arguments for federated learning baselines
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True, help='Yaml file for configuration')

    args = parser.parse_args()
    return args
    

def fed_run():
    """
    Main function for the baselines of federated learning
    """
    args = fed_args()
    with open(args.config, "r") as yaml_file:
        try:
            config = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)
    
    algo_list = ["FedAvg", "SCAFFOLD", "FedProx", "FedNova"]
    assert config["client"]["fed_algo"] in algo_list, "The federated learning algorithm is not supported"

    dataset_list = ['MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN', 'CIFAR100']
    assert config["system"]["dataset"] in dataset_list, "The dataset is not supported"

    model_list = ["LeNet", 'AlexCifarNet', "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "CNN"]
    assert config["system"]["model"] in model_list, "The model is not supported"

    np.random.seed(config["system"]["i_seed"])
    torch.manual_seed(config["system"]["i_seed"])
    random.seed(config["system"]["i_seed"])

    client_dict = {}
    recorder = Recorder()

    trainset_config, testset = divide_data(num_client=config["system"]["num_client"], num_local_class=config["system"]["num_local_class"], dataset_name=config["system"]["dataset"],
                                           i_seed=config["system"]["i_seed"])
    max_acc = 0
    # Initialize the clients w.r.t. the federated learning algorithms and the specific federated settings
    for client_id in trainset_config['users']:
        if config["client"]["fed_algo"] == 'FedAvg':
            client_dict[client_id] = FedClient(client_id, dataset_id=config["system"]["dataset"], epoch=config["client"]["num_local_epoch"], model_name=config["system"]["model"])
        elif config["client"]["fed_algo"] == 'SCAFFOLD':
            client_dict[client_id] = ScaffoldClient(client_id, dataset_id=config["system"]["dataset"], epoch=config["client"]["num_local_epoch"], model_name=config["system"]["model"])
        elif config["client"]["fed_algo"] == 'FedProx':
            client_dict[client_id] = FedProxClient(client_id, dataset_id=config["system"]["dataset"], epoch=config["client"]["num_local_epoch"], model_name=config["system"]["model"])
        elif config["client"]["fed_algo"] == 'FedNova':
            client_dict[client_id] = FedNovaClient(client_id, dataset_id=config["system"]["dataset"], epoch=config["client"]["num_local_epoch"], model_name=config["system"]["model"])
        client_dict[client_id].load_trainset(trainset_config['user_data'][client_id])

    # Initialize the clients w.r.t. the federated learning algorithms and the specific federated settings
    if config["client"]["fed_algo"] == 'FedAvg':
        fed_server = FedServer(trainset_config['users'], dataset_id=config["system"]["dataset"], model_name=config["system"]["model"])
    elif config["client"]["fed_algo"] == 'SCAFFOLD':
        fed_server = ScaffoldServer(trainset_config['users'], dataset_id=config["system"]["dataset"], model_name=config["system"]["model"])
        scv_state = fed_server.scv.state_dict()
    elif config["client"]["fed_algo"] == 'FedProx':
        fed_server = FedServer(trainset_config['users'], dataset_id=config["system"]["dataset"], model_name=config["system"]["model"])
    elif config["client"]["fed_algo"] == 'FedNova':
        fed_server = FedNovaServer(trainset_config['users'], dataset_id=config["system"]["dataset"], model_name=config["system"]["model"])
    fed_server.load_testset(testset)
    global_state_dict = fed_server.state_dict()

    # Main process of federated learning in multiple communication rounds

    pbar = tqdm(range(config["system"]["num_round"]))
    for global_round in pbar:
        for client_id in trainset_config['users']:
            # Local training
            if config["client"]["fed_algo"] == 'FedAvg':
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss)
            elif config["client"]["fed_algo"] == 'SCAFFOLD':
                client_dict[client_id].update(global_state_dict, scv_state)
                state_dict, n_data, loss, delta_ccv_state = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss, delta_ccv_state)
            elif config["client"]["fed_algo"] == 'FedProx':
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss)
            elif config["client"]["fed_algo"] == 'FedNova':
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss, coeff, norm_grad = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss, coeff, norm_grad)

        # Global aggregation
        fed_server.select_clients()
        if config["client"]["fed_algo"] == 'FedAvg':
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == 'SCAFFOLD':
            global_state_dict, avg_loss, _, scv_state = fed_server.agg()  # scarffold
        elif config["client"]["fed_algo"] == 'FedProx':
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == 'FedNova':
            global_state_dict, avg_loss, _ = fed_server.agg()

        # Testing and flushing
        accuracy = fed_server.test()
        fed_server.flush()

        # Record the results
        recorder.res['server']['iid_accuracy'].append(accuracy)
        recorder.res['server']['train_loss'].append(avg_loss)

        if max_acc < accuracy:
            max_acc = accuracy
        pbar.set_description(
            'Global Round: %d' % global_round +
            '| Train loss: %.4f ' % avg_loss +
            '| Accuracy: %.4f' % accuracy +
            '| Max Acc: %.4f' % max_acc)

        # Save the results
        if not os.path.exists(config["system"]["res_root"]):
            os.makedirs(config["system"]["res_root"])

        with open(os.path.join(config["system"]["res_root"], '[\'%s\',' % config["client"]["fed_algo"] +
                                        '\'%s\',' % config["system"]["model"] +
                                        str(config["client"]["num_local_epoch"]) + ',' +
                                        str(config["system"]["num_local_class"]) + ',' +
                                        str(config["system"]["i_seed"])) + ']', "w") as jsfile:
            json.dump(recorder.res, jsfile, cls=PythonObjectEncoder)


if __name__ == "__main__":
    fed_run()
