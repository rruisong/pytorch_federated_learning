#!/usr/bin/env python
import os
import random
import json
import pickle
import argparse
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

    parser.add_argument('-nc', '--sys-n_client', type=int, required=True, help='Number of the clients')
    parser.add_argument('-ck', '--sys-n_local_class', type=int, required=True, help='Number of the classes in each client')
    parser.add_argument('-ds', '--sys-dataset', type=str, required=True, help='Dataset name, one of the following four datasets: MNIST, CIFAR-10, FashionMnist, SVHN')
    parser.add_argument('-md', '--sys-model', type=str, required=True, help='Model name')
    parser.add_argument('-is', '--sys-i_seed', type=int, required=True, help='Seed used in experiments')
    parser.add_argument('-rr', '--sys-res_root', type=str, required=True, help='Root directory of the results')
    parser.add_argument('-nr', '--sys-n_round', type=int, required=True, help='Number of global communication rounds')
    parser.add_argument('-os', '--sys-oneshot', type=bool, default=False, help='Ture if only run with one-shot communication, otherwise false.')

    parser.add_argument('-cis', '--client-instance', type=str, required=True, help='Instance of federated learning algorithm used in clients')
    parser.add_argument('-cil', '--client-instance_lr', type=float, required=True, help='Learning rate in clients')
    parser.add_argument('-cib', '--client-instance_bs', type=int, required=True, help='Batch size in clients')
    parser.add_argument('-cie', '--client-instance_n_epoch', type=int, required=True, help='Number of local training epochs in clients')
    parser.add_argument('-sim', '--client-instance_momentum', type=float, required=True, help='Momentum of local training in clients')
    parser.add_argument('-sin', '--client-instance_n_worker', type=int, required=True, help='Number of workers in the server')

    args = parser.parse_args()
    return args


def fed_run():
    """
    Main function for the baselines of federated learning
    """
    args = fed_args()

    algo_list = ["FedAvg", "SCAFFOLD", "FedProx", "FedNova"]
    assert args.client_instance in algo_list, "The federated learning algorithm is not supported"

    dataset_list = ['MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN', 'CIFAR100']
    assert args.sys_dataset in dataset_list, "The dataset is not supported"

    model_list = ["LeNet", 'AlexCifarNet', "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "CNN"]
    assert args.sys_model in model_list, "The model is not supported"

    np.random.seed(args.sys_i_seed)
    torch.manual_seed(args.sys_i_seed)
    random.seed(args.sys_i_seed)

    client_dict = {}
    recorder = Recorder()

    trainset_config, testset = divide_data(num_client=args.sys_n_client, num_local_class=args.sys_n_local_class, dataset_name=args.sys_dataset,
                                           i_seed=args.sys_i_seed)
    max_acc = 0
    # Initialize the clients w.r.t. the federated learning algorithms and the specific federated settings
    for client_id in trainset_config['users']:
        if args.client_instance == 'FedAvg':
            client_dict[client_id] = FedClient(client_id, dataset_id=args.sys_dataset, epoch=args.client_instance_n_epoch, model_name=args.sys_model)
        elif args.client_instance == 'SCAFFOLD':
            client_dict[client_id] = ScaffoldClient(client_id, dataset_id=args.sys_dataset, epoch=args.client_instance_n_epoch, model_name=args.sys_model)
        elif args.client_instance == 'FedProx':
            client_dict[client_id] = FedProxClient(client_id, dataset_id=args.sys_dataset, epoch=args.client_instance_n_epoch, model_name=args.sys_model)
        elif args.client_instance == 'FedNova':
            client_dict[client_id] = FedNovaClient(client_id, dataset_id=args.sys_dataset, epoch=args.client_instance_n_epoch, model_name=args.sys_model)
        client_dict[client_id].load_trainset(trainset_config['user_data'][client_id])

    # Initialize the clients w.r.t. the federated learning algorithms and the specific federated settings
    if args.client_instance == 'FedAvg':
        fed_server = FedServer(trainset_config['users'], dataset_id=args.sys_dataset, model_name=args.sys_model)
    elif args.client_instance == 'SCAFFOLD':
        fed_server = ScaffoldServer(trainset_config['users'], dataset_id=args.sys_dataset, model_name=args.sys_model)
        scv_state = fed_server.scv.state_dict()
    elif args.client_instance == 'FedProx':
        fed_server = FedServer(trainset_config['users'], dataset_id=args.sys_dataset, model_name=args.sys_model)
    elif args.client_instance == 'FedNova':
        fed_server = FedNovaServer(trainset_config['users'], dataset_id=args.sys_dataset, model_name=args.sys_model)
    fed_server.load_testset(testset)
    global_state_dict = fed_server.state_dict()

    # Main process of federated learning in multiple communication rounds
    if args.sys_oneshot:
        n_round = 1
    else:
        n_round = args.sys_n_round
    pbar = tqdm(range(n_round))
    for global_round in pbar:
        for client_id in trainset_config['users']:
            # Local training
            if args.client_instance == 'FedAvg':
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss)
            elif args.client_instance == 'SCAFFOLD':
                client_dict[client_id].update(global_state_dict, scv_state)
                state_dict, n_data, loss, delta_ccv_state = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss, delta_ccv_state)
            elif args.client_instance == 'FedProx':
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss)
            elif args.client_instance == 'FedNova':
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss, coeff, norm_grad = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss, coeff, norm_grad)

        # Global aggregation
        fed_server.select_clients()
        if args.client_instance == 'FedAvg':
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif args.client_instance == 'SCAFFOLD':
            global_state_dict, avg_loss, _, scv_state = fed_server.agg()  # scarffold
        elif args.client_instance == 'FedProx':
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif args.client_instance == 'FedNova':
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
        if not os.path.exists(args.sys_res_root):
            os.makedirs(args.sys_res_root)

        with open(os.path.join(args.sys_res_root, '[\'%s\',' % args.client_instance +
                                        '\'%s\',' % args.sys_model +
                                        str(args.client_instance_n_epoch) + ',' +
                                        str(args.sys_n_local_class) + ',' +
                                        str(args.sys_i_seed)) + ']', "w") as jsfile:
            json.dump(recorder.res, jsfile, cls=PythonObjectEncoder)


if __name__ == "__main__":
    fed_run()
