# PyTorch Implementation of Federated Learning Baselines

PyTorch-Federated-Learning provides various federated learning baselines implemented using the PyTorch framework. The codebase follows a client-server architecture and is highly intuitive and accessible.

If you find this repository useful, please let me know with your stars:star:. Thank you!

* **Current Baseline implementations**: Pytorch implementations of the federated learning baselines. The currently supported baselines are FedAvg, FedNova, FedProx and SCAFFOLD
* **Dataset preprocessing**: Downloading the benchmark datasets automatically and dividing them into a number of clients w.r.t. federated settings. The currently supported datasets are MNIST, Fashion-MNIST, SVHN, CIFAR-10, CIFAR-100. Other datasets need to be downloaded manually.
* **Postprocessing**: Visualization of the training results for evaluation.


## Installation

### Dependencies

 - Python (3.8)
 - PyTorch (1.8.1)
 - OpenCV (4.5)
 - numpy (1.21.5)


### Install requirements

Run: `pip install -r requirements.txt` to install the required packages.

## Federated Dataset Preprocessing

This preprocessing aims to divide the entire datasets into a dedicated number of clients with respect to federated settings.
Depending on the the number of classes in each local dataset, the entire dataset are split into Non-IID datasets in terms of label distribution skew.


## Execute the Federated Learning Baselines

### Test Run
Hyperparameters are defined in a yaml file, e.g. "./config/test_config.yaml", and then just run with this configuration:

```
python fl_main.py --config "./config/test_config.yaml"
```


## Evaluation Procedures

Please run `python postprocessing/eval_main.py -rr 'results'` to plot the testing accuracy and training loss by the increasing number of epochs or communication rounds. 
Note that the labels in the figure is the name of result files

