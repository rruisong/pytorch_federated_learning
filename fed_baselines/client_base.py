from utils.models import *
from torch.utils.data import DataLoader
from utils.fed_utils import assign_dataset, init_model


class FedClient(object):
    def __init__(self, name, epoch, dataset_id, model_name):
        """
        Initialize the client k for federated learning.
        :param name: Name of the client k
        :param epoch: Number of local training epochs in the client k
        :param dataset_id: Local dataset in the client k
        :param model_name: Local model in the client k
        """
        # Initialize the metadata in the local client
        self.target_ip = '127.0.0.3'
        self.port = 9999
        self.name = name

        # Initialize the parameters in the local client
        self._epoch = epoch
        self._batch_size = 50
        self._lr = 0.001
        self._momentum = 0.9
        self.num_workers = 2
        self.loss_rec = []
        self.n_data = 0

        # Initialize the local training and testing dataset
        self.trainset = None
        self.test_data = None

        # Initialize the local model
        self._num_class, self._image_dim, self._image_channel = assign_dataset(dataset_id)
        self.model_name = model_name
        self.model = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.param_len = sum([np.prod(p.size()) for p in model_parameters])

        # Training on GPU
        gpu = 0
        self._device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    def load_trainset(self, trainset):
        """
        Client loads the training dataset.
        :param trainset: Dataset for training.
        """
        self.trainset = trainset
        self.n_data = len(trainset)

    def update(self, model_state_dict):
        """
        Client updates the model from the server.
        :param model_state_dict: Global model.
        """
        self.model = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        self.model.load_state_dict(model_state_dict)

    def train(self):
        """
        Client trains the model on local dataset
        :return: Local updated model, number of local data points, training loss
        """
        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True)

        self.model.to(self._device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=1e-4)
        loss_func = nn.CrossEntropyLoss()

        # Training process
        for epoch in range(self._epoch):
            for step, (x, y) in enumerate(train_loader):
                with torch.no_grad():
                    b_x = x.to(self._device)  # Tensor on GPU
                    b_y = y.to(self._device)  # Tensor on GPU

                with torch.enable_grad():
                    self.model.train()
                    output = self.model(b_x)
                    loss = loss_func(output, b_y.long())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        return self.model.state_dict(), self.n_data, loss.data.cpu().numpy()
