from fed_baselines.client_base import FedClient
import copy
from utils.models import *

from torch.utils.data import DataLoader


class FedProxClient(FedClient):
    def __init__(self, name, epoch, dataset_id, model_name):
        super().__init__(name, epoch, dataset_id, model_name)
        self.mu = 0.1

    def train(self):
        """
        Client trains the model on local dataset using FedProx
        :return: Local updated model, number of local data points, training loss
        """
        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True)

        self.model.to(self._device)
        global_weights = copy.deepcopy(list(self.model.parameters()))

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=1e-4)
        loss_func = nn.CrossEntropyLoss()

        epoch_loss_collector = []

        # pbar = tqdm(range(self._epoch))
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

                    #fedprox
                    prox_term = 0.0
                    for p_i, param in enumerate(self.model.parameters()):
                        prox_term += (self.mu / 2) * torch.norm((param - global_weights[p_i])) ** 2
                    loss += prox_term
                    epoch_loss_collector.append(loss.item())

                    loss.backward()
                    optimizer.step()



        return self.model.state_dict(), self.n_data, loss.data.cpu().numpy()