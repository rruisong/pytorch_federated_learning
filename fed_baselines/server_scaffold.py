from utils.fed_utils import init_model
from fed_baselines.server_base import FedServer
import copy


class ScaffoldServer(FedServer):
    def __init__(self, client_list, dataset_id, model_name):
        super().__init__(client_list, dataset_id, model_name)
        # server control variate
        self.scv = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        # Dict of all client control variates
        self.client_ccv_state = {}

    def agg(self):
        """
        Server aggregates normalized models from connected clients using SCAFFOLD
        :return: Updated global model after aggregation, Averaged loss value, Number of the local data points, server control variate
        """
        client_num = len(self.selected_clients)
        if client_num == 0 or self.n_data == 0:
            return self.model.state_dict(), 0, 0

        self.scv.to(self._device)
        self.model.to(self._device)
        model_state = self.model.state_dict()
        new_scv_state = copy.deepcopy(model_state)
        avg_loss = 0
        # print('number of selected clients in Cloud: ' + str(client_num))
        for i, name in enumerate(self.selected_clients):
            if name not in self.client_state:
                continue
            for key in self.client_state[name]:

                if i == 0:
                    model_state[key] = self.client_state[name][key] * self.client_n_data[name] / self.n_data
                    new_scv_state[key] = self.client_ccv_state[name][key] * self.client_n_data[name] / self.n_data

                else:
                    model_state[key] = model_state[key] + self.client_state[name][key] * self.client_n_data[
                        name] / self.n_data
                    new_scv_state[key] = new_scv_state[key] + self.client_ccv_state[name][key] * self.client_n_data[
                        name] / self.n_data

            avg_loss = avg_loss + self.client_loss[name] * self.client_n_data[name] / self.n_data

        scv_state = self.scv.state_dict()

        self.model.load_state_dict(model_state)
        self.scv.load_state_dict(new_scv_state)
        self.round = self.round + 1

        return model_state, avg_loss, self.n_data, scv_state

    def rec(self, name, state_dict, n_data, loss, ccv_state):
        """
        Server receives the local updates from the connected client k.
        :param name: Name of client k
        :param state_dict: Model dict from the client k
        :param n_data: Number of local data points in the client k
        :param loss: Loss of local training in the client k
        :param ccv_state: Normalization coefficient
        """
        self.n_data = self.n_data + n_data
        self.client_state[name] = {}
        self.client_n_data[name] = {}
        self.client_ccv_state[name] = {}

        self.client_state[name].update(state_dict)
        self.client_n_data[name] = n_data
        self.client_loss[name] = {}
        self.client_loss[name] = loss
        self.client_ccv_state[name].update(ccv_state)


    def flush(self):
        """
        Flushing the client information in the server
        """
        self.n_data = 0
        self.client_state = {}
        self.client_n_data = {}
        self.client_loss = {}
        self.client_ccv_state = {}
