import torch
from .tensorlist import TensorList


class Memory:

    def __init__(self, capacity, feature_sizes, labels_size, device, learning_rates):
        """
        :param capacity:       Memory capacity - number of samples (int)
        :param feature_sizes:  {feature name: feature size shape(C,H,W)} of feature sizes,
                               C is the number of *compressed* channels
        :param labels_size:    (H,W) size of the labels image
        """
        self.samples = TensorList([torch.zeros(capacity, *fz).to(device) for fz in feature_sizes])
        self.weights = TensorList([torch.zeros(capacity).to(device) for fz in feature_sizes])
        self.labels = TensorList([torch.zeros(capacity, *labels_size).to(device) for fz in feature_sizes])
        self.pixel_weights = TensorList([torch.zeros(capacity, *labels_size).to(device) for fz in feature_sizes])

        self._capacity = capacity
        self.current_size = 0
        self.device = device
        self.previous_replace_ind = [None] * len(feature_sizes)
        self.learning_rates = learning_rates

    def initialize(self, init_features, init_labels, pixel_weights):

        K = init_features[0].shape[0]  # Number of samples
        assert init_labels.shape[0] == K

        for i, ft in enumerate(init_features):
            self.samples[i][:K] = ft.detach()
            self.weights[i][:K] = 1.0/K
            self.weights[i][0] = 2.0/K
            self.weights[i][:K] = self.weights[i][:K] / self.weights[i][:K].sum()
            self.labels[i][:K] = init_labels.float()
            self.pixel_weights[i][:K] = pixel_weights

        self.current_size = K

    def insert_at(self, positions: list, features, labels, pixel_weights):
        """ Insert a sample
        :param positions:  Per-feature memory locations
        :param features:   Feature maps to insert
        :param labels:     The labels image to insert
        :param weights:    Per-feature weights
        """
        for i, (p, ft) in enumerate(zip(positions, features)):
            self.samples[i][p] = ft.detach()
            self.labels[i][p] = labels
            self.pixel_weights[i][p] = pixel_weights

    def update(self, features, labels, pixel_weights):

        self.previous_replace_ind = self.update_sample_weights(self.previous_replace_ind)
        self.insert_at(self.previous_replace_ind, features, labels, pixel_weights)
        self.current_size = min(self.current_size + 1, self._capacity)

    def update_sample_weights(self, previous_replace_ind):
        # Update weights and get index to replace
        replace_ind = []
        num_samp = self.current_size

        for idx, (sw, lr, prev_ind) in enumerate(zip(self.weights, self.learning_rates, previous_replace_ind)):
            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                _, r_ind = torch.min(sw, 0)
                r_ind = r_ind.item()

                # Update weights
                if prev_ind is None:
                    sw /= (1 - lr)
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            replace_ind.append(r_ind)

        return replace_ind
