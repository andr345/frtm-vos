import torch


class Memory:

    def __init__(self, capacity, feature_size, labels_size, device, learning_rates):
        """
        :param capacity:       Memory capacity - number of samples (int)
        :param feature_sizes:  {feature name: feature size shape(C,H,W)} of feature sizes,
                               C is the number of *compressed* channels
        :param labels_size:    (H,W) size of the labels image
        """
        self.samples = torch.zeros(capacity, *feature_size, device=device)
        self.weights = torch.zeros(capacity, device=device)
        self.labels = torch.zeros(capacity, *labels_size, device=device)
        self.pixel_weights = torch.zeros(capacity, *labels_size, device=device)

        self._capacity = capacity
        self.current_size = 0
        self.device = device
        self.previous_replace_ind = None
        self.learning_rates = learning_rates

    @property
    def capacity(self):
        return self._capacity

    def clear(self):
        self.current_size = 0
        for w in self.weights:
            w.zero_()

    def initialize(self, init_features, init_labels, pixel_weights):

        K = init_features.shape[0]  # Number of samples
        assert init_labels.shape[0] == K

        self.samples[:K] = init_features.detach()
        self.weights[:K] = 1.0 / K
        self.weights[0] = 2.0 / K
        self.weights[:K] = self.weights[:K] / self.weights[:K].sum()

        self.labels[:K] = init_labels.float()
        self.pixel_weights[:K] = pixel_weights

        self.current_size = K

    def insert_at(self, p, ft, labels, pixel_weights):
        """ Insert a sample
        :param p:  Per-feature memory locations
        :param ft:   Feature maps to insert
        :param labels:     The labels image to insert
        :param weights:    Per-feature weights
        """
        self.samples[p] = ft.detach()
        self.labels[p] = labels
        self.pixel_weights[p] = pixel_weights

    def update(self, features, labels, pixel_weights):

        self.previous_replace_ind = self.update_sample_weights(self.previous_replace_ind)
        self.insert_at(self.previous_replace_ind, features, labels, pixel_weights)
        self.current_size = min(self.current_size + 1, self._capacity)

    def update_sample_weights(self, previous_replace_ind):
        # Update weights and get index to replace

        num_samp = self.current_size

        sw = self.weights
        lr = self.learning_rates
        prev_ind = previous_replace_ind

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

        return r_ind
