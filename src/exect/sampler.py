from torch.utils.data import Sampler

class SortedClusterSampler(Sampler):
    def __init__(self, cluster_list):
        # Sort the clusters while preserving the original indices
        sorted_indices = sorted(range(len(cluster_list)), key=lambda k: cluster_list[k])
        self.sorted_indices = sorted_indices

    def __iter__(self):
        return iter(self.sorted_indices)

    def __len__(self):
        return len(self.sorted_indices)