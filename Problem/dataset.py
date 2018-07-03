import torch.utils.data as data


class AdultDataset(data.Dataset):

    def __init__(self, X, y):

        ######

        # 3.1 YOUR CODE HERE

        ######

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        ######

        # 3.1 YOUR CODE HERE

        ######