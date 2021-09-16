class trainer:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg

    def train(self, data_iter, cfg, optimizer, device):
        if self.cfg.uda_mode:
            pass
        else:
            raise NotImplementedError

        pass

    def load(self):
        pass

    def save(self):
        pass

