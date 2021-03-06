import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class Train():
    def __init__(self, epochs=2, batch_size=1, log_interval=1):
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.device = self.set_device()
        self.num_workers = self.set_num_workers()
        self.pin_memory = self.set_pin_memory()

    def set_device(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        return self.device

    def set_num_workers(self):
        if self.device == "cuda":
            return 1
        else:
            return 0

    def set_pin_memory(self):
        if self.device == "cuda":
            return True
        else:
            return False

    def number_of_correct(self, pred, y):
        # count number of correct predictions
        return pred.squeeze().eq(y).sum().item()

    def get_likely_index(self, tensor):
        # find most likely label index for each element in the batch
        return tensor.argmax(dim=-1)

    def train_epoch(self, model, train_loader, optimizer):
        model.architecture.train()

        criterions = []
        correct_sum = 0

        pbar = tqdm(total=len(train_loader.dataset))

        for batch_idx, (X, y) in enumerate(train_loader):

            X = X.to(self.device)
            y = y.to(self.device)

            # apply model on whole batch directly on device
            output = model.architecture(X)

            # count number of correct predictions
            pred = self.get_likely_index(output)
            correct = self.number_of_correct(pred, y)
            correct_sum += correct
            #accuracy.append(correct/len(X))

            # negative log-likelihood for a tensor of size (batch x 1 x n_output)
            criterion = model.criterion(output.squeeze(), y)

            optimizer.zero_grad()
            criterion.backward()
            optimizer.step()

            # record criterion
            criterions.append(criterion.item())

            # print training stats
            pbar.update(len(X))
            if batch_idx % self.log_interval == 0:
                s = "-- TRAIN Loss: {criterion:.4f}, Accuracy: {perc_correct:.1f}%"
                d = {
                    'criterion': criterion.item(),
                    'perc_correct': 100. * correct / len(X)
                }
                pbar.set_description(s.format(**d))

        pbar.close()

        accuracy = correct_sum/len(train_loader.dataset)

        return np.mean(criterions), accuracy

    def test_epoch(self, model, test_loader):
        # put the model in the evaluation mode
        model.architecture.eval()

        correct_sum = 0
        criterion = 0
        with torch.no_grad():  # stops autograd engine from calculating the gradients
            for X, y in test_loader:

                X = X.to(self.device)
                y = y.to(self.device)

                # apply model on whole batch directly on device
                output = model.architecture(X)

                # count number of correct predictions
                pred = self.get_likely_index(output)
                correct_sum += self.number_of_correct(pred, y)

                # save criterions
                criterion += model.criterion(output.squeeze(), y)/len(test_loader)

        accuracy = correct_sum/len(test_loader.dataset)

        s = "-- TEST  Loss: {criterion:.4f}, Accuracy: {perc_correct:.1f}%"
        d = {
            'criterion': criterion.item(),
            'perc_correct': 100. * accuracy
        }
        print(s.format(**d))

        return criterion.item(), accuracy

    def earlystop(self, model, criterion):
        earlystop = False
        # check if earlystopping class exists
        if model.earlystopping:
            # update earlystopping parameters
            model.earlystopping(criterion, model.architecture)
            # check if to stop
            if model.earlystopping.early_stop:
                # load best model at the end of training
                model.architecture.load_state_dict(torch.load(model.earlystopping.checkpoint_path)) 
                print("Early stopping")
                # force training stop
                earlystop = True
        return earlystop

    def train(self, model, train_set, test_set):
        # associate the architecture parameters to the chosen optimizer class
        optimizer = model.optimizer(model.architecture.parameters())
        # if scheduler exits associate it to the optimizer
        if model.scheduler:
            scheduler = model.scheduler(optimizer)
        # send architecture to device
        model.architecture.to(self.device)
        # set dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        train_criterion = []
        train_accuracy = []
        test_criterion = []
        test_accuracy = []

        for epoch in range(1, self.epochs + 1):
            print(f'Epoch: {epoch}')
            # append train metrics
            criterion, accuracy = self.train_epoch(model, train_loader, optimizer)
            train_criterion.append(criterion)
            train_accuracy.append(accuracy)
            # append test metrics
            criterion, accuracy = self.test_epoch(model, test_loader)
            test_criterion.append(criterion)
            test_accuracy.append(accuracy)
            # update learning rate
            if model.scheduler:
                scheduler.step()

            # check early stopping
            if self.earlystop(model, criterion):  #change criterion to change metric to check
                break

        columns = ['train_criterion', 'train_accuracy', 'test_criterion', 'test_accuracy']
        df = pd.DataFrame(
            np.array([train_criterion, train_accuracy, test_criterion, test_accuracy]).T,
            index = np.arange(1, len(train_criterion)+1),
            columns=columns
            )

        return df

    def __call__(self, model, train_set, test_set):
        return self.train(model, train_set, test_set)


# EXAMPLE
'''
T = Train()
print(T.set_device())
'''