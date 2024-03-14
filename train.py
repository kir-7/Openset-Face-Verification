import torch
import numpy as np
import time 


from models.resnet import get_pretained_resnet
from models.metrics import ArcFace

from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from utils import create_dataloader


class Logger(object):

    def __init__(self, mode, length, calculate_mean=False):
        self.mode = mode
        self.length = length
        self.calculate_mean = calculate_mean
        if self.calculate_mean:
            self.fn = lambda x, i: x / (i + 1)
        else:
            self.fn = lambda x, i: x

    def __call__(self, loss, metrics, i):
        track_str = '\r{} | {:5d}/{:<5d}| '.format(self.mode, i + 1, self.length)
        loss_str = 'loss: {:9.4f} | '.format(self.fn(loss, i))
        metric_str = ' | '.join('{}: {:9.4f}'.format(k, self.fn(v, i)) for k, v in metrics.items())
        print(track_str + loss_str + metric_str + '   ', end='')
        if i + 1 == self.length:
            print('')


class BatchTimer(object):
    """Batch timing class.
    Use this class for tracking training and testing time/rate per batch or per sample.
    
    Keyword Arguments:
        rate {bool} -- Whether to report a rate (batches or samples per second) or a time (seconds
            per batch or sample). (default: {True})
        per_sample {bool} -- Whether to report times or rates per sample or per batch.
            (default: {True})
    """

    def __init__(self, rate=True, per_sample=True):
        self.start = time.time()
        self.end = None
        self.rate = rate
        self.per_sample = per_sample

    def __call__(self, y_pred, y):
        self.end = time.time()
        elapsed = self.end - self.start
        self.start = self.end
        self.end = None

        if self.per_sample:
            elapsed /= len(y_pred)
        if self.rate:
            elapsed = 1 / elapsed

        return torch.tensor(elapsed)
    

def accuracy(logits, y):
    _, preds = torch.max(logits, 1)
    return (preds == y).float().mean()


def pass_epoch(model, metric, loss_fn, loader, optimizer=None, scheduler=None, batch_metrics={'time': BatchTimer()}, show_running=True,
                device='cpu', writer=None):
    
    mode = "Train" if model.training else "Valid"

    logger = Logger(mode, length=len(loader), calculate_mean=show_running)

    loss = 0
    metrics = {}

    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        features = model(x)
        logits = metric(features, y)
        
        output = loss_fn(logits, y)

        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        

        metrics_batch = {}
        for metric_name, metric_fn in batch_metrics.items():
            metrics_batch[metric_name] = metric_fn(logits, y).detach().cpu()
            metrics[metric_name] = metrics.get(metric_name, 0) + metrics_batch[metric_name]

        if writer is not None and model.training:
            if writer.iteration % writer.interval == 0:
                writer.add_scalars('loss', {mode: loss_batch.detach().cpu()}, writer.iteration)
                for metric_name, metric_batch in metrics_batch.items():
                    writer.add_scalars(metric_name, {mode: metric_batch}, writer.iteration)
            writer.iteration += 1

        loss_batch = loss_batch.detach().cpu()
        loss += loss_batch
        if show_running:
            logger(loss, metrics, i)
        else:
            logger(loss_batch, metrics_batch, i)

        
    if model.training and scheduler is not None:
        scheduler.step()
    
    loss = loss/(i+1)
    metrics =  {k: v / (i + 1) for k, v in metrics.items()}

    if writer is not None and not model.training:
        writer.add_scalars('loss', {mode: loss.detach()}, writer.iteration)

        for metric_name, metric in metrics.items():
            writer.add_scalars(metric_name, {mode: metric})

    return loss, metrics


def train(model, train_loader, val_loader, n_classes, emb_dim, epochs, device, show_running=True, save_model=None):


    if model is None:
        print("Failed to load pre Trained model!")  
        return
    
    loss_fn = nn.CrossEntropy()

    adam = optim.Adma(model.parameters(), lr=0.001)
    scheduler = MultiStepLR(adam, [5, 10])    

    metrics = metrics = {
                    'fps': BatchTimer(),
                    'acc': accuracy
                    }

    arc = ArcFace(n_classes, emb_dim)


    writer = SummaryWriter()
    writer.iteration, writer.interval = 0, 10

    print("\n\nInitial")
    print("-"*10)

    model.eval()


    pass_epoch(model, arc, loss_fn, val_loader, adam, scheduler,batch_metrics=metrics, show_running=True, device=device,writer=writer)

    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        model.train()
        pass_epoch(
            model, loss_fn, train_loader, adam, scheduler,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )


        model.eval()
        pass_epoch(
            model, loss_fn, val_loader,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

    writer.close()

    if save_model is not None:
        torch.save(model, save_model)
