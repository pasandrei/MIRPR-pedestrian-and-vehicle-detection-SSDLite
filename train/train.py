from train.loss_fn import ssd_loss
from train.helpers import *
from train.validate import evaluate


def train(model, optimizer, train_loader, valid_loader,
          device, params):
    '''
    args: model - nn.Module CNN to train
          optimizer - torch.optim
          params - json config
    trains model, saves best model by validation
    '''
    for epoch in range(params.n_epochs):

        print('Train start...')

        model.train()
        batch_loss = 0.0
        epoch_loss = 0.0

        for batch_idx, (input_, label) in enumerate(train_loader):
            input_ = input_.to(device)
            label[0], label[1] = label[0].to(device), label[1].to(device)

            optimizer.zero_grad()

            output = model(input_)

            output[0].to(device)
            output[1].to(device)

            loss = ssd_loss(output, label, device)
            batch_loss += loss.item()
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % params.train_stats_step == 0:
                print_batch_stats(epoch, batch_idx, train_loader, batch_loss, params)
                batch_loss = 0

        if (epoch + 1) % params.eval_step == 0:
            evaluate(model, optimizer, valid_loader, epoch_loss, epoch, device, params)
            epoch_loss = 0
        break
