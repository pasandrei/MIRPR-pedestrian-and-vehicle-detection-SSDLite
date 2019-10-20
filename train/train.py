from train import helpers


def train(model, train_loader, valid_loader, test_loader,
          device, params):
    
    for epoch in range(epochs):
        
        model.train()
        
        for batch_idx, samples in enumerate(train_loader):             
            input_, label = samples['image'].to(device), samples['label'].to(device)

            optimizer.zero_grad()

            output = model(input_)
            loss = criterion(output,label)
            total_loss += loss
            loss.backward()
            optimizer.step()
            
            if (batch_idx + 1) % params.train_stats_step == 0:
                helpers.print_stats()
            
        if (epoch + 1) % params.eval_step == 0:
            helpers.evaluate()

    writer.close()      