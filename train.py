import time
import copy
import torch
import os
import pandas as pd
import tqdm
from torchmetrics.classification import MultilabelAccuracy, Accuracy
import torch.distributed as dist

# get current lr
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

# function to start training
def train_val(model, device, params):
    num_epochs=params['num_epochs']
    loss_func=params['loss_func']
    opt=params['optimizer']
    train_dl=params['train_dl']
    val_dl=params['val_dl']
    sanity_check=params['sanity_check']
    lr_scheduler=params['lr_scheduler']
    path2weights=params['path2weights']
    loss_mode = params['loss_mode']
    if loss_mode == 'multi':
        multimode = True
    else:
        multimode = False

    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}
    metric_cls_history = {'train': [], 'val': []}

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    

    for epoch in range(num_epochs):
        start_time = time.time()
        current_lr = get_lr(opt)

        print(f"Epoch {epoch}/{num_epochs-1}")

        model.train()
        if multimode:
            train_loss, train_metric,train_cls_metric = loss_epoch_multi_label(model, device, loss_func, train_dl, sanity_check, opt)
        else:
            train_loss, train_metric = loss_epoch(model, device, loss_func, train_dl, sanity_check, opt)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric.item())
        if multimode:
            metric_cls_history['train'].append(train_cls_metric)

        model.eval()
        with torch.no_grad():
            if multimode:
                val_loss, val_metric,val_cls_metric = loss_epoch_multi_label(model, device, loss_func, val_dl, sanity_check)
            else:
                val_loss, val_metric = loss_epoch(model, device, loss_func, val_dl, sanity_check)

        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric.item())
        if multimode:
            metric_cls_history['val'].append(val_cls_metric)

        if dist.is_initialized():
            if dist.get_rank() == 0:
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), path2weights + f'/{epoch}_weight.pth')
                print(f'Epoch {epoch}: Validation Loss: {val_loss}, Time: {(time.time() - start_time) / 60:.4f} min')
        else:
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(), path2weights + f'{epoch}_weight.pt')
                else:
                    torch.save(model.state_dict(), path2weights + f'{epoch}_weight.pt')
                
            print(f'Epoch {epoch}: Validation Loss: {val_loss}, Time: {(time.time() - start_time) / 60:.4f} min')

        lr_scheduler.step(val_loss)
        
        if current_lr != get_lr(opt):
            print('Loading best model weights!')
            model.load_state_dict(best_model_wts)
        if multimode:
            print(f'train loss: {train_loss:.6f}, val loss: {val_loss:.6f}, accuracy: {val_metric:.2f},cls_acc : {val_cls_metric}, time: {(time.time()-start_time)/60:.4f} min')
            lossdf = pd.DataFrame(loss_history)
            accdf = pd.DataFrame(metric_history)
            acc_clsdf = pd.DataFrame(metric_cls_history)

            lossdf.to_csv(path2weights + 'result/loss.csv')
            accdf.to_csv(path2weights + 'result/acc.csv')
            acc_clsdf.to_csv(path2weights + 'result/cls_acc.csv')
        else:
            print(f'train loss: {train_loss:.6f}, val loss: {val_loss:.6f}, accuracy: {val_metric:.2f}, time: {(time.time()-start_time)/60:.4f} min')
            lossdf = pd.DataFrame(loss_history)
            accdf = pd.DataFrame(metric_history)

            lossdf.to_csv(path2weights + 'result/loss.csv')
            accdf.to_csv(path2weights + 'result/acc.csv')
            metric_cls_history = None

    # model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history, metric_cls_history

def metric_batch_multi_label(output, target, device):
    # output: [batch_size, num_classes], target: [batch_size, num_classes]
    
    pred = output.sigmoid() >= 0.5
    
    num_classes = target.shape[1]
    mla_ova = MultilabelAccuracy(num_labels=num_classes).to(device=device)
    mla = MultilabelAccuracy(num_labels=num_classes, average=None).to(device=device)
    
    class_accuracies = mla(pred, target)
    overall_accuracy = mla_ova(pred, target)
    
    return class_accuracies, overall_accuracy


def loss_batch_multi_label(loss_func, output, target, device, opt=None):
    # output: [batch_size, num_classes], target: [batch_size, num_classes]
    loss_b = loss_func(output, target)
    class_metric_b , metric_b = metric_batch_multi_label(output, target, device)

    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()

    return loss_b.item(), metric_b, class_metric_b

def loss_epoch_multi_label(model, device, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    running_class_metrics = torch.zeros(dataset_dl.dataset.num_classes).to(device)
    num_classes = dataset_dl.dataset.num_classes
    b_count = 0
    with tqdm.tqdm(dataset_dl, unit="batch") as tepoch:
        for xb, yb in tepoch:
            b_count+=1
            xb = xb.to(device)
            yb = yb.to(device)
            output = model(xb)

            loss_b, metric_b, class_metric_b = loss_batch_multi_label(loss_func, output, yb, device, opt)

            running_loss += loss_b

            if metric_b is not None:
                running_metric += metric_b
            
            if class_metric_b is not None:
                running_class_metrics += class_metric_b

            if sanity_check is True:
                break
    if dist.is_initialized():
        loss_tensor = torch.tensor([running_loss], device=device)
        metric_tensor = torch.tensor([running_metric], device=device)
        class_metrics_tensor = running_class_metrics.clone()

        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(class_metrics_tensor, op=dist.ReduceOp.SUM)

        running_loss = loss_tensor.item() / dist.get_world_size()
        running_metric = metric_tensor.item() / dist.get_world_size()
        running_class_metrics = class_metrics_tensor / dist.get_world_size()

    loss = running_loss / b_count
    metric = running_metric / b_count # 수정된 부분
    class_metrics = {f'class_{i+1}': (running_class_metrics[i] / b_count).item() for i in range(num_classes)}
    return loss, metric, class_metrics

def loss_epoch(model, device, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    num_classes = dataset_dl.dataset.num_classes
    b_count = 0
    with tqdm.tqdm(dataset_dl, unit="batch") as tepoch:
        for xb, yb in tepoch:
            b_count+=1
            xb = xb.to(device)
            yb = yb.to(device)
            output = model(xb)
            

            loss_b, metric_b = loss_batch(loss_func, output, yb, device, num_classes, opt)

            running_loss += loss_b

            if metric_b is not None:
                running_metric += metric_b

            if sanity_check is True:
                break
    if dist.is_initialized():
        loss_tensor = torch.tensor([running_loss], device=device)
        metric_tensor = torch.tensor([running_metric], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        running_loss = loss_tensor.item() / dist.get_world_size()
        running_metric = metric_tensor.item() / dist.get_world_size()

    loss = running_loss / b_count
    metric = running_metric / b_count # 수정된 부분
    
    return loss, metric


def metric_batch(output, target, device,num_classes):
    if num_classes == 1:
        # 이진 분류 (Binary classification)
        output = torch.sigmoid(output)
        predicted = torch.round(output)
        acc = Accuracy(average='micro', task='binary').to(device)
    else:
        _, predicted = torch.max(output, 1)
        
        acc = Accuracy(average='micro', task='multiclass', num_classes=num_classes).to(device)
    acc.update(predicted, target)
    accuracy = acc.compute()
    
    return accuracy


def loss_batch(loss_func, output, target, device, num_classes, opt=None):
    # output: [batch_size, num_classes], target: [batch_size, num_classes]
    if num_classes == 1:
        assert output.shape == target.shape, "Output and target must have the same shape for BCEWithLogitsLoss"
        loss_b = loss_func(output, target)
    else:
        target = torch.argmax(target, dim=1)
        
        loss_b = loss_func(output, target)
    metric_b = metric_batch(output, target, device, num_classes)

    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()

    return loss_b.item(), metric_b


# check the directory to save weights.pt
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except os.OSerror:
        print('Error')
createFolder('./models')