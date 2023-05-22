import torch

#from train###############################################################################################
def metric_batch_multi_label(output, target):
    # output: [batch_size, num_classes], target: [batch_size, num_classes]
    pred = output.sigmoid() >= 0.5 # 이진분류: sigmoid 함수를 사용하여 확률 값을 0~1 사이로 만들고, 0.5 이상인 경우 1, 미만인 경우 0으로 예측합니다.
    corrects = pred.eq(target).sum().item() # 정답과 예측값이 일치하는 개수를 계산합니다.
    total = target.size(0)*target.size(1)
    return corrects / total


def loss_batch_multi_label(loss_func, output, target, opt=None):
    # output: [batch_size, num_classes], target: [batch_size, num_classes]
    loss_b = loss_func(output, target)
    metric_b = metric_batch_multi_label(output, target)

    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()

    return loss_b.item(), metric_b
#from train###############################################################################################


def eval_test_set(model, device, dataloader, loss_func):
    # 모델 파라미터 불러오기
    model.load_state_dict(torch.load('model_weights.pt'))

    # 모델 평가하기
    model.eval() # 모델을 평가 모드로 설정

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)

            output = model(xb)

            loss_b, metric_b = loss_batch_multi_label(loss_func, output, yb)

            running_loss += loss_b

            if metric_b is not None:
                running_metric += metric_b

    # 평가 결과 출력
    test_loss = running_loss / len(dataloader.dataset)
    test_metric = running_metric / (len(dataloader.dataset) * yb.size(1))

    return test_loss, test_metric
