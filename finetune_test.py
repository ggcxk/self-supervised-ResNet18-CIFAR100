
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_cifar100_data_loaders(download, shuffle=False, batch_size=256):
    train_dataset = datasets.CIFAR100('.', train=True, download=download,
                                    transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                num_workers=0, drop_last=False, shuffle=shuffle)
    
    test_dataset = datasets.CIFAR100('.', train=False, download=download,
                                    transform=transforms.ToTensor())


    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                                num_workers=2, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader


def accuracy(output, target, topk=(1,)):
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# Create model
model = torchvision.models.resnet18(pretrained=False, num_classes=10)

# freeze all layers except the last
for name, param in model.named_parameters():
    if name not in ['fc.weight', 'fc.bias']:
        param.requires_grad = False
# init the fc layer
num_classes = 100
model.fc = torch.nn.Linear(512,num_classes)
model.fc.weight.data.normal_(mean=0.0, std=0.01)
model.fc.bias.data.zero_()

# loading the trained check point data
checkpoint = torch.load('./checkpoint/checkpoint_0020.pth.tar', map_location=device)

# rename moco pre-trained keys
state_dict = checkpoint['state_dict']
for k in list(state_dict.keys()):
    if k.startswith('backbone.'):
        if k.startswith('backbone') and not k.startswith('backbone.fc'):
            # remove prefix
            state_dict[k[len("backbone."):]] = state_dict[k]
    del state_dict[k]
log = model.load_state_dict(state_dict, strict=False)
assert log.missing_keys == ['fc.weight', 'fc.bias']

model = model.to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
# optimize only the linear classifier
parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
assert len(parameters) == 2
optimizer = torch.optim.Adam(parameters, lr=0.01, weight_decay=0.0008)


cifar100_train_loader, cifar100_test_loader = get_cifar100_data_loaders(download=True)

# supervise learning on CIFAR100 dataset

epochs = 100
top1_train_accuracy_list = [0]
top1_accuracy_list = [0]
top5_accuracy_list = [0]
epoch_list = [0]

for epoch in range(epochs):
    top1_train_accuracy = 0
    for counter, (x_batch, y_batch) in enumerate(cifar100_train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        top1 = accuracy(logits, y_batch, topk=(1,))
        top1_train_accuracy += top1[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    top1_train_accuracy /= (counter + 1)
    top1_accuracy = 0
    top5_accuracy = 0
    for counter, (x_batch, y_batch) in enumerate(cifar100_test_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
    
        top1, top5 = accuracy(logits, y_batch, topk=(1,5))
        top1_accuracy += top1[0]
        top5_accuracy += top5[0]
    
    top1_accuracy /= (counter + 1)
    top5_accuracy /= (counter + 1)

    top1_train_accuracy_list.append(top1_train_accuracy.item())
    top1_accuracy_list.append(top1_accuracy.item())
    top5_accuracy_list.append(top5_accuracy.item())
    epoch_list.append(epoch+1)
    print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
    

# plot the result

top1_train_accuracy_list.pop(0)
top1_accuracy_list.pop(0)
top5_accuracy_list.pop(0)
epoch_list.pop(0)

plt.figure(figsize = (16, 9))
plt.rcParams.update({'font.size': 14})
plt.title('CIFAR100 Accuracy Plot')
plt.plot(epoch_list,top1_train_accuracy_list, c='b')
plt.plot(epoch_list,top1_accuracy_list, c='g')
plt.plot(epoch_list,top5_accuracy_list, c='r')
plt.legend(['Training Accuracy', 'Top 1 Test Accuracy', 'Top 5 Test Accuracy'])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show() 





