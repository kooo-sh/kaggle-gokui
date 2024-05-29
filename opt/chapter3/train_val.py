import torchvision
import torch
from make_dataloader import setup_train_val_loaders
from tqdm import tqdm

# コード引用あり＠5節
def train_1epoch(model, train_loader, lossfun, optimizer, device):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for x, y in tqdm(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = lossfun(out, y)
        _, pred = torch.max(out.detach(), 1)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_acc += torch.sum(pred == y)

    avg_loss = total_loss / len(train_loader.dataset)
    avg_acc = total_acc / len(train_loader.dataset)
    return avg_acc, avg_loss


# コード引用あり＠5節
def validate_1epoch(model, val_loader, lossfun, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0

    with torch.no_grad():
        for x, y in tqdm(val_loader):
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            loss = lossfun(out.detach(), y)
            _, pred = torch.max(out, 1)

            total_loss += loss.item() * x.size(0)
            total_acc += torch.sum(pred == y)

    avg_loss = total_loss / len(val_loader.dataset)
    avg_acc = total_acc / len(val_loader.dataset)
    return avg_acc, avg_loss


def train(model, optimizer, train_loader, val_loader, n_epochs, device):
    lossfun = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(n_epochs)):
        train_acc, train_loss = train_1epoch(
            model, train_loader, lossfun, optimizer, device
        )
        val_acc, val_loss = validate_1epoch(model, val_loader, lossfun, device)
        print(
            f"epoch={epoch}, train loss={train_loss}, train accuracy={train_acc}, val loss={val_loss}, val accuracy={val_acc}"
        )

def train_subsec5(data_dir, batch_size, dryrun=False, device="mps"):
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_loader, val_loader = setup_train_val_loaders(
        data_dir, batch_size, dryrun
    )
    train(
        model, optimizer, train_loader, val_loader, n_epochs=1, device=device
    )

    return model