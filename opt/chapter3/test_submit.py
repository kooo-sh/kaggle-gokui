import torch
import os
from torchvision import transforms
import torchvision
from tqdm import tqdm
import numpy as np

# コード引用あり＠5節
def setup_center_crop_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

# コード引用あり＠5節
def setup_test_loader(data_dir, batch_size, dryrun):
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, "test"), transform=setup_center_crop_transform()
    )
    image_ids = [
        os.path.splitext(os.path.basename(path))[0] for path, _ in dataset.imgs
    ]

    if dryrun:
        dataset = torch.utils.data.Subset(dataset, range(0, 100))
        image_ids = image_ids[:100]

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=8
    )
    return loader, image_ids

# コード引用あり＠5節
def predict(model, loader, device):
    pred_fun = torch.nn.Softmax(dim=1)
    preds = []
    for x, _ in tqdm(loader):
        with torch.set_grad_enabled(False):
            x = x.to(device)
            y = pred_fun(model(x))
        y = y.cpu().numpy()
        y = y[:, 1]  # cat:0, dog: 1
        preds.append(y)
    preds = np.concatenate(preds)
    return preds


# コード引用あり＠5節
def write_prediction(image_ids, prediction, out_path):
    with open(out_path, "w") as f:
        f.write("id,label\n")
        for i, p in zip(image_ids, prediction):
            f.write("{},{}\n".format(i, p))

# コード引用あり＠5節
def predict_subsec5(
    data_dir, out_dir, model, batch_size, dryrun=False, device="mps"
):
    test_loader, image_ids = setup_test_loader(
        data_dir, batch_size, dryrun=dryrun
    )
    preds = predict(model, test_loader, device)
    write_prediction(image_ids, preds, out_dir + "/out.csv")