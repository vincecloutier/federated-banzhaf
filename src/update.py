import torch
from torch import nn, autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset

from utils import get_device, initialize_model


class ClientSplit(Dataset):
    """A PyTorch Dataset wrapper for creating a subset based on specific indices."""
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class LocalUpdate:
    """Handles local training for a subset of data in federated learning."""
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.device = get_device()
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        client_dataset = ClientSplit(dataset, idxs)
        self.trainloader = DataLoader(client_dataset, batch_size=self.args.local_bs, shuffle=True)

    def update_weights(self, model, global_round):
        """Performs local training on the model for a specified number of epochs."""
        model.train()
        scaler = GradScaler()
        epoch_loss = []

        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.args.lr, epochs=self.args.local_ep, steps_per_epoch=len(self.trainloader))

        for local_epoch in range(self.args.local_ep):
            batch_loss = []

            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                with autocast(device_type='cuda', dtype=torch.float16):
                    output = model(images)
                    loss = self.criterion(output, labels)

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

                sched.step()
                batch_loss.append(loss.item())

            avg_batch_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(avg_batch_loss)

            print(f"| Global Round : {global_round + 1} | Local Epoch : {local_epoch + 1} | Loss: {avg_batch_loss:.6f}")

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


def train_client(idx, args, global_weights, train_dataset, user_groups, epoch, device = None):
    """Multiprocessing helper function."""
    if device is None:
        device = get_device()
        
    model = initialize_model(args)
    model.load_state_dict(global_weights)
    model.to(device)
    model.train()

    local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
    w, _ = local_model.update_weights(model=model, global_round=epoch)
    delta = {key: (global_weights[key] - w[key]).to(device) for key in global_weights.keys()}

    del local_model, model
    torch.cuda.empty_cache()

    return idx, w, delta


def test_inference(model, test_dataset):
    """Returns the test accuracy and loss on the global model trained on the entire dataset."""
    model.eval()
    device = get_device()
    criterion = nn.CrossEntropyLoss().to(device)

    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    total_loss, total, correct = 0.0, 0.0, 0.0

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        total_loss += batch_loss.item()

        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy, total_loss


def gradient(args, model, dataset):
    """Computes the gradient of the loss with respect to the model parameters for a given dataset."""
    model.eval()
    model.zero_grad()
    device = get_device()

    criterion = nn.CrossEntropyLoss().to(device)
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    inputs, targets = next(iter(data_loader))
    inputs, targets = inputs.to(device), targets.to(device)

    outputs = model(inputs)
    loss = criterion(outputs, targets)

    grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=False, retain_graph=False)

    grad_dict = {}
    for (name, param), grad in zip(model.named_parameters(), grad_params):
        if param.requires_grad:
            grad_dict[name] = grad.clone().detach()

    del inputs, targets, outputs, loss, grad_params
    torch.cuda.empty_cache()

    return grad_dict


def compute_hessian(model, dataset, v_list):
    """Computes the Hessian-vector product (HVP) Hv by averaging over batches."""
    model.eval()
    device = get_device()
    criterion = nn.CrossEntropyLoss().to(device)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False)

    params = [p for p in model.parameters() if p.requires_grad]
    hvp_total = [torch.zeros_like(p) for p in params]

    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        grad_params = torch.autograd.grad(loss, params, create_graph=True)
        grad_params_flat = torch.cat([g.contiguous().view(-1) for g in grad_params])
        v_flat = torch.cat([v.contiguous().view(-1) for v in v_list])

        grad_dot_v = torch.dot(grad_params_flat, v_flat)
        hvp = torch.autograd.grad(grad_dot_v, params)

        for i, hv in enumerate(hvp):
            hvp_total[i] += hv.detach()

        model.zero_grad()

    hvp_avg = [hv / len(data_loader) for hv in hvp_total]
    hv_dict = {name: hv.clone().detach() for (name, param), hv in zip(model.named_parameters(), hvp_avg) if param.requires_grad}

    del hvp_total, hvp_avg, hvp
    torch.cuda.empty_cache()

    return hv_dict