import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
from optims.c_flat import C_Flat

EPSILON = 1e-8
num_workers = 8


class Replay(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        # Loader
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args['batch_size'], shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args['batch_size'], shuffle=False, num_workers=num_workers
        )

        # Procedure
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)

        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.args['init_lr'],
                weight_decay=self.args['init_weight_decay'],
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args['init_milestones'], gamma=self.args['init_lr_decay']
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            base_optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.args['lrate'],
                momentum=0.9,
                weight_decay=self.args['weight_decay'],
            )
            optimizer = C_Flat(params=self._network.parameters(), base_optimizer=base_optimizer, model=self._network,
                               cflat=self.args['cflat'], rho=self.args['rho'], lamb=self.args['lamb'])
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args['milestones'], gamma=self.args['lrate_decay']
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def create_loss_fn(self, inputs, targets):
        """
        Create a closure to calculate the loss
        """
        def loss_fn():
            logits = self._network(inputs)["logits"]
            loss_clf = F.cross_entropy(logits, targets)
            return logits, [loss_clf]

        return loss_fn

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['init_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['init_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['init_epoch'],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['epochs']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                loss_fn = self.create_loss_fn(inputs, targets)
                optimizer.set_closure(loss_fn)
                logits, loss_list = optimizer.step()
                losses += sum(loss_list).item()

                # acc
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['epochs'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['epochs'],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        with torch.no_grad():
            _, sample_inputs, sample_targets = next(iter(test_loader))
            sample_inputs, sample_targets = sample_inputs.to(self._device), sample_targets.to(self._device)
            plot_loss_surface_3d(self._network, self.create_loss_fn, sample_inputs, sample_targets, self._device, self._cur_task)

        logging.info(info)


def plot_loss_surface_3d(model, loss_fn, inputs, targets, device, current_task,steps=30, alpha=3e-1):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    model.eval()

    # 获取当前参数
    theta = [p.clone().detach() for p in model.parameters() if p.requires_grad]
    
    # 创建两个扰动方向：delta 和 epsilon
    delta = [torch.randn_like(p) for p in theta]
    epsilon = [torch.randn_like(p) for p in theta]

    x = np.linspace(-alpha, alpha, steps)
    y = np.linspace(-alpha, alpha, steps)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(steps):
        for j in range(steps):
            # 应用扰动
            for p, t, d, e in zip(model.parameters(), theta, delta, epsilon):
                p.data = t + x[i] * d + y[j] * e

            logits = model(inputs)["logits"]
            loss = F.cross_entropy(logits, targets)
            Z[i, j] = loss.item()

    # 恢复原始参数
    for p, t in zip(model.parameters(), theta):
        p.data = t

    # 绘图
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    np.savez(f"/data/wfq/C-Flat-main/surface/{current_task}_loss_surface_data.npz", X=X, Y=Y, Z=Z)
    ax.set_title("Loss Surface")
    ax.set_xlabel("Direction 1")
    ax.set_ylabel("Direction 2")
    ax.set_zlabel("Loss")
    plt.tight_layout()
    plt.savefig(f"./loss_surface_task{current_task}.jpg")
    plt.close()
