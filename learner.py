from tqdm import tqdm
import wandb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import torch.optim as optim

from data.util import get_dataset, IdxDataset
from module.loss import GeneralizedCELoss
from module.util import get_model
from util import EMA
import random
import torchvision.utils as vutils


class Learner(object):
    def __init__(self, args):
        data2model = {'cmnist': "MLP",
                      'cifar10c': "ResNet18",
                      'bffhq': "ResNet18",
                      'dgw_generator': 'autoregressor'
                      }

        data2batch_size = {'cmnist': 256,
                           'cifar10c': 256,
                           'bffhq': 64}

        data2preprocess = {'cmnist': None,
                           'cifar10c': True,
                           'bffhq': True}

        if args.wandb:
            import wandb
            wandb.init(project='Debiasing-Global-Workspace', name=args.exp if not args.dgw_generator_training else args.exp + '_dgw_generator')
            wandb.run.name = args.exp

        run_name = args.exp
        if args.tensorboard:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(f'result/summary/{run_name}')

        self.model = data2model[args.dataset]
        self.batch_size = data2batch_size[args.dataset]

        print(f'model: {self.model} || dataset: {args.dataset}')
        print(f'working with experiment: {args.exp}...')
        self.log_dir = os.makedirs(os.path.join(args.log_dir, args.dataset, args.exp), exist_ok=True)
        self.device = torch.device(args.device)
        self.args = args

        print(self.args)

        # logging directories
        self.log_dir = os.path.join(args.log_dir, args.dataset, args.exp)
        self.summary_dir = os.path.join(args.log_dir, args.dataset, "summary", args.exp)
        self.summary_gradient_dir = os.path.join(self.log_dir, "gradient")
        self.result_dir = os.path.join(self.log_dir, "result")
        os.makedirs(self.summary_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        self.train_dataset = get_dataset(
            args.dataset,
            data_dir=args.data_dir,
            dataset_split="train",
            transform_split="train",
            percent=args.percent,
            use_preprocess=data2preprocess[args.dataset],
            use_type0=args.use_type0,
            use_type1=args.use_type1
        )
        self.valid_dataset = get_dataset(
            args.dataset,
            data_dir=args.data_dir,
            dataset_split="valid",
            transform_split="valid",
            percent=args.percent,
            use_preprocess=data2preprocess[args.dataset],
            use_type0=args.use_type0,
            use_type1=args.use_type1
        )

        self.test_dataset = get_dataset(
            args.dataset,
            data_dir=args.data_dir,
            dataset_split="test",
            transform_split="valid",
            percent=args.percent,
            use_preprocess=data2preprocess[args.dataset],
            use_type0=args.use_type0,
            use_type1=args.use_type1
        )

        train_target_attr = []
        for data in self.train_dataset.data:
            train_target_attr.append(int(data.split('_')[-2]))
        train_target_attr = torch.LongTensor(train_target_attr)

        attr_dims = []
        attr_dims.append(torch.max(train_target_attr).item() + 1)
        self.num_classes = attr_dims[0]
        self.train_dataset = IdxDataset(self.train_dataset)

        # JYH: Reproducibility for dataloader
        # def seed_worker(worker_id):
        #     worker_seed = torch.initial_seed() % 2 ** 32
        #     np.random.seed(worker_seed)
        #     random.seed(worker_seed)
        #
        # g = torch.Generator()
        # g.manual_seed(args.seed)

        # make loader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            # worker_init_fn=seed_worker,
            # generator=g,
        )

        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            # worker_init_fn=seed_worker,
            # generator=g,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            # worker_init_fn=seed_worker,
            # generator=g,
        )

        # define model and optimizer
        self.model_b = get_model(self.model, attr_dims[0]).to(self.device)
        self.model_d = get_model(self.model, attr_dims[0]).to(self.device)

        if args.dgw_generator_training:
            self.model_generator = get_model(data2model['dgw_generator'], None).to(self.device)
            self.optimizer_generator = torch.optim.Adam(
                self.model_generator.parameters(),
                lr=args.lr,
            )
            self.dgw_generator_criterion = nn.MSELoss()

        self.optimizer_b = torch.optim.Adam(
            self.model_b.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        self.optimizer_d = torch.optim.Adam(
            self.model_d.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        # define loss
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.bias_criterion = nn.CrossEntropyLoss(reduction='none')

        print(f'self.criterion: {self.criterion}')
        print(f'self.bias_criterion: {self.bias_criterion}')

        if args.dgw_generator_training:
            print(f'self.dgw_generator_criterion: {self.dgw_generator_criterion}')

        self.sample_loss_ema_b = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes,
                                     alpha=args.ema_alpha)
        self.sample_loss_ema_d = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes,
                                     alpha=args.ema_alpha)

        print(f'alpha : {self.sample_loss_ema_d.alpha}')

        self.best_valid_acc_b, self.best_test_acc_b = 0., 0.
        self.best_valid_acc_d, self.best_test_acc_d = 0., 0.
        print('finished model initialization....')

    # evaluation code for vanilla
    def evaluate(self, model, data_loader):
        model.eval()
        total_correct, total_num = 0, 0
        for data, attr, index in tqdm(data_loader, leave=False):
            label = attr[:, 0]
            data = data.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                logit = model(data)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()
                total_correct += correct.sum()
                total_num += correct.shape[0]

        accs = total_correct / float(total_num)
        model.train()

        return accs

    # evaluation code for ours
    def evaluate_lfa(self, model_b, model_l, data_loader, model='label'):
        model_b.eval()
        model_l.eval()

        total_correct, total_num = 0, 0

        for data, attr, index in tqdm(data_loader, leave=False):
            label = attr[:, 0]
            # label = attr
            data = data.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                if self.args.dataset == 'cmnist':
                    z_l = model_l.extract(data)
                    z_b = model_b.extract(data)
                else:
                    z_l, z_b = [], []
                    hook_fn = self.model_l.avgpool.register_forward_hook(self.concat_dummy(z_l))
                    _ = self.model_l(data)
                    hook_fn.remove()
                    z_l = z_l[0]
                    hook_fn = self.model_b.avgpool.register_forward_hook(self.concat_dummy(z_b))
                    _ = self.model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]
                z_origin = torch.cat((z_l, z_b), dim=1)
                if model == 'bias':
                    pred_label = model_b.fc(z_origin)
                else:
                    pred_label = model_l.fc(z_origin)
                pred = pred_label.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()
                total_correct += correct.sum()
                total_num += correct.shape[0]

        accs = total_correct / float(total_num)
        model_b.train()
        model_l.train()

        return accs

    # Todo: Need to reproduce Table 4 in LFA paper.
    # 1. Swapping Intrinsic/Bias needs to be implemented.
    # 1-1. z_swap is needed.
    def evaluate_dgw(self, model_b, model_l, data_loader, model='label'):
        model_b.eval()
        model_l.eval()

        total_correct, total_num = 0, 0

        for data, attr, index in tqdm(data_loader, leave=False):
            label = attr[:, 0]
            # label = attr
            data = data.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                if self.args.dataset == 'cmnist':
                    z_l = model_l.extract(data)
                    z_b = model_b.extract(data)
                else:
                    z_l, z_b = [], []
                    hook_fn = self.model_l.avgpool.register_forward_hook(self.concat_dummy(z_l))
                    _ = self.model_l(data)
                    hook_fn.remove()
                    z_l = z_l[0]
                    hook_fn = self.model_b.avgpool.register_forward_hook(self.concat_dummy(z_b))
                    _ = self.model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]
                z_origin = torch.cat((z_l, z_b), dim=1)
                if model == 'bias':  # Todo: Original Bias in Table 4 in LFA paper?
                    pred_label = model_b.fc(z_origin)
                else:   # Todo: Original Intrinsic in Table 4 in LFA paper?
                    pred_label = model_l.fc(z_origin)
                pred = pred_label.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()
                total_correct += correct.sum()
                total_num += correct.shape[0]

        accs = total_correct / float(total_num)
        model_b.train()
        model_l.train()

        return accs

    def save_vanilla(self, step, best=None):
        if best:
            model_path = os.path.join(self.result_dir, "best_model.th")
        else:
            model_path = os.path.join(self.result_dir, "model_{}.th".format(step))
        state_dict = {
            'steps': step,
            'state_dict': self.model_b.state_dict(),
            'optimizer': self.optimizer_b.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)
        print(f'{step} model saved ...')

    def save_lfa(self, step, best=None):
        if best:
            model_path = os.path.join(self.result_dir, "best_model_l.th")
        else:
            model_path = os.path.join(self.result_dir, "model_l_{}.th".format(step))
        state_dict = {
            'steps': step,
            'state_dict': self.model_l.state_dict(),
            'optimizer': self.optimizer_l.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)

        if best:
            model_path = os.path.join(self.result_dir, "best_model_b.th")
        else:
            model_path = os.path.join(self.result_dir, "model_b_{}.th".format(step))
        state_dict = {
            'steps': step,
            'state_dict': self.model_b.state_dict(),
            'optimizer': self.optimizer_b.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)

        print(f'{step} model saved ...')

    def save_dgw(self, step, best=None):
        if best:
            model_path = os.path.join(self.result_dir, "best_model_l.th")
        else:
            model_path = os.path.join(self.result_dir, "model_l_{}.th".format(step))
        state_dict = {
            'steps': step,
            'state_dict': self.model_l.state_dict(),
            'optimizer': self.optimizer_l.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)

        if best:
            model_path = os.path.join(self.result_dir, "best_model_b.th")
        else:
            model_path = os.path.join(self.result_dir, "model_b_{}.th".format(step))
        state_dict = {
            'steps': step,
            'state_dict': self.model_b.state_dict(),
            'optimizer': self.optimizer_b.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)

        # JYH: Save two GWSs as well.
        # GW1
        if best:
            model_path = os.path.join(self.result_dir, "best_model_gw_1.th".format(step))
        else:
            model_path = os.path.join(self.result_dir, "model_gw_1_{}.th".format(step))
        state_dict = {
            'steps': step,
            'state_dict': self.model_gw_1.state_dict(),
            'optimizer': self.optimizer_gw_1.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)
        # GW2
        if best:
            model_path = os.path.join(self.result_dir, "best_model_gw_2.th".format(step))
        else:
            model_path = os.path.join(self.result_dir, "model_gw_2_{}.th".format(step))
        state_dict = {
            'steps': step,
            'state_dict': self.model_gw_2.state_dict(),
            'optimizer': self.optimizer_gw_2.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)

        print(f'{step} model saved ...')

    def board_vanilla_loss(self, step, loss_b):
        if self.args.wandb:
            wandb.log({
                "loss_b_train": loss_b,
            }, step=step, )

        if self.args.tensorboard:
            self.writer.add_scalar(f"loss/loss_b_train", loss_b, step)

    def board_lfa_loss(self, step, loss_dis_conflict, loss_dis_align, loss_swap_conflict, loss_swap_align,
                       lambda_swap):

        if self.args.wandb:
            wandb.log({
                "loss_dis_conflict": loss_dis_conflict,
                "loss_dis_align": loss_dis_align,
                "loss_swap_conflict": loss_swap_conflict,
                "loss_swap_align": loss_swap_align,
                "loss": (loss_dis_conflict + loss_dis_align) + lambda_swap * (loss_swap_conflict + loss_swap_align)
            }, step=step, )

        if self.args.tensorboard:
            self.writer.add_scalar(f"loss/loss_dis_conflict", loss_dis_conflict, step)
            self.writer.add_scalar(f"loss/loss_dis_align", loss_dis_align, step)
            self.writer.add_scalar(f"loss/loss_swap_conflict", loss_swap_conflict, step)
            self.writer.add_scalar(f"loss/loss_swap_align", loss_swap_align, step)
            self.writer.add_scalar(f"loss/loss", (loss_dis_conflict + loss_dis_align) + lambda_swap * (
                    loss_swap_conflict + loss_swap_align), step)

    def board_vanilla_acc(self, step, epoch, inference=None):
        valid_accs_b = self.evaluate(self.model_b, self.valid_loader)
        test_accs_b = self.evaluate(self.model_b, self.test_loader)

        print(f'epoch: {epoch}')

        if valid_accs_b >= self.best_valid_acc_b:
            self.best_valid_acc_b = valid_accs_b
        if test_accs_b >= self.best_test_acc_b:
            self.best_test_acc_b = test_accs_b
            self.save_vanilla(step, best=True)

        if self.args.wandb:
            wandb.log({
                "acc_b_valid": valid_accs_b,
                "acc_b_test": test_accs_b,
            },
                step=step, )
            wandb.log({
                "best_acc_b_valid": self.best_valid_acc_b,
                "best_acc_b_test": self.best_test_acc_b,
            },
                step=step, )

        print(f'valid_b: {valid_accs_b} || test_b: {test_accs_b}')

        if self.args.tensorboard:
            self.writer.add_scalar(f"acc/acc_b_valid", valid_accs_b, step)
            self.writer.add_scalar(f"acc/acc_b_test", test_accs_b, step)

            self.writer.add_scalar(f"acc/best_acc_b_valid", self.best_valid_acc_b, step)
            self.writer.add_scalar(f"acc/best_acc_b_test", self.best_test_acc_b, step)

    def board_lfa_acc(self, step, inference=None):
        # check label network
        valid_accs_d = self.evaluate_lfa(self.model_b, self.model_l, self.valid_loader, model='label')
        test_accs_d = self.evaluate_lfa(self.model_b, self.model_l, self.test_loader, model='label')
        if inference:
            print(f'test acc: {test_accs_d.item()}')
            import sys
            sys.exit(0)

        if valid_accs_d >= self.best_valid_acc_d:
            self.best_valid_acc_d = valid_accs_d
        if test_accs_d >= self.best_test_acc_d:
            self.best_test_acc_d = test_accs_d
            self.save_lfa(step, best=True)

        if self.args.wandb:
            wandb.log({
                "acc_d_valid": valid_accs_d,
                "acc_d_test": test_accs_d,
            },
                step=step, )
            wandb.log({
                "best_acc_d_valid": self.best_valid_acc_d,
                "best_acc_d_test": self.best_test_acc_d,
            },
                step=step, )

        if self.args.tensorboard:
            self.writer.add_scalar(f"acc/acc_d_valid", valid_accs_d, step)
            self.writer.add_scalar(f"acc/acc_d_test", test_accs_d, step)
            self.writer.add_scalar(f"acc/best_acc_d_valid", self.best_valid_acc_d, step)
            self.writer.add_scalar(f"acc/best_acc_d_test", self.best_test_acc_d, step)

        print(f'valid_d: {valid_accs_d} || test_d: {test_accs_d} ')

    def board_dgw_acc(self, step, inference=None):
        # check label network
        valid_accs_d = self.evaluate_dgw(self.model_b, self.model_l, self.valid_loader, model='label')
        test_accs_d = self.evaluate_dgw(self.model_b, self.model_l, self.test_loader, model='label')
        if inference:
            print(f'test acc: {test_accs_d.item()}')
            import sys
            sys.exit(0)

        if valid_accs_d >= self.best_valid_acc_d:
            self.best_valid_acc_d = valid_accs_d
        if test_accs_d >= self.best_test_acc_d:
            self.best_test_acc_d = test_accs_d
            self.save_dgw(step, best=True)

        if self.args.wandb:
            wandb.log({
                "acc_d_valid": valid_accs_d,
                "acc_d_test": test_accs_d,
            },
                step=step, )
            wandb.log({
                "best_acc_d_valid": self.best_valid_acc_d,
                "best_acc_d_test": self.best_test_acc_d,
            },
                step=step, )

        if self.args.tensorboard:
            self.writer.add_scalar(f"acc/acc_d_valid", valid_accs_d, step)
            self.writer.add_scalar(f"acc/acc_d_test", test_accs_d, step)
            self.writer.add_scalar(f"acc/best_acc_d_valid", self.best_valid_acc_d, step)
            self.writer.add_scalar(f"acc/best_acc_d_test", self.best_test_acc_d, step)

        print(f'valid_d: {valid_accs_d} || test_d: {test_accs_d} ')

    def concat_dummy(self, z):
        def hook(model, input, output):
            z.append(output.squeeze())
            return torch.cat((output, torch.zeros_like(output)), dim=1)

        return hook

    def train_vanilla(self, args):
        # training vanilla ...
        train_iter = iter(self.train_loader)
        train_num = len(self.train_dataset.dataset)
        epoch, cnt = 0, 0

        for step in tqdm(range(args.num_steps)):
            try:
                index, data, attr, _ = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                index, data, attr, _ = next(train_iter)

            data = data.to(self.device)
            attr = attr.to(self.device)
            label = attr[:, args.target_attr_idx]

            logit_b = self.model_b(data)
            loss_b_update = self.criterion(logit_b, label)
            loss = loss_b_update.mean()

            self.optimizer_b.zero_grad()
            loss.backward()
            self.optimizer_b.step()

            ##################################################
            #################### LOGGING #####################
            ##################################################

            if step % args.save_freq == 0:
                self.save_vanilla(step)

            if step % args.log_freq == 0:
                self.board_vanilla_loss(step, loss_b=loss)

            if step % args.valid_freq == 0:
                self.board_vanilla_acc(step, epoch)

            cnt += len(index)
            if cnt == train_num:
                print(f'finished epoch: {epoch}')
                epoch += 1
                cnt = 0

    def train_lfa(self, args):
        epoch, cnt = 0, 0
        print('************** main training starts... ************** ')
        train_num = len(self.train_dataset)

        # self.model_l   : model for predicting intrinsic attributes ((E_i,C_i) in the main paper)
        # self.model_l.fc: fc layer for predicting intrinsic attributes (C_i in the main paper)
        # self.model_b   : model for predicting bias attributes ((E_b, C_b) in the main paper)
        # self.model_b.fc: fc layer for predicting bias attributes (C_b in the main paper)

        if args.dataset == 'cmnist':
            self.model_l = get_model('mlp_DISENTANGLE', self.num_classes).to(self.device)
            self.model_b = get_model('mlp_DISENTANGLE', self.num_classes).to(self.device)
        else:
            if self.args.use_resnet20:  # Use this option only for comparing with LfF
                self.model_l = get_model('ResNet20_OURS', self.num_classes).to(self.device)
                self.model_b = get_model('ResNet20_OURS', self.num_classes).to(self.device)
                print('our resnet20....')
            else:
                self.model_l = get_model('resnet_DISENTANGLE', self.num_classes).to(self.device)
                self.model_b = get_model('resnet_DISENTANGLE', self.num_classes).to(self.device)

        self.optimizer_l = torch.optim.Adam(
            self.model_l.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        self.optimizer_b = torch.optim.Adam(
            self.model_b.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        if args.use_lr_decay:
            self.scheduler_b = optim.lr_scheduler.StepLR(self.optimizer_b, step_size=args.lr_decay_step,
                                                         gamma=args.lr_gamma)
            self.scheduler_l = optim.lr_scheduler.StepLR(self.optimizer_l, step_size=args.lr_decay_step,
                                                         gamma=args.lr_gamma)

        self.bias_criterion = GeneralizedCELoss(q=0.7)

        print(f'criterion: {self.criterion}')
        print(f'bias criterion: {self.bias_criterion}')
        train_iter = iter(self.train_loader)

        for step in tqdm(range(args.num_steps)):
            try:
                index, data, attr, image_path = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                index, data, attr, image_path = next(train_iter)

            data = data.to(self.device)
            attr = attr.to(self.device)
            label = attr[:, args.target_attr_idx].to(self.device)

            # Feature extraction
            # Prediction by concatenating zero vectors (dummy vectors).
            # We do not use the prediction here.
            if args.dataset == 'cmnist':
                z_l = self.model_l.extract(data)
                z_b = self.model_b.extract(data)
            else:
                z_b = []
                # Use this only for reproducing CIFARC10 of LfF
                if self.args.use_resnet20:
                    hook_fn = self.model_b.layer3.register_forward_hook(self.concat_dummy(z_b))
                    _ = self.model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]

                    z_l = []
                    hook_fn = self.model_l.layer3.register_forward_hook(self.concat_dummy(z_l))
                    _ = self.model_l(data)
                    hook_fn.remove()

                    z_l = z_l[0]

                else:
                    hook_fn = self.model_b.avgpool.register_forward_hook(self.concat_dummy(z_b))
                    _ = self.model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]

                    z_l = []
                    hook_fn = self.model_l.avgpool.register_forward_hook(self.concat_dummy(z_l))
                    _ = self.model_l(data)
                    hook_fn.remove()

                    z_l = z_l[0]

            # z=[z_l, z_b]
            # Gradients of z_b are not backpropagated to z_l (and vice versa) in order to guarantee disentanglement of representation.
            z_conflict = torch.cat((z_l, z_b.detach()), dim=1)
            z_align = torch.cat((z_l.detach(), z_b), dim=1)

            # Prediction using z=[z_l, z_b]
            pred_conflict = self.model_l.fc(z_conflict)
            pred_align = self.model_b.fc(z_align)

            loss_dis_conflict = self.criterion(pred_conflict, label).detach()
            loss_dis_align = self.criterion(pred_align, label).detach()

            # EMA sample loss
            self.sample_loss_ema_d.update(loss_dis_conflict, index)
            self.sample_loss_ema_b.update(loss_dis_align, index)

            # class-wise normalize
            loss_dis_conflict = self.sample_loss_ema_d.parameter[index].clone().detach()
            loss_dis_align = self.sample_loss_ema_b.parameter[index].clone().detach()

            loss_dis_conflict = loss_dis_conflict.to(self.device)
            loss_dis_align = loss_dis_align.to(self.device)

            # JYH - Utilize the relative difficulty score of each data sample
            # from "Learning from failure: Training debiased classiﬁer from biased classiﬁer"
            # make this function to get `loss_weight`
            for c in range(self.num_classes):
                class_index = torch.where(label == c)[0].to(self.device)
                max_loss_conflict = self.sample_loss_ema_d.max_loss(c)
                max_loss_align = self.sample_loss_ema_b.max_loss(c)
                loss_dis_conflict[class_index] /= max_loss_conflict
                loss_dis_align[class_index] /= max_loss_align

            loss_weight = loss_dis_align / (
                    loss_dis_align + loss_dis_conflict + 1e-8)  # Eq.1 (reweighting module) in the main paper
            loss_dis_conflict = self.criterion(pred_conflict, label) * loss_weight.to(
                self.device)  # Eq.2 W(z)CE(C_i(z),y)
            loss_dis_align = self.bias_criterion(pred_align, label)  # Eq.2 GCE(C_b(z),y)

            # feature-level augmentation : augmentation after certain iteration (after representation is disentangled at a certain level)
            if step > args.curr_step:
                indices = np.random.permutation(z_b.size(0))
                z_b_swap = z_b[indices]  # z tilde
                label_swap = label[indices]  # y tilde

                # Prediction using z_swap=[z_l, z_b tilde]
                # Again, gradients of z_b tilde are not backpropagated to z_l (and vice versa) in order to guarantee disentanglement of representation.
                z_mix_conflict = torch.cat((z_l, z_b_swap.detach()), dim=1)
                z_mix_align = torch.cat((z_l.detach(), z_b_swap), dim=1)

                # Prediction using z_swap
                pred_mix_conflict = self.model_l.fc(z_mix_conflict)
                pred_mix_align = self.model_b.fc(z_mix_align)

                # JYH: Question - why did they use W(z), not W(z_mix)?
                loss_swap_conflict = self.criterion(pred_mix_conflict, label) * loss_weight.to(
                    self.device)  # Eq.3 W(z)CE(C_i(z_swap),y)
                loss_swap_align = self.bias_criterion(pred_mix_align, label_swap)  # Eq.3 GCE(C_b(z_swap),y tilde)
                lambda_swap = self.args.lambda_swap  # Eq.3 lambda_swap_b

            else:
                # before feature-level augmentation
                loss_swap_conflict = torch.tensor([0]).float()
                loss_swap_align = torch.tensor([0]).float()
                lambda_swap = 0

            loss_dis = loss_dis_conflict.mean() + args.lambda_dis_align * loss_dis_align.mean()  # Eq.2 L_dis
            loss_swap = loss_swap_conflict.mean() + args.lambda_swap_align * loss_swap_align.mean()  # Eq.3 L_swap
            loss = loss_dis + lambda_swap * loss_swap  # Eq.4 Total objective

            self.optimizer_l.zero_grad()
            self.optimizer_b.zero_grad()
            loss.backward()
            self.optimizer_l.step()
            self.optimizer_b.step()

            if step >= args.curr_step and args.use_lr_decay:
                self.scheduler_b.step()
                self.scheduler_l.step()

            if args.use_lr_decay and step % args.lr_decay_step == 0:
                print('******* learning rate decay .... ********')
                print(f"self.optimizer_b lr: {self.optimizer_b.param_groups[-1]['lr']}")
                print(f"self.optimizer_l lr: {self.optimizer_l.param_groups[-1]['lr']}")

            if step % args.save_freq == 0:
                self.save_lfa(step)

            if step % args.log_freq == 0:
                bias_label = attr[:, 1]
                align_flag = torch.where(label == bias_label)[0]
                self.board_lfa_loss(
                    step=step,
                    loss_dis_conflict=loss_dis_conflict.mean(),
                    loss_dis_align=args.lambda_dis_align * loss_dis_align.mean(),
                    loss_swap_conflict=loss_swap_conflict.mean(),
                    loss_swap_align=args.lambda_swap_align * loss_swap_align.mean(),
                    lambda_swap=lambda_swap
                )

            if step % args.valid_freq == 0:
                self.board_lfa_acc(step)

            cnt += data.shape[0]
            if cnt == train_num:
                print(f'finished epoch: {epoch}')
                epoch += 1
                cnt = 0

    def train_dgw(self, args):
        epoch, cnt = 0, 0
        print('************** main training starts... ************** ')
        train_num = len(self.train_dataset)

        # self.model_l   : model for predicting intrinsic attributes ((E_i,C_i) in the main paper)
        # self.model_l.fc: fc layer for predicting intrinsic attributes (C_i in the main paper)
        # self.model_b   : model for predicting bias attributes ((E_b, C_b) in the main paper)
        # self.model_b.fc: fc layer for predicting bias attributes (C_b in the main paper)

        if args.dataset == 'cmnist':
            self.model_l = get_model('mlp_DISENTANGLE', self.num_classes).to(self.device)
            self.model_b = get_model('mlp_DISENTANGLE', self.num_classes).to(self.device)
            # JYH: Create two debiasing global workspaces
            self.model_gw_1 = get_model("global_workspace", 0, embedding_dim=32, n_concepts=args.n_concepts,
                                        num_iterations=args.n_iters, in_feature=7, latent_dim=args.dim_slots).to(self.device)
            self.model_gw_2 = get_model("global_workspace", 0, embedding_dim=32, n_concepts=args.n_concepts,
                                        num_iterations=args.n_iters, in_feature=7, latent_dim=args.dim_slots).to(self.device)
        else:
            if self.args.use_resnet20:  # Use this option only for comparing with LfF
                self.model_l = get_model('ResNet20_OURS', self.num_classes).to(self.device)
                self.model_b = get_model('ResNet20_OURS', self.num_classes).to(self.device)
                print('our resnet20....')
            else:
                self.model_l = get_model('resnet_DISENTANGLE', self.num_classes).to(self.device)
                self.model_b = get_model('resnet_DISENTANGLE', self.num_classes).to(self.device)
                # JYH: Create two debiasing global workspaces
                self.model_gw_1 = get_model("global_workspace", 0, embedding_dim=1024, n_concepts=args.n_concepts,
                                            num_iterations=args.n_iters, in_feature=14, latent_dim=args.dim_slots).to(self.device)
                self.model_gw_2 = get_model("global_workspace", 0, embedding_dim=1024, n_concepts=args.n_concepts,
                                            num_iterations=args.n_iters, in_feature=14, latent_dim=args.dim_slots).to(self.device)

        self.optimizer_l = torch.optim.Adam(
            self.model_l.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        self.optimizer_b = torch.optim.Adam(
            self.model_b.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        # JYH: optimizers for debiasing workspaces.
        self.optimizer_gw_1 = torch.optim.Adam(
            self.model_gw_1.parameters(),
            lr=args.lr_cct,
            weight_decay=args.weight_decay,
        )

        self.optimizer_gw_2 = torch.optim.Adam(
            self.model_gw_2.parameters(),
            lr=args.lr_cct,
            weight_decay=args.weight_decay,
        )

        if args.use_lr_decay:
            self.scheduler_b = optim.lr_scheduler.StepLR(self.optimizer_b, step_size=args.lr_decay_step,
                                                         gamma=args.lr_gamma)
            self.scheduler_l = optim.lr_scheduler.StepLR(self.optimizer_l, step_size=args.lr_decay_step,
                                                         gamma=args.lr_gamma)
            # JYH: schedulers for debiasing workspaces.
            self.scheduler_gw_1 = optim.lr_scheduler.StepLR(self.optimizer_gw_1, step_size=args.lr_decay_step,
                                                            gamma=args.lr_gamma)
            self.scheduler_gw_2 = optim.lr_scheduler.StepLR(self.optimizer_gw_2, step_size=args.lr_decay_step,
                                                            gamma=args.lr_gamma)

        self.bias_criterion = GeneralizedCELoss(q=0.7)

        print(f'criterion: {self.criterion}')
        print(f'bias criterion: {self.bias_criterion}')
        train_iter = iter(self.train_loader)

        for step in tqdm(range(args.num_steps)):
            try:
                index, data, attr, image_path = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                index, data, attr, image_path = next(train_iter)

            data = data.to(self.device)
            attr = attr.to(self.device)
            label = attr[:, args.target_attr_idx].to(self.device)

            # Feature extraction
            # Prediction by concatenating zero vectors (dummy vectors).
            # We do not use the prediction here.
            if args.dataset == 'cmnist':
                z_l = self.model_l.extract(data)
                z_b = self.model_b.extract(data)
            else:
                z_b = []
                # Use this only for reproducing CIFARC10 of LfF
                if self.args.use_resnet20:
                    hook_fn = self.model_b.layer3.register_forward_hook(self.concat_dummy(z_b))
                    _ = self.model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]

                    z_l = []
                    hook_fn = self.model_l.layer3.register_forward_hook(self.concat_dummy(z_l))
                    _ = self.model_l(data)
                    hook_fn.remove()

                    z_l = z_l[0]

                else:
                    hook_fn = self.model_b.avgpool.register_forward_hook(self.concat_dummy(z_b))
                    _ = self.model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]

                    z_l = []
                    hook_fn = self.model_l.avgpool.register_forward_hook(self.concat_dummy(z_l))
                    _ = self.model_l(data)
                    hook_fn.remove()

                    z_l = z_l[0]

            # z=[z_l, z_b]
            # Gradients of z_b are not backpropagated to z_l (and vice versa) in order to guarantee disentanglement of representation.
            z_conflict = torch.cat((z_l, z_b.detach()), dim=1)
            z_align = torch.cat((z_l.detach(), z_b), dim=1)

            # JYH
            # Global workspace 1 learns to decompose z_conflict embedding.
            slot_z_conflict, attn_z_conflict = self.model_gw_1(z_conflict)
            # Latent embedding mixup representation for z_conflict.
            ratio = np.random.beta(args.rep_alpha, args.rep_alpha, 1)[0]
            # z_conflict = args.rep_alpha * z_conflict + (1 - args.rep_alpha) * slot_z_conflict.squeeze(1)
            z_conflict = ratio * z_conflict + (1 - ratio) * slot_z_conflict.squeeze(1)

            # Global workspace 2 learns to decompose z_align embedding.
            slot_z_align, attn_z_align = self.model_gw_2(z_align)
            # Latent embedding mixup representation for z_align.
            ratio = np.random.beta(args.rep_alpha, args.rep_alpha, 1)[0]
            # z_align = args.rep_alpha * z_align + (1 - args.rep_alpha) * slot_z_align.squeeze(1)
            z_align = ratio * z_align + (1 - ratio) * slot_z_align.squeeze(1)


            # Prediction using z=[z_l, z_b]
            pred_conflict = self.model_l.fc(z_conflict)
            pred_align = self.model_b.fc(z_align)

            loss_dis_conflict = self.criterion(pred_conflict, label).detach()
            loss_dis_align = self.criterion(pred_align, label).detach()

            # EMA sample loss
            self.sample_loss_ema_d.update(loss_dis_conflict, index)
            self.sample_loss_ema_b.update(loss_dis_align, index)

            # class-wise normalize
            loss_dis_conflict = self.sample_loss_ema_d.parameter[index].clone().detach()
            loss_dis_align = self.sample_loss_ema_b.parameter[index].clone().detach()

            loss_dis_conflict = loss_dis_conflict.to(self.device)
            loss_dis_align = loss_dis_align.to(self.device)

            loss_ent = self.ent_loss(attn_z_conflict) + self.ent_loss(attn_z_align)

            # JYH - Utilize the relative difficulty score of each data sample
            # from "Learning from failure: Training debiased classiﬁer from biased classiﬁer"
            # make this function to get `loss_weight`
            for c in range(self.num_classes):
                class_index = torch.where(label == c)[0].to(self.device)
                max_loss_conflict = self.sample_loss_ema_d.max_loss(c)
                max_loss_align = self.sample_loss_ema_b.max_loss(c)
                loss_dis_conflict[class_index] /= max_loss_conflict
                loss_dis_align[class_index] /= max_loss_align

            loss_weight = loss_dis_align / (
                    loss_dis_align + loss_dis_conflict + 1e-8)  # Eq.1 (reweighting module) in the main paper
            loss_dis_conflict = self.criterion(pred_conflict, label) * loss_weight.to(
                self.device)  # Eq.2 W(z)CE(C_i(z),y)
            loss_dis_align = self.bias_criterion(pred_align, label)  # Eq.2 GCE(C_b(z),y)

            # feature-level augmentation : augmentation after certain iteration (after representation is disentangled at a certain level)
            if step > args.curr_step:
                indices = np.random.permutation(z_b.size(0))
                z_b_swap = z_b[indices]  # z tilde
                label_swap = label[indices]  # y tilde

                # Prediction using z_swap=[z_l, z_b tilde]
                # Again, gradients of z_b tilde are not backpropagated to z_l (and vice versa) in order to guarantee disentanglement of representation.
                z_mix_conflict = torch.cat((z_l, z_b_swap.detach()), dim=1)
                z_mix_align = torch.cat((z_l.detach(), z_b_swap), dim=1)

                # Global workspace 1 learns to decompose z_mix_conflict embedding as well.
                slot_z_mix_conflict, attn_z_mix_conflict = self.model_gw_1(z_mix_conflict)
                # Latent embedding mixup representation for z_mix_conflict.
                ratio = np.random.beta(args.rep_alpha, args.rep_alpha, 1)[0]
                # z_mix_conflict = args.rep_alpha * z_mix_conflict + (1 - args.rep_alpha) * slot_z_mix_conflict.squeeze(1)
                z_mix_conflict = ratio * z_mix_conflict + (1 - ratio) * slot_z_mix_conflict.squeeze(1)

                # Global workspace 2 learns to decompose z_mix_align embedding as well.
                slot_z_mix_align, attn_z_mix_align = self.model_gw_2(z_mix_align)
                # Latent embedding mixup representation for z_mix_alig.
                ratio = np.random.beta(args.rep_alpha, args.rep_alpha, 1)[0]
                # z_mix_align = args.rep_alpha * z_mix_align + (1 - args.rep_alpha) * slot_z_mix_align.squeeze(1)
                z_mix_align = ratio * z_mix_align + (1 - ratio) * slot_z_mix_align.squeeze(1)

                # Prediction using z_swap
                pred_mix_conflict = self.model_l.fc(z_mix_conflict)
                pred_mix_align = self.model_b.fc(z_mix_align)

                # JYH: Question - why did they use W(z), not W(z_mix)?
                loss_swap_conflict = self.criterion(pred_mix_conflict, label) * loss_weight.to(
                    self.device)  # Eq.3 W(z)CE(C_i(z_swap),y)
                loss_swap_align = self.bias_criterion(pred_mix_align, label_swap)  # Eq.3 GCE(C_b(z_swap),y tilde)
                lambda_swap = self.args.lambda_swap  # Eq.3 lambda_swap_b
                # JYH: introduce the entropy loss from CCT in order to improve stability of both representations.
                loss_ent += self.ent_loss(attn_z_mix_conflict) + self.ent_loss(attn_z_mix_align)

            else:
                # before feature-level augmentation
                loss_swap_conflict = torch.tensor([0]).float()
                loss_swap_align = torch.tensor([0]).float()
                lambda_swap = 0

            loss_dis = loss_dis_conflict.mean() + args.lambda_dis_align * loss_dis_align.mean()  # Eq.2 L_dis
            loss_swap = loss_swap_conflict.mean() + args.lambda_swap_align * loss_swap_align.mean()  # Eq.3 L_swap
            loss = loss_dis + lambda_swap * loss_swap  # Eq.4 Total objective
            # JYH: introduce the entropy loss from CCT in order to improve stability of both representations.
            loss += args.lambda_ent * loss_ent

            self.optimizer_l.zero_grad()
            self.optimizer_b.zero_grad()
            self.optimizer_gw_1.zero_grad()
            self.optimizer_gw_2.zero_grad()
            loss.backward()
            self.optimizer_l.step()
            self.optimizer_b.step()
            self.optimizer_gw_1.step()
            self.optimizer_gw_2.step()

            if step >= args.curr_step and args.use_lr_decay:
                self.scheduler_b.step()
                self.scheduler_l.step()
                self.scheduler_gw_1.step()
                self.scheduler_gw_2.step()

            if args.use_lr_decay and step % args.lr_decay_step == 0:
                print('******* learning rate decay .... ********')
                print(f"self.optimizer_b lr: {self.optimizer_b.param_groups[-1]['lr']}")
                print(f"self.optimizer_l lr: {self.optimizer_l.param_groups[-1]['lr']}")
                print(f"self.optimizer_gw_1 lr: {self.optimizer_gw_1.param_groups[-1]['lr']}")
                print(f"self.optimizer_gw_2 lr: {self.optimizer_gw_2.param_groups[-1]['lr']}")

            if step % args.save_freq == 0:
                self.save_dgw(step)

            if step % args.log_freq == 0:
                # Todo: bias_label is here.
                bias_label = attr[:, 1]
                align_flag = torch.where(label == bias_label)[0]
                self.board_lfa_loss(
                    step=step,
                    loss_dis_conflict=loss_dis_conflict.mean(),
                    loss_dis_align=args.lambda_dis_align * loss_dis_align.mean(),
                    loss_swap_conflict=loss_swap_conflict.mean(),
                    loss_swap_align=args.lambda_swap_align * loss_swap_align.mean(),
                    lambda_swap=lambda_swap
                )

            if step % args.valid_freq == 0:
                self.board_dgw_acc(step)

            cnt += data.shape[0]
            if cnt == train_num:
                print(f'finished epoch: {epoch}')
                epoch += 1
                cnt = 0

    def train_dgw_reconstruction(self, args):

        # folde setting
        self.img_save_dir = os.path.join(self.log_dir, "dgw_recon_images")
        os.makedirs(self.img_save_dir, exist_ok=True)

        # Train
        self.model_generator.train()

        # Load model_l, model_b, and model_gw_1 and 2
        if args.dataset == 'cmnist':
            self.model_l = get_model('mlp_DISENTANGLE', self.num_classes).to(self.device)
            self.model_b = get_model('mlp_DISENTANGLE', self.num_classes).to(self.device)
            self.model_gw_1 = get_model("global_workspace", 0, embedding_dim=32, n_concepts=args.n_concepts,
                                        num_iterations=args.n_iters).to(self.device)
            self.model_gw_2 = get_model("global_workspace", 0, embedding_dim=32, n_concepts=args.n_concepts,
                                        num_iterations=args.n_iters).to(self.device)
        else:
            self.model_l = get_model('resnet_DISENTANGLE', self.num_classes).to(self.device)
            self.model_b = get_model('resnet_DISENTANGLE', self.num_classes).to(self.device)
            self.model_gw_1 = get_model("global_workspace", 0, embedding_dim=1024, n_concepts=args.n_concepts,
                                        num_iterations=args.n_iters).to(self.device)
            self.model_gw_2 = get_model("global_workspace", 0, embedding_dim=1024, n_concepts=args.n_concepts,
                                        num_iterations=args.n_iters).to(self.device)

        self.model_l.load_state_dict(torch.load(os.path.join(args.pretrained_path, 'best_model_l.th'))['state_dict'])
        self.model_b.load_state_dict(torch.load(os.path.join(args.pretrained_path, 'best_model_b.th'))['state_dict'])
        self.model_gw_1.load_state_dict(torch.load(os.path.join(args.pretrained_path, 'best_model_gw_1.th'))['state_dict'])
        self.model_gw_2.load_state_dict(torch.load(os.path.join(args.pretrained_path, 'best_model_gw_2.th'))['state_dict'])
        print('Loading the pretrained models done.')

        # Eval Mode
        self.model_l.eval()
        self.model_b.eval()
        self.model_gw_1.eval()
        self.model_gw_2.eval()

        for step in tqdm(range(args.num_steps)):
            try:
                index, data, attr, image_path = next(train_iter)
            except:  # check lfa is also in this except
                train_iter = iter(self.train_loader)
                index, data, attr, image_path = next(train_iter)

            data = data.to(self.device)
            attr = attr.to(self.device)
            label = attr[:, args.target_attr_idx].to(self.device)

            # Assuming model_l and model_b return the required 16 channel output directly
            with torch.no_grad():  # Ensure gradients are not calculated for model_l and model_b
                output_l = self.model_l.extract(data).detach()  # Detach to prevent gradients from flowing back
                output_b = self.model_b.extract(data).detach()

            # Concatenate the outputs to form the 32 channel input for the generator
            gen_input = torch.cat((output_l, output_b), dim=1)
            # Global workspace 1 learns to decompose output_l
            slot_output_l, attn_output_l = self.model_gw_1(gen_input)
            slot_output_b, attn_output_b = self.model_gw_2(gen_input)
            slot_output = torch.flatten(torch.cat((slot_output_l, slot_output_b), dim=1), start_dim=1)

            self.optimizer_generator.zero_grad()
            gen_output = self.model_generator(slot_output)
            target = data.view_as(gen_output)
            loss = self.dgw_generator_criterion(gen_output, target)  # Define your target accordingly
            loss.backward()
            self.optimizer_generator.step()
            # Log loss to wandb
            wandb.log({"Generator Loss": loss.item()}, step=step)

            if step % 100 == 0:  # Save sample images every 100 steps
                # Concatenate original data and generated data
                gen_output = gen_output.view(-1, 3, 28, 28)  # reshape
                combined_images = torch.cat((data[:8], gen_output[:8]), dim=0)  # Taking 8 samples for visualization
                # Make a grid with the first row being original images and the second row being generated images
                image_grid = vutils.make_grid(combined_images, normalize=True, value_range=(0, 1), scale_each=True,
                                              nrow=8, padding=2)
                # Save the grid to a file
                vutils.save_image(image_grid, os.path.join(self.img_save_dir, f"sample_images_step_{step}.png"))
                wandb.log({"Generated Images": [wandb.Image(image_grid, caption=f"Step {step}")]}, step=step)

            if step % 5_000 == 0:
                torch.save(self.model_generator.state_dict(),
                           os.path.join(self.result_dir, f"generator_step_{step}.th"))

        # save final version
        torch.save(self.model_generator.state_dict(), os.path.join(self.result_dir, f"generator_step_{step}.th"))

    def test_lfa(self, args):
        if args.dataset == 'cmnist':
            self.model_l = get_model('mlp_DISENTANGLE', self.num_classes).to(self.device)
            self.model_b = get_model('mlp_DISENTANGLE', self.num_classes).to(self.device)
        else:
            self.model_l = get_model('resnet_DISENTANGLE', self.num_classes).to(self.device)
            self.model_b = get_model('resnet_DISENTANGLE', self.num_classes).to(self.device)

        self.model_l.load_state_dict(torch.load(os.path.join(args.pretrained_path, 'best_model_l.th'))['state_dict'])
        self.model_b.load_state_dict(torch.load(os.path.join(args.pretrained_path, 'best_model_b.th'))['state_dict'])
        self.board_lfa_acc(step=0, inference=True)

    def test_dgw(self, args):
        if args.dataset == 'cmnist':
            self.model_l = get_model('mlp_DISENTANGLE', self.num_classes).to(self.device)
            self.model_b = get_model('mlp_DISENTANGLE', self.num_classes).to(self.device)
        else:
            self.model_l = get_model('resnet_DISENTANGLE', self.num_classes).to(self.device)
            self.model_b = get_model('resnet_DISENTANGLE', self.num_classes).to(self.device)

        self.model_l.load_state_dict(torch.load(os.path.join(args.pretrained_path, 'best_model_l.th'))['state_dict'])
        self.model_b.load_state_dict(torch.load(os.path.join(args.pretrained_path, 'best_model_b.th'))['state_dict'])
        self.board_dgw_acc(step=0, inference=True)

    def ent_loss(self, probs, eps=1e-8):
        ent = -probs * torch.log(probs + eps)
        return ent.mean()
