import logging
import os
import shutil

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from loss.nt_xent import NTXentLoss

# from models.resnet_clr import ResNetSimCLR
from models.model import ModelCLR

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.manual_seed(0)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy(
            "./config.yaml", os.path.join(model_checkpoints_folder, "config.yaml")
        )


class SimCLR(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter()
        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(
            self.device, config["batch_size"], **config["loss"]
        )
        self.truncation = config["truncation"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["bert_base_model"]
        )  # , do_lower_case=config['model_bert']['do_lower_case'])

    def _get_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Running on:", device)
        return device

    def train(self):
        # Dataloaders
        train_loader, valid_loader = self.dataset.get_data_loaders()

        # Model Resnet Initialize
        model = ModelCLR(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)

        optimizer = torch.optim.Adam(
            model.parameters(),
            eval(self.config["learning_rate"]),
            weight_decay=eval(self.config["weight_decay"]),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1
        )

        scaler = GradScaler()

        # Checkpoint folder
        model_checkpoints_folder = os.path.join(self.writer.log_dir, "checkpoints")

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        print("Training...")

        for epoch_counter in range(self.config["epochs"]):
            # print(f'Epoch {epoch_counter}')
            for xis, xls in tqdm(train_loader):
                optimizer.zero_grad()
                # optimizer_bert.zero_grad()

                xls = self.tokenizer(
                    list(xls),
                    return_tensors="pt",
                    padding=True,
                    truncation=self.truncation,
                )

                xis = xis.to(self.device)
                xls = xls.to(self.device)

                # Only use autocast for the forward pass
                with autocast():
                    # get the representations and the projections (Mixed Precision Training)
                    zis, zls = model(xis, xls)  # [N,C]

                    # get the representations and the projections
                    # zls = model_bert(xls)  # [N,C]
                    # zls = xls
                    # normalize projection feature vectors

                    loss = self.nt_xent_criterion(zis, zls)

                    # loss = self._step(model_res, model_bert, xis, xls, n_iter)

                if n_iter % self.config["log_every_n_steps"] == 0:
                    self.writer.add_scalar("train_loss", loss, global_step=n_iter)

                # Scales the loss to create scaled gradients
                scaler.scale(loss).backward()

                # Unscales the gradients
                scaler.step(optimizer)

                # optimizer_bert.step()
                # optimizer.step()

                # Update the scale for next iteration
                scaler.update()
                n_iter += 1

            print(f"Epoch {epoch_counter} ------ Train Loss: {loss}")

            # validate the model if requested
            if epoch_counter % self.config["eval_every_n_epochs"] == 0:
                valid_loss = self._validate(model, valid_loader, n_iter)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(
                        model.state_dict(),
                        os.path.join(model_checkpoints_folder, "model.pth"),
                    )
                self.writer.add_scalar(
                    "validation_loss", valid_loss, global_step=valid_n_iter
                )
                valid_n_iter += 1
                print(f"Validation {epoch_counter} - Valid Loss: {valid_loss}")

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step(valid_loss)
            self.writer.add_scalar(
                "cosine_lr_decay",
                scheduler.get_last_lr()[0],
                global_step=n_iter,
            )

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join(
                "./runs", self.config["fine_tune_from"], "checkpoints"
            )
            state_dict = torch.load(os.path.join(checkpoints_folder, "model.pth"))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader, n_iter):
        # validation steps
        with torch.no_grad():
            model.eval()
            # model_bert.eval()
            valid_loss = 0.0
            counter = 0
            # print(f'Validation step')
            for xis, xls in tqdm(valid_loader):
                xls = self.tokenizer(
                    list(xls),
                    return_tensors="pt",
                    padding=True,
                    truncation=self.truncation,
                )

                xis = xis.to(self.device)
                xls = xls.to(self.device)

                # get the representations and the projections
                zis, zls = model(xis, xls)  # [N,C]

                loss = self.nt_xent_criterion(zis, zls)

                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        # model_bert.train()
        return valid_loss
