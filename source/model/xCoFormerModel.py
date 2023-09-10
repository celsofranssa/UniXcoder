import importlib

import torch
from pytorch_lightning.core.lightning import LightningModule
from transformers import get_linear_schedule_with_warmup
from hydra.utils import instantiate

from source.metric.MRRMetric import MRRMetric


class xCoFormerModel(LightningModule):

    def __init__(self, hparams):
        super(xCoFormerModel, self).__init__()
        self.save_hyperparameters(hparams)

        # encoders
        self.desc_encoder = instantiate(hparams.desc_encoder)
        self.code_encoder = instantiate(hparams.code_encoder)

        # loss function
        self.loss = instantiate(hparams.loss)

        # metric
        self.mrr = MRRMetric()

    def forward(self, desc, code):
        desc_repr = self.desc_encoder(desc)
        code_repr = self.code_encoder(code)
        return desc_repr, code_repr

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        desc, code = batch["desc"], batch["code"]
        desc_repr, code_repr = self(desc, code)
        train_loss = self.loss(desc_repr, code_repr)

        # log training loss
        self.log('train_LOSS', train_loss)

        return train_loss

    def validation_step(self, batch, batch_idx):
        desc, code = batch["desc"], batch["code"]
        desc_repr, code_repr = self(desc, code)
        self.log("val_MRR", self.mrr(desc_repr, code_repr), prog_bar=True)
        self.log("val_LOSS", self.loss(desc_repr, code_repr), prog_bar=True)

    def validation_epoch_end(self, outs):
        self.mrr.compute()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        idx, desc, code = batch["idx"], batch["desc"], batch["code"]
        desc_repr, code_repr = self(desc, code)

        return {
            "idx": idx,
            "desc_rpr": desc_repr,
            "code_rpr": code_repr
        }

    def configure_optimizers(self):
        desc_optimizer = torch.optim.AdamW(self.desc_encoder.parameters(), lr=self.hparams.lr, eps=1e-8)
        code_optimizer = torch.optim.AdamW(self.code_encoder.parameters(), lr=self.hparams.lr, eps=1e-8)
        desc_scheduler = get_linear_schedule_with_warmup(desc_optimizer, num_warmup_steps=0,
                                                         num_training_steps=self.num_training_steps)
        code_scheduler = get_linear_schedule_with_warmup(code_optimizer, num_warmup_steps=0,
                                                         num_training_steps=self.num_training_steps)

        return (
            {"optimizer": desc_optimizer, "lr_scheduler": desc_scheduler, "frequency": self.hparams.desc_frequency_opt},
            {"optimizer": code_optimizer, "lr_scheduler": code_scheduler, "frequency": self.hparams.code_frequency_opt},
        )


    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and number of epochs."""
        steps_per_epochs = len(self.train_dataloader()) / self.trainer.accumulate_grad_batches
        max_epochs = self.trainer.max_epochs
        return steps_per_epochs * max_epochs
