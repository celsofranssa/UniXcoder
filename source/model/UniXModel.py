import importlib

import torch
from pytorch_lightning.core.lightning import LightningModule
from transformers import get_linear_schedule_with_warmup
from hydra.utils import instantiate

from source.metric.MRRMetric import MRRMetric


class UniXModel(LightningModule):

    def __init__(self, hparams):

        super(UniXModel, self).__init__()
        self.save_hyperparameters(hparams)

        # encoders
        self.encoder = instantiate(hparams.encoder)


        # loss function
        self.loss = instantiate(hparams.loss)

        # metric
        self.mrr = MRRMetric()


    def forward(self, desc, code):
        desc_repr = self.encoder(desc)
        code_repr = self.encoder(code)
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
        optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.hparams.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                    num_training_steps=self.num_training_steps)

        return (
            {"optimizer": optimizer,
             "lr_scheduler": {"scheduler": scheduler, "interval": "step", "name": "SCHDLR"}},
        )

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and number of epochs."""
        steps_per_epochs = len(self.train_dataloader()) / self.trainer.accumulate_grad_batches
        max_epochs = self.trainer.max_epochs
        return steps_per_epochs * max_epochs

