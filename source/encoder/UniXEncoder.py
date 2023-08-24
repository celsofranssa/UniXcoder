import torch
from pytorch_lightning import LightningModule
from transformers import RobertaModel


class UniXEncoder(LightningModule):
    def __init__(self, architecture):
        super(UniXEncoder, self).__init__()
        self.encoder = RobertaModel.from_pretrained(
            architecture
        )

    #
    # def forward(self, code_inputs=None, nl_inputs=None):
    #     if code_inputs is not None:
    #         #print(f"\ncode inputs({code_inputs.shape}): \n {code_inputs}\n")
    #         outputs = self.encoder(code_inputs, attention_mask=code_inputs.ne(1))[0]
    #         outputs = (outputs * code_inputs.ne(1)[:, :, None]).sum(1) / code_inputs.ne(1).sum(-1)[:, None]
    #         return torch.nn.functional.normalize(outputs, p=2, dim=1)
    #     else:
    #         outputs = self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[0]
    #         outputs = (outputs * nl_inputs.ne(1)[:, :, None]).sum(1) / nl_inputs.ne(1).sum(-1)[:, None]
    #         return torch.nn.functional.normalize(outputs, p=2, dim=1)

    def forward(self, input):
        outputs = self.encoder(input, attention_mask=input.ne(1))[0]
        outputs = (outputs * input.ne(1)[:, :, None]).sum(1) / input.ne(1).sum(-1)[:, None]
        return torch.nn.functional.normalize(outputs, p=2, dim=1)

