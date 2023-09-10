import hydra
import os
from omegaconf import OmegaConf
from source.helper.EvalHelper import EvalHelper
from source.helper.FitHelper import FitHelper
from source.helper.PredictHelper import PredictHelper
from source.helper.xCoFormerFitHelper import xCoFormerFitHelper
from source.helper.xCoFormerPredictHelper import xCoFormerPredictHelper


def fit(params):
    if params.model.name in ["xCoFormer", "RoBERTa", "BERT", "CodeBERT"]:
        helper = xCoFormerFitHelper(params)
    elif params.model.name == "UNIX":
        helper = FitHelper(params)
    helper.perform_fit()


def predict(params):
    if params.model.name in ["xCoFormer", "RoBERTa", "BERT", "CodeBERT"]:
        helper = xCoFormerPredictHelper(params)
    elif params.model.name == "UNIX":
        helper = PredictHelper(params)
    helper.perform_predict()


def eval(params):
    eval_helper = EvalHelper(params)
    eval_helper.perform_eval()


@hydra.main(config_path="settings", config_name="settings.yaml")
def perform_tasks(params):
    os.chdir(hydra.utils.get_original_cwd())
    OmegaConf.resolve(params)
    if "fit" in params.tasks:
        fit(params)

    if "predict" in params.tasks:
        predict(params)

    if "eval" in params.tasks:
        eval(params)


if __name__ == '__main__':
    perform_tasks()
