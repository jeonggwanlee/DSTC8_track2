import os
import torch
import logging

from typing import Any, Optional, Tuple

from pytext.trainers import Trainer
from pytext.config import PyTextConfig
from pytext.config.pytext_config import ConfigBase
from pytext.common.constants import Stage
from pytext.models.model import Model
from pytext.utils import cuda_utils

from mldc.data.data_handler import BatchPreparationPipeline
from mldc.metrics.metrics import MetaLearnMetricReporter


TASKS_AGGR = 0
SUPPORT_ON_SLOW = 1
TARGET_ON_FAST = 2

EPSILON = 0.001
LOG = logging.getLogger("mldc.trainer")


class RetrievalTrainer(Trainer):

  class Config(ConfigBase):
    random_seed: int = 0
    # Whether metrics on training data should be computed and reported.
    report_train_metrics: bool = True

  def test(self, test_task_iters: BatchPreparationPipeline,
           model: Model,
           metric_reporter: MetaLearnMetricReporter):

    for mbidx, meta_batch in enumerate(test_task_iters):
      support, target, context = meta_batch
      for (s_inputs, t_inputs), (s_targets, t_targets), (s_context, t_context) in zip(support, target, context):
        task = t_context['task_id'][0]
        model.train()
        model.contextualize(s_context)
        model(*s_inputs, responses=s_targets)  # model remembers responses
        model.eval()

        with torch.no_grad():
          t_pred = model(*t_inputs)
          t_loss = model.get_loss(t_pred, t_targets, t_context).item()

          metric_reporter.add_batch_stats(task, t_loss, s_inputs,
                                          t_predictions=t_pred, t_targets=t_targets)

    metric_reporter.report_metric(stage=Stage.TEST, epoch=0, reset=False)

  def predict(self, test_task_iters: BatchPreparationPipeline,
              model: Model,
              metric_reporter: MetaLearnMetricReporter):

    for meta_batch in test_task_iters:
      support, target, context = meta_batch
      for (s_inputs, t_inputs), (s_targets, t_targets), (s_context, t_context) in zip(support, target, context):
        task = t_context['task_id'][0]
        model.train()
        model.contextualize(s_context)
        model(*s_inputs, responses=s_targets)  # model remembers responses
        model.eval()

        with torch.no_grad():
          resps, resp_lens = model(*t_inputs)

          yield dict(task=task, resps=resps, resp_lens=resp_lens,
                     s_inputs=s_inputs, s_targets=s_targets, s_context=s_context,
                     t_inputs=t_inputs, t_targets=t_targets, t_context=t_context)

  def train(
      self,
      text_embedder,
      train_task_iters: Optional[BatchPreparationPipeline],
      eval_task_iters: BatchPreparationPipeline,
      model: Model,
      metric_reporter: MetaLearnMetricReporter,
      train_config: PyTextConfig,
      rank: int = 0,
    ) -> Tuple[torch.nn.Module, Any]:

    diat = text_embedder.decode_ids_as_text
    if cuda_utils.CUDA_ENABLED:
      model = model.cuda()
    best_model_path = None

    # Start outer loop (meta learner "epochs") #############################################
    if not train_task_iters:
      LOG.warning("Model does not need meta-training")
    else:
      for epoch in range(1, 2):  # single epoch
        temp = next(train_task_iters)
        for bidx, (support, target, context) in zip(range(100), train_task_iters):
          for (s_inputs, t_inputs), (s_targets, t_targets), (s_context, t_context) in zip(support, target, context):
            # support : (2)
            # s_inputs : (6)
            # s_inputs[0].shape : (128, 3, 38) # 3 means 3 consecutive sentence ## 'denver', 'no , the thunderstorm has drifted north .', 'that makes me mad ! why is that ?'
            # s_inputs[1].shape : (128, 3, 38, 768) # I guess BertEmbedding
            # s_inputs[2].shape : (128, 2, 37) # 2 means the next consecutive sentence of s_inputs[0] ##  'no , the thunderstorm has drifted north .', 'that makes me mad ! why is that ?'
            # s_inputs[3].shape : (128) # [3, 3, 3, 3, 3....]
            # s_inputs[4].shape : (128, 3) # each length of sentences in s_inputs[0]
            # s_inputs[5].shape : (128, 2) # each length of sentences in s_inputs[2]
            # s_targets : (2)
            # s_targets[0].shape : (128, 2, 34) ## 'no, the thunderstorm has drifted north .', 'you would like the storm ?'
            # s_targets[1].shape : (128, 2) # each length of sentences in s_targets[0]
            # type(s_context) : dict # keys : {'target_seq_lens', 'orig_text', 'dlg_len', 'dlg_id', 'domain_id', 'task_id', 'index'}
            # s_context['target_seq_lens'].shape : (128, 2) # each length"+1" of sentences in s_targets[0]
            # s_context['orig_text'].__len__() : 128
            # s_context['orig_text'][0]'s original text == "turns": ["Hello how may I help you?", "Is there still supposed to be a thunderstorm today as     there was originally?", "what location?", "Denver", "No, the thunderstorm has drifted north.", "That makes me mad! Why is that?", "You would like the storm?", "Yes! It really upsets me that there isn't goin    g to be one now.", "I'm sorry, I will contact mother nature immediately!", "Why is there not going to be one?", "The radar say so."]
            # s_context['dlg_len'] = 4
            # s_context['dlg_id'] : (128) # '2d1d4ed2', '20debe73', ... ## "id"
            # s_context['domain_id'] : (128) # 'WEATHER_CHECK', 'WEATHER_CHECK'... ## "domain"
            # s_context['task_id'] : (128) # 'd941f2bb', '5f2bb1b2', ... ## "task_id"
            # s_context['index'] : (128) # 25650, 25414, 25454, 25445, 25465, 25370, 25333, 25411, 25203, 25108, 25631, 25532, 25155, 25472, 25365, 25356, 25258, 25282, 25242, 25518, 25150, 25237, 25372

            # t_inputs : (6)
            # text_embedder.decode_ids_as_text(s_inputs[0][0][0].cpu().numpy()) = 'what is your order number ?'
            task = t_context['task_id'][0]

            # Adapt the model using the support set
            model.train()
            for step in range(1):
              model.contextualize(s_context)
              model(*s_inputs, responses=s_targets)  # model remembers responses

            # Evaluate the model using the target set
            model.eval()    # model now retrieves from examples seen so far
            model.contextualize(t_context)
            t_pred = model(*t_inputs)
            t_loss = model.get_loss(t_pred, t_targets, t_context).item()
            metric_reporter.add_batch_stats(task, t_loss, s_inputs,
                                            t_predictions=t_pred, t_targets=t_targets)

        metric_reporter.report_metric(stage=Stage.TRAIN, epoch=epoch, reset=False)

      logging.info("Evaluating model on eval tasks")
      with torch.no_grad():
        for bidx, (support, target, context) in enumerate(eval_task_iters):
          for (s_inputs, t_inputs), (s_targets, t_targets), (s_context, t_context) in zip(support, target, context):
            task = t_context["task_id"][0]
            model.train()
            model.contextualize(s_context)
            model(*s_inputs, responses=s_targets)  # model remembers responses
            model.eval()
            t_pred = model(*t_inputs)
            t_loss = model.get_loss(t_pred, t_targets, t_context).item()

            metric_reporter.add_batch_stats(task, t_loss, s_inputs,
                                            t_predictions=t_pred, t_targets=t_targets)

      metric_reporter.report_metric(stage=Stage.EVAL, epoch=epoch, reset=False)

    best_model_path = os.path.join(
        train_config.modules_save_dir, "model.pt"
    )
    torch.save(model.state_dict(), best_model_path)

    return model, None
