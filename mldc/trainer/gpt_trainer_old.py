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
from pytorch_pretrained_bert import OpenAIGPTDoubleHeadsModel
from pytorch_pretrained_bert import OpenAIAdam

TASKS_AGGR = 0
SUPPORT_ON_SLOW = 1
TARGET_ON_FAST = 2

EPSILON = 0.001
LOG = logging.getLogger("mldc.trainer")


class GptTrainer(Trainer):

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
            train_task_iters: Optional[BatchPreparationPipeline],    # Optional[X] is equivalent to Union[X, None].
            eval_task_iters: BatchPreparationPipeline,
            model: Model,
            metric_reporter: MetaLearnMetricReporter,
            train_config: PyTextConfig,
            rank: int = 0,
    ) -> Tuple[torch.nn.Module, Any]:

        if cuda_utils.CUDA_ENABLED:
            model = model.cuda()

        best_model_path = None
        meta_lr = 0.001
        update_lr = 0.01
        from pytorch_transformers import AdamW
        if model.representation.gptmode == 'gpt2':
            meta_optim = AdamW(model.parameters(), lr=meta_lr)
        else:
            meta_optim = OpenAIAdam(model.parameters(), lr=meta_lr)

        # Start outer loop (meta learner "epochs") #############################################
        if not train_task_iters:
            LOG.warning("Model does not need meta-training")
        else:
            logging.info("Training model on train tasks")
            for epoch in range(1, 2):  # single epoch
                for bidx, (support, target, context) in zip(range(100), train_task_iters): # 100 different tasks
                    # support.__len__() : task num
                    #class MetaDataHandler(DialogueDataHandler):
                    #    class Config(DialogueDataHandler.Config):
                    #        # Support set size per task, i.e. base-learner minibatch size
                    #        support_batch_size: int = 64  # 128
                    #        meta_batch_size: int = 4  # 2
                    losses_q = [0 for ]

                    print("support.__len__() ", support.__len__())
                    for enum_i, ((s_inputs, t_inputs), (s_targets, t_targets), (s_context, t_context)) in enumerate(zip(support, target, context)): # task num
                        # same task
                        support_set = s_inputs
                        target_set = t_inputs
                        # all same domain
                        # support : (2)
                        # s_inputs : (6)
                        # s_inputs[0].shape : (128, 3, 38) # 3 means 3 consecutive sentence ## 'denver', 'no , the thunderstorm has drifted north .', 'that makes me mad ! why is that ?'
                        # s_inputs[1].shape : (128, 3, 38, 768) # I guess BertEmbedding ## Now None!!
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

                        # mldc/data/data_handler.py def _train_input_from_batch(self, batch):
                        # seq_input = getattr(batch, ModelInput.SEQ)  # seq_input (4) # (128, 5, 35), (128) n seqs, (128, 5) n words per seq, None
                        # target = getattr(batch, ModelOutput.TOK)  # (2) (128, 48), (128)
                        # teacher_forcing_input, teacher_forcing_lens = self._make_teacher_forcing(*target)
                        # return (# flatten the seq input into the list of parameters
                        #   seq_input[0],  # (128, 5, 35)
                        #   seq_input[3],  # None
                        #   teacher_forcing_input,
                        #   seq_input[1],  # n seqs
                        #   seq_input[2],  # n words per seq
                        #   teacher_forcing_lens,  # n words per output seq

                        diat = text_embedder.decode_ids_as_text
                        task = t_context['task_id'][0]
                        s_domain = s_context['domain_id'][0]
                        #t_domain = t_context['domain_id'][0]
                        print("b_idx", bidx, "enum_i", enum_i,"s_domain :", s_domain)
                        #print("t_domain :", s_domain)
                        #print("task :", task)
                        # text_embedder.decode_ids_as_text(s_inputs[0][0][0].cpu().numpy()) = 'what is your order number ?'
                        # inputs input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids

                        # TODO
                        num_instance = support_set[0].shape[0]
                        # Adapt the model usingthe support set
                        model.train()
                        #spt_input_ids, spt_mc_token_ids, spt_lm_labels, spt_mc_labels, spt_token_type_ids = support_set
                        #for s_idx, (sii, smti, sll, sml, stti) in enumerate(zip(spt_input_ids, spt_mc_token_ids,
                        #                                                        spt_lm_labels, spt_mc_labels,
                        #                                                        spt_token_type_ids)):
                        for s_idx, support_ins in enumerate(zip(*support_set)):
                            sii, smti, sll, sml, stti = support_ins
                            if model.representation.gptmode == "gpt2":
                                lm_loss, mc_loss, _, _, _ = model(*support_ins)
                            else:
                                lm_loss, mc_loss = model(*support_ins)
                            loss = (lm_loss * 2 + mc_loss * 1)
                            grad = torch.autograd.grad(loss, model.parameters())
                            fast_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grad, model.parameters())))








                            ## input_ids, mc_token_ids=None, lm_labels=None, mc_labels=None, token_type_ids=None,

                        #task_num = s_inputs.shape[0] # batchsz
                        #for task_idx in range(task_num):
                        #  s_inputs_task = s_inputs[task_idx]

                        # Adapt the model using the support set
                        # model.train()
                        # for step in range(1):
                        #   #model.contextualize(s_context)
                        #   #model(*s_inputs, responses=s_targets)  # model remembers responses
                        #   lm_loss, mc_loss, _, _, _ = model(*s_inputs)

                        # # Evaluate the model using the target set
                        # model.eval()    # model now retrieves from examples seen so far
                        # model.contextualize(t_context)
                        # t_pred = model(*t_inputs)
                        # t_loss = model.get_loss(t_pred, t_targets, t_context).item()
                        # metric_reporter.add_batch_stats(task, t_loss, s_inputs,
                        #                                 t_predictions=t_pred, t_targets=t_targets)

                metric_reporter.report_metric(stage=Stage.TRAIN, epoch=epoch, reset=False)

            import ipdb; ipdb.set_trace()
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
