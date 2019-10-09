import os
import torch
import logging
import copy
import random
from typing import Any, Optional, Tuple
import numpy as np
from itertools import chain

from pytext.trainers import Trainer
from pytext.config import PyTextConfig
from pytext.config.pytext_config import ConfigBase
from pytext.common.constants import Stage
from pytext.models.model import Model
from pytext.utils import cuda_utils
from pytext.main import gen_config_impl, train_model, load
from pytext.task import save
from pytext.workflow import save_and_export

from mldc.data.data_handler import BatchPreparationPipeline
from mldc.metrics.metrics import MetaLearnMetricReporter
from pytorch_pretrained_bert import OpenAIGPTDoubleHeadsModel

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu

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

    def meta_test_query_PRINT_preprocess(self, query_ins, tokenizer, byte_decoder):

        input_ids = query_ins[0][0].cpu().numpy().tolist()
        token_type_ids = query_ins[4][0].cpu().numpy().tolist()
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
        token_type_tokens = tokenizer.convert_ids_to_tokens(token_type_ids, skip_special_tokens=False)
        input_tokens = [self.transform_byte2normal(tokenizer, byte_decoder, token) for token in input_tokens]
        token_type_tokens = [self.transform_byte2normal(tokenizer, byte_decoder, token) for token in token_type_tokens]

        return input_tokens, token_type_tokens

    def _clip_grad_norm(self, grads, max_norm, norm_type):
        total_norm = 0
        for g_i, grad in enumerate(grads):
            param_norm = grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for grad in grads:
                grad.data.mul_(clip_coef)
        return grads

    def transform_byte2normal(self, tokenizer, byte_decoder, token):
        if token is None:
            return None
        temp = []
        for tok in token:
            temp.append(byte_decoder[tok])
        temp2 = bytearray(temp).decode('utf-8', errors=tokenizer.errors)
        return temp2

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

    def predict(self,
                text_embedder,
                test_task_iters: BatchPreparationPipeline,
                model: Model,
                metric_reporter: MetaLearnMetricReporter):

        tokenizer = text_embedder.tokenizer
        byte_decoder = tokenizer.byte_decoder

        #import ipdb; ipdb.set_trace()
        #if cuda_utils.CUDA_ENABLED:
        model = model.cuda()

        b_ignore_domain = False

        if b_ignore_domain:
            for bidx, meta_batch in enumerate(test_task_iters):
                support_query, target, context = meta_batch

                for task_i, ((support_set, query_set), (s_targets, t_targets), (s_context, t_context)) in enumerate(zip(support_query, target, context)):
                    s_domain = s_context['domain_id'][0]
                    task_id = t_context['task_id'][0]
                    print("b_idx {} task i {} s_domain {}".format(bidx, task_id, s_domain))

                    import ipdb; ipdb.set_trace()

        # Meta Learning Options
        update_lr = 0.01
        update_step = 1
        # Gpt options
        lm_coef = 2.0
        mc_coef = 1.0
        gradient_accumulation_steps = 8
        # gradient clipping
        max_norm = 1
        norm_type = 2

        for bidx, meta_batch in enumerate(test_task_iters):
            support_query, target, context = meta_batch

            model_params = list(model.parameters())
            original_params = copy.deepcopy(list(model.parameters()))
            task_num = len(support_query)

            for task_i, ((support_set, query_set), (s_targets, t_targets), (s_context, t_context)) in enumerate(zip(support_query, target, context)):
                s_domain = s_context['domain_id'][0]
                task_id = t_context['task_id'][0]
                print("b_idx {} task i ({}/{}) s_domain {}".format(bidx, task_i+1, task_num, s_domain))

                support_set = tuple([torch.squeeze(s, dim=1) for s in support_set])
                query_set = tuple([torch.squeeze(q, dim=1) for q in query_set])


                print("support_set_size", support_set[0].__len__())
                support_sets = self.support_sets_transformation(support_set, spt_batchsz=4, manual_size=-1)
                #support_sets = self.support_sets_transformation(support_set, spt_batchsz=4, manual_size=128)
                meta_test_query_set, labels = self.meta_test_query_set_preprocess(query_set, text_embedder,
                                                                                  byte_decoder, batchsz=1)
                support_sets = tuple([ss.to('cuda') for ss in support_sets])

                # restore original params for next tasks
                for mp_idx, mp in enumerate(model_params):
                    mp.data = original_params[mp_idx]

                for k in range(update_step):
                    support_sets_loss = 0
                    # in-train (support)
                    model.train()
                    print("support_set batch num : {}, batch_size {}".format(len(list(zip(*support_sets))), 4))
                    for si, support_set in enumerate(zip(*(support_sets))):
                        lm_loss, mc_loss, _, _, _ = model(text_embedder, *support_set, mode='teacher')
                        loss = (lm_loss * lm_coef + mc_loss * mc_coef) / gradient_accumulation_steps
                        support_sets_loss += loss.item()
                        # 2. compute grad on theta_pi
                        grad = torch.autograd.grad(loss, model.parameters(), allow_unused=True)
                        # gradient clipping
                        new_grad = self._clip_grad_norm(grad[1:], max_norm, norm_type)
                        grad = tuple([grad[0]] + list(new_grad))
                        # 3. theta_pi = theta_pi - train_lr * grad
                        fast_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grad[1:], model_params[1:])))
                        fast_weights = [model_params[0]] + fast_weights  # model_params[0] is dummy!!!  caution!
                        for mp_idx, mp in enumerate(model_params):
                            mp.data = fast_weights[mp_idx]

                    # in-test (query)
                    # Evaluate the model using the target set
                    #print("meta_test_query")
                    model.eval()
                    assert (meta_test_query_set.__len__() == 1)
                    for q_i, (query_ins, label) in enumerate(zip(meta_test_query_set, labels)):
                        query_ins = tuple([ss.to('cuda') for ss in query_ins if type(ss) == torch.Tensor])
                        query_ins = (query_ins[0], None, None, None, query_ins[1])
                        input_tokens, token_type_tokens = self.meta_test_query_PRINT_preprocess(query_ins, tokenizer, byte_decoder)
                        #print("[q_i]", q_i)
                        #print("[input]", ''.join(input_tokens))
                        sentence, sentence_ids, _, _, _ = model(text_embedder, *query_ins, mode='infer')
                        #print("\n[prdct]", sentence)
                        #print("[label]", label)
                        #print("-" * 200)

                    tokens = text_embedder.decode_ids_as_tokens(sentence_ids)
                    sent = text_embedder.decode_tokens_as_text(tokens)

                    yield dict(task=task_id, resps=np.array([[sentence_ids]]), resp_lens=[len(tokens)],
                               s_inputs=support_sets, s_targets=s_targets, s_context=s_context,
                               t_inputs=query_set, t_targets=t_targets, t_context=t_context)

                # restore original params for next tasks
                for mp_idx, mp in enumerate(model_params):
                    mp.data = original_params[mp_idx]
                #model(*s_inputs, responses=s_targets)  # model remembers responses


                #model.eval()

                #with torch.no_grad():
                #    resps, resp_lens = model(*t_inputs)

                #    yield dict(task=task_id, resps=resps, resp_lens=resp_lens,
                #               s_inputs=s_inputs, s_targets=s_targets, s_context=s_context,
                #               t_inputs=t_inputs, t_targets=t_targets, t_context=t_context)

    def support_sets_transformation(self, support_set, spt_batchsz=4, mc_num=2, max_len=300, manual_size=-1):
        sii, smti, sll, sml, stti = support_set
        batch_size = sii.shape[0]
        if manual_size == -1:
            manual_size = (batch_size // spt_batchsz) * spt_batchsz
        perm = torch.randperm(batch_size)[:manual_size]
        sii = sii[perm]
        smti = smti[perm]
        sll = sll[perm]
        sml = sml[perm]
        stti = stti[perm]
        sii = sii.reshape([-1, spt_batchsz, mc_num, max_len])
        smti = smti.reshape([-1, spt_batchsz, mc_num])
        sll = sll.reshape([-1, spt_batchsz, mc_num, max_len])
        sml = sml.reshape([-1, spt_batchsz])
        stti = stti.reshape([-1, spt_batchsz, mc_num, max_len])
        support_sets = (sii, smti, sll, sml, stti)

        return support_sets

    def meta_train_query_set_preprocess(self, query_set, batchsz=4):
       #print("meta_train query_size :", query_set[0].shape)
        original_batch_size = query_set[0].shape[0]
        perm = random.sample(range(original_batch_size), batchsz)
        #print("meta_train :", perm)
        query_set = tuple([q[perm] for q in query_set])
        query_set_ = []

        for qii, qmti, qll, qml, qtti in zip(*query_set):
            qii = qii[-1]
            qll = qll[-1]
            qtti = qtti[-1]
            query = (qii, None, qll, None, qtti)
            query_set_.append(query)

        query_set = list(zip(*query_set_))
        query_set_ = []
        for q_i, q in enumerate(query_set):
            if q_i in [0, 2, 4]: # input_ids, lm_labels, token_type_ids.
                query_set_.append(torch.stack(q, dim=0))
            else:
                query_set_.append(None)

        return query_set_

    def divide_in_and_out_batchsz(self, spt_or_que_set: torch.Tensor, batchsz=1, mode='support', mc_num=2, max_len=300, manual_size=-1):
        """# mode 'support' or 'meta-train-query' """

        ii, mti, ll, ml, tti = spt_or_que_set
        orig_batchsz = ii.shape[0]
        if manual_size == -1:
            manual_size = (orig_batchsz // batchsz) * batchsz
        perm = torch.randperm(orig_batchsz)[:manual_size]
        spt_or_que_set = tuple([gpt_inp_elem[perm] for gpt_inp_elem in spt_or_que_set])
        ii, mti, ll, ml, tti = spt_or_que_set

        if mode == 'meta-train-query':
            ii = ii[:, 1, :].reshape([ii.shape[0], -1, ii.shape[2]])
            mti = mti[:, 1].reshape([mti.shape[0], -1])
            ll = ll[:, 1, :].reshape([ll.shape[0], -1, ll.shape[2]])
            tti = tti[:, 1, :].reshape([tti.shape[0], -1, tti.shape[2]])
            spt_or_que_set = (ii, mti, ll, ml, tti)
            assert(mc_num == 1)
        elif mode == 'support':
            assert(mc_num == 2)

        ii = ii.reshape([-1, batchsz, mc_num, max_len])
        mti = mti.reshape([-1, batchsz, mc_num])
        ll = ll.reshape([-1, batchsz, mc_num, max_len])
        ml = ml.reshape([-1, batchsz])
        tti = tti.reshape([-1, batchsz, mc_num, max_len])

        if mode == 'support':
            spt_or_que_set = (ii, mti, ll, ml, tti)
        elif mode == 'meta-train-query':
            spt_or_que_set = (ii, [None] * ii.shape[0], ll, [None] * ii.shape[0], tti)

        return spt_or_que_set


    def save_export_filename(self, config: PyTextConfig, task, filepath) -> None:
        print("\n=== Saving model to: " + filepath)
        config.save_snapshot_path = filepath
        save(config, task.model, task.data_handler.metadata_to_save())
        task.export(task.model, config.export_caffe2_path, None, config.export_onnx_path)

    def meta_test_query_set_preprocess(self, query_set, text_embedder, byte_decoder, batchsz=6):

        original_batch_size = query_set[0].shape[0]
        perm = random.sample(range(original_batch_size), batchsz)
        query_set = tuple([q[perm] for q in query_set])

        tokenizer = text_embedder.tokenizer
        query_set_ = []
        labels = []

        # input_ids : (1, N), qmti : None, qll : (1, max_len-N), qml : None, qtti : (1, N)
        for qii, qmti, qll, qml, qtti in zip(*query_set):
            not_pad_part = qtti[1] != text_embedder.pad_idx
            len_except_pad = not_pad_part.sum().item() # N
            reverse_iter = len_except_pad - 1
            if text_embedder.sys_idx == qtti[1][len_except_pad - 1].item():
                while qtti[1][reverse_iter].item() == text_embedder.sys_idx: reverse_iter -= 1
            else:
                while qtti[1][reverse_iter].item() == text_embedder.usr_idx: reverse_iter -= 1
            history = qii[1][:reverse_iter+1]
            history = torch.unsqueeze(history, dim=0)
            label = qii[1][reverse_iter+1:len_except_pad]
            label = torch.unsqueeze(label, dim=0)
            qtti = qtti[1][:reverse_iter+1]
            qtti = torch.unsqueeze(qtti, dim=0)
            assert (history.shape == qtti.shape)
            assert (history.shape[1] == qtti.shape[1] == (self.max_len - label.shape[1]))
            query = (history, None, label, None, qtti)
            query_set_.append(query)
            label = label.numpy().reshape(-1).tolist()
            label_tokens = tokenizer.convert_ids_to_tokens(label, skip_special_tokens=False)
            label_transform = [self.transform_byte2normal(tokenizer, byte_decoder, token) for token in label_tokens]
            labels.append(''.join(label_transform))

        return query_set_, labels

        # labels = []
        # tokenizer = text_embedder.tokenizer

        # for qii, qmti, qll, qml, qtti in zip(*query_set):
        #     not_pad = qtti[-1] != text_embedder.pad_idx
        #     real_len = not_pad.sum().item()
        #     reverse_iter = real_len - 1
        #     if text_embedder.sys_idx == qtti[-1][real_len - 1].item():
        #         while qtti[-1][reverse_iter].item() == text_embedder.sys_idx: reverse_iter -= 1
        #     else:
        #         while qtti[-1][reverse_iter].item() == text_embedder.usr_idx: reverse_iter -= 1
        #     history = qii[-1][:reverse_iter + 1]
        #     label = qii[-1][reverse_iter+1:real_len]
        #     #qmti_temp = torch.tensor(qii_temp.shape[0], device='cuda').unsqueeze(0)
        #     qtti = qtti[-1][:reverse_iter + 1]
        #     query = (torch.unsqueeze(history, dim=0), None, None, None, torch.unsqueeze(qtti, dim=0))
        #     query_set_.append(query)
        #     label = label.cpu().numpy().tolist()
        #     label_tokens = tokenizer.convert_ids_to_tokens(label, skip_special_tokens=False)
        #     label_transform = [self.transform_byte2normal(tokenizer, byte_decoder, token) for token in label_tokens]
        #     labels.append(''.join(label_transform))
        #
        # return query_set_, labels


    def train(
            self,
            task,
            text_embedder,
            #train_task_iters: Optional[BatchPreparationPipeline],    # Optional[X] is equivalent to Union[X, None].
            train_task_iters,    # Optional[X] is equivalent to Union[X, None].
            eval_task_iters: BatchPreparationPipeline,
            model: Model,
            metric_reporter: MetaLearnMetricReporter,
            train_config: PyTextConfig,
            rank: int = 0,
    ) -> Tuple[torch.nn.Module, Any]:

        tokenizer = text_embedder.tokenizer
        self.tokenizer =tokenizer
        byte_decoder = tokenizer.byte_decoder
        self.byte_decoder = byte_decoder
        self.max_len = 300
        self.text_embedder = text_embedder
        self.pad_token_idx = text_embedder.pad_idx

        def gptinput2tokens(support_set, batch_id=0):
            """ Until now, this function only support "support set", not query set """
            """ Debugging purpose """

            def gputensor2tokens(tensor_gpu):
                ids = tensor_gpu.cpu().numpy().tolist()
                byte_tokens = self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
                tokens = [self.transform_byte2normal(self.tokenizer, self.byte_decoder, token) for token in byte_tokens]
                return tokens

            neg_inids_toks = gputensor2tokens(support_set[0][batch_id][0])
            pos_inids_toks = gputensor2tokens(support_set[0][batch_id][1])
            neg_mc_id = support_set[1][batch_id][0].item()
            pos_mc_id = support_set[1][batch_id][1].item()
            neg_lm_toks = gputensor2tokens(support_set[2][batch_id][0]) 
            pos_lm_toks = gputensor2tokens(support_set[2][batch_id][1])
            neg_tt_toks = gputensor2tokens(support_set[4][batch_id][0])
            pos_tt_toks = gputensor2tokens(support_set[4][batch_id][1])

            neg_imlt = (neg_inids_toks, neg_tt_toks, neg_lm_toks, neg_mc_id)
            pos_imlt = (pos_inids_toks, pos_tt_toks, pos_lm_toks, pos_mc_id)
            return neg_imlt, pos_imlt

        def print_toks(input_tokens, token_type_labels, lm_labels=None, mc_token_ids=None, max_print_length=120):
            """print tokens with pretty way"""
            """ Debugging purpose"""

            num_toks_per_line = 20  # number of tokens per one line
            one_word_size = 8
            frommin2max = list(range(0, max_print_length + 1, num_toks_per_line))  # +1 is just trick
            min_maxs = []
            for _ in range(max_print_length // num_toks_per_line):
                min_maxs.append((frommin2max[_], frommin2max[_ + 1]))

            for mini, maxi in min_maxs:
                for num_idx in range(mini, maxi):  # number
                    print('{:8d}'.format(num_idx), end='')  ## depend on one_word_size
                print()

                for idx in range(mini, maxi):
                    nii = input_tokens[idx]
                    if idx == max_print_length:
                        break
                    print('{}'.format(nii[:one_word_size].rjust(one_word_size)), end='')
                print()

                for idx in range(mini, maxi):
                    ntl = token_type_labels[idx]
                    if idx == max_print_length:
                        break
                    print('{}'.format(ntl[:one_word_size].rjust(one_word_size)), end='')
                print()

                if lm_labels:
                    for idx in range(mini, maxi):
                        nll = lm_labels[idx]
                        if idx == max_print_length:
                            break
                        if nll:
                            print('{}'.format(nll[:one_word_size].rjust(one_word_size)), end='')
                        else:
                            print('{}'.format('-1'.rjust(one_word_size)), end='')
                    print()

                if mc_token_ids:
                    for idx in range(mini, maxi):  # mc_token
                        if mc_token_ids == idx:
                            print('{}'.format('<mctok>'.rjust(one_word_size)), end='')
                        else:
                            print('{}'.format('-nomc-'.rjust(one_word_size)), end='')
                    print()
                print('-' * (one_word_size * num_toks_per_line))


        #[INFO] This is the first line of function "train"
        learning_options = ['meta-learning', 'ignore-domain']
        if type(train_task_iters) != BatchPreparationPipeline:
            learning_op = learning_options[1] # meta-learning method
        else:
            learning_op = learning_options[0] # domain-agnostic, ignoring domain

        if learning_op == 'ignore-domain':
            """ in ignore_domain case, train_task_iters is not BatchPreparationPipeline, but Dataset.
                So we need to dataset preprocessing...
            """

            # Data processing phase
            filename = "whole_ignore_domain.pickle"#filename = 'ALARM_SET_for_ignore_domain.pickle'#filename = 'whole_train_eval.pickle'
            LOG.info("checking load file name {}".format(filename))
            # If not save file("filename"), Make it, otherwise Loading "filename"
            if not os.path.exists(filename):
            #if True: # you always make it.

                # This is create gpt_input_unit function. skip it.
                def create_gpt_input_unit(history, pos_resp, neg_resp,
                                          history_turn_types, presp_turn_type, nresp_turn_type):
                    """ this function only for ignore domain case """

                    assert (len(list(chain(*history))) == len(list(chain(*history_turn_types))))
                    assert (len(pos_resp) == len(presp_turn_type))
                    assert (len(neg_resp) == len(nresp_turn_type))

                    input_ids, mc_token_ids, lm_labels, mc_labels, turn_type_ids = [], [], [], [], []

                    pos_seq = history + [pos_resp]
                    pos_turn_types = history_turn_types + [presp_turn_type]

                    pos_input_ids = list(chain(*pos_seq))
                    pos_mc_token_ids = len(pos_input_ids) - 1
                    pos_lm_labels = [-1] * len(list(chain(*history))) + pos_resp  #
                    pos_token_types = list(chain(*pos_turn_types))
                    try:
                        assert (len(pos_input_ids) == len(pos_lm_labels))
                        assert (len(pos_input_ids) == len(pos_token_types))
                    except:
                        print("len(pos_input_ids): ", len(pos_input_ids))
                        print("len(pos_lm_labels): ", len(pos_lm_labels))
                        print("len(pos_token_types): ", len(pos_token_types))

                    if len(pos_input_ids) > self.max_len:
                        excess = len(pos_input_ids) - self.max_len
                        pos_input_ids = pos_input_ids[-self.max_len:]
                        pos_input_ids2 = pos_input_ids[excess:]
                        if pos_input_ids == pos_input_ids2:
                            print("real True")
                        # print("origin", pos_mc_token_ids)
                        pos_mc_token_ids = pos_mc_token_ids - excess
                        # print("now", pos_mc_token_ids)
                        pos_lm_labels = pos_lm_labels[-self.max_len:]
                        pos_token_types = pos_token_types[-self.max_len:]
                    else:
                        num_pad = self.max_len - len(pos_input_ids)
                        pos_input_ids = pos_input_ids + [self.pad_token_idx] * num_pad
                        pos_lm_labels = pos_lm_labels + [-1] * num_pad
                        pos_token_types = pos_token_types + [self.pad_token_idx] * num_pad

                    assert (len(pos_input_ids) == self.max_len)
                    assert (pos_mc_token_ids < self.max_len)
                    assert (len(pos_lm_labels) == self.max_len)
                    assert (len(pos_token_types) == self.max_len)

                    neg_seq = history + [neg_resp]
                    neg_turn_types = history_turn_types + [nresp_turn_type]

                    neg_input_ids = list(chain(*neg_seq))
                    neg_mc_token_ids = len(neg_input_ids) - 1
                    neg_lm_labels = [-1] * len(neg_input_ids)
                    neg_token_types = list(chain(*neg_turn_types))
                    try:
                        assert (len(neg_input_ids) == len(neg_lm_labels))
                        assert (len(neg_input_ids) == len(neg_token_types))
                    except:
                        print("len(neg_input_ids): ", len(neg_input_ids))
                        print("len(neg_lm_labels): ", len(neg_lm_labels))
                        print("len(neg_token_types): ", len(neg_token_types))

                    if len(neg_input_ids) > self.max_len:
                        excess = len(neg_input_ids) - self.max_len
                        neg_input_ids = neg_input_ids[-self.max_len:]
                        neg_mc_token_ids = neg_mc_token_ids - excess
                        neg_lm_labels = neg_lm_labels[-self.max_len:]
                        neg_token_types = neg_token_types[-self.max_len:]
                    else:
                        num_pad = self.max_len - len(neg_input_ids)
                        neg_input_ids = neg_input_ids + [self.pad_token_idx] * num_pad
                        neg_lm_labels = neg_lm_labels + [-1] * num_pad
                        neg_token_types = neg_token_types + [self.pad_token_idx] * num_pad

                    assert (len(neg_input_ids) == self.max_len)
                    assert (neg_mc_token_ids < self.max_len)
                    assert (len(neg_token_types) == self.max_len)
                    assert (len(neg_lm_labels) == self.max_len)

                    input_ids.append(neg_input_ids)
                    input_ids.append(pos_input_ids)
                    mc_token_ids.append(neg_mc_token_ids)
                    mc_token_ids.append(pos_mc_token_ids)
                    lm_labels.append(neg_lm_labels)
                    lm_labels.append(pos_lm_labels)
                    pos_mc_labels = 1
                    mc_labels.append(pos_mc_labels)
                    turn_type_ids.append(neg_token_types)
                    turn_type_ids.append(pos_token_types)

                    return input_ids, mc_token_ids, lm_labels, mc_labels, turn_type_ids

                # Create each instance from examples
                def create_instance_dict(examples, domains, fixed_n_turns, n_turns, all_responses):
                    """ create instance dict considering domains"""

                    new_domains = []
                    for dom in domains:
                        new_domains.append(dom.split('dialogues/')[1].split('.txt')[0])

                    history_mat, pos_resp_mat, neg_resp_mat = [], [], []
                    history_turn_type_mat, pos_resp_turn_type_mat, neg_resp_turn_type_mat = [], [], []
                    raw_history_mat, raw_pos_resp_mat, raw_neg_resp_mat = [], [], []
                    
                    LOG.info("Create instance_dict related to {}".format(domains))
                    LOG.info("examples len {}".format(len(examples)))
                    for exam_i, example in enumerate(examples):

                        if not example.domain_id in new_domains: # This means We will pick example only related to domain
                            continue

                        pos_turns = example.seq_word_feat
                        neg_turns = example.neg_seq_word_feat
                        raw_pos_turns = example.orig_text
                        raw_neg_turns = example.neg_orig_text

                        # ensure that the last turn is a user turn, i.e. the number of turns is even
                        if len(pos_turns) % 2 == 1:
                            pos_turns = pos_turns[:-1]
                            neg_turns = neg_turns[:-1]
                            raw_pos_turns = raw_pos_turns[:-1]
                            raw_neg_turns = raw_neg_turns[:-1]

                        if fixed_n_turns:
                            endpoint = len(pos_turns) - 1
                            startpoint = 0
                        elif len(pos_turns) >= n_turns:
                            endpoint = np.random.randint(n_turns - 1, len(pos_turns))
                            startpoint = endpoint - n_turns + 1
                        else:
                            endpoint = len(pos_turns) - 1
                            startpoints = list(range(endpoint - 1, -1, -2))
                            startpoint = startpoints[np.random.randint(0, len(startpoints))]

                        # INPUT
                        history_turns = [turn.tolist()[self.text_embedder.pieces_slice] for turn in pos_turns[startpoint:endpoint]]
                        raw_history_turns = [turn[self.text_embedder.pieces_slice] for turn in raw_pos_turns[startpoint:endpoint]]
                        history_mat.append(history_turns)
                        raw_history_mat.append(raw_history_turns)
                        input_token_type_id = []
                        for turn in history_turns:
                            speaker = self.text_embedder.decode_ids_as_text([turn[0]])
                            if speaker == '<system>':
                                input_token_type_id.append([self.text_embedder.sys_idx] * len(turn))
                            elif speaker == '<user>':
                                input_token_type_id.append([self.text_embedder.usr_idx] * len(turn))
                        history_turn_type_mat.append(input_token_type_id)

                        # TARGET
                        if all_responses: # answers to all input turns
                            ep_slice = slice(startpoint + 1, endpoint + 1, 2)
                        else: # answer to last input turn
                            ep_slice = slice(endpoint, endpoint + 1)

                        pos_resps = [turn.tolist()[self.text_embedder.pieces_slice] for turn in  pos_turns[ep_slice]]
                        pos_resps_raw = [turn[self.text_embedder.pieces_slice] for turn in raw_pos_turns[ep_slice]]
                        pos_resp_mat.append(pos_resps)
                        raw_pos_resp_mat.append(pos_resps_raw)
                        target_token_type_id = []
                        for resp in pos_resps:
                            speaker = self.text_embedder.decode_ids_as_text([resp[0]])
                            if speaker == '<system>': target_token_type_id.append([self.text_embedder.sys_idx] * len(resp))
                            elif speaker == '<user>': target_token_type_id.append([self.text_embedder.usr_idx] * len(resp))
                        pos_resp_turn_type_mat.append(target_token_type_id)

                        neg_resps = [turn.tolist()[self.text_embedder.pieces_slice] for turn in neg_turns[ep_slice]]
                        neg_resps_raw = [turn[self.text_embedder.pieces_slice] for turn in raw_neg_turns[ep_slice]]
                        raw_neg_resp_mat.append(neg_resps_raw)
                        neg_resp_mat.append(neg_resps)
                        neg_tar_token_type_id = []
                        for resp in neg_resps:
                            speaker = self.text_embedder.decode_ids_as_text([resp[0]])
                            if speaker == '<system>': neg_tar_token_type_id.append([self.text_embedder.sys_idx] * len(resp))
                            elif speaker == '<user>': neg_tar_token_type_id.append([self.text_embedder.usr_idx] * len(resp))
                        neg_resp_turn_type_mat.append(neg_tar_token_type_id)

                    # end for
                    matrix_iterator = zip(history_mat, pos_resp_mat, neg_resp_mat, raw_history_mat, raw_pos_resp_mat,
                                          raw_neg_resp_mat,
                                          history_turn_type_mat, pos_resp_turn_type_mat, neg_resp_turn_type_mat)
                    LOG.info("\n\t\toriginal of Dialogs : {}".format(len(history_mat)))
                    instance_dict = {'input_ids': [], "mc_token_ids": [], "lm_labels": [], "mc_labels": [],
                                     "token_type_ids": []}
                    for history, pos_resps, neg_resps, raw_history, raw_pos_resps, raw_neg_resps, \
                        history_turn_types, pos_resp_turn_types, neg_resp_turn_types in matrix_iterator:

                        if all_responses:
                            resps_zip = zip(pos_resps, raw_pos_resps, neg_resps, raw_neg_resps, pos_resp_turn_types,
                                            neg_resp_turn_types)
                            for resp_idx, (
                            pos_resp, rpos_resp, neg_resp, rneg_resp, presp_turn_type, nresp_turn_type) in enumerate(resps_zip):
                                history_idx = 2 * resp_idx + 1
                                this_history = history[:history_idx]
                                this_raw_history = raw_history[:history_idx]
                                this_history_turn_types = history_turn_types[:history_idx]
                                input_ids, mc_token_ids, lm_labels, mc_labels, turn_type_ids = create_gpt_input_unit(
                                    this_history, pos_resp, neg_resp,
                                    this_history_turn_types, presp_turn_type, nresp_turn_type)
                                instance_dict['input_ids'].append(input_ids)
                                instance_dict['mc_token_ids'].append(mc_token_ids)
                                instance_dict['lm_labels'].append(lm_labels)
                                instance_dict['mc_labels'].append(mc_labels)
                                instance_dict['token_type_ids'].append(turn_type_ids)
                        else:
                            assert (len(pos_resps) == 1)
                            resp_tuple = next(zip(pos_resps, raw_pos_resps, neg_resps, raw_neg_resps, pos_resp_turn_types,
                                                  neg_resp_turn_types))
                            pos_resp, rpos_resp, neg_resp, rneg_resp, presp_turn_type, nresp_turn_type = resp_tuple
                            input_ids, mc_token_ids, lm_labels, mc_labels, turn_type_ids = create_gpt_input_unit(
                                                                        history, pos_resp, neg_resp,
                                                                        history_turn_types, presp_turn_type, nresp_turn_type)
                            instance_dict['input_ids'].append(input_ids)
                            instance_dict['mc_token_ids'].append(mc_token_ids)
                            instance_dict['lm_labels'].append(lm_labels)
                            instance_dict['mc_labels'].append(mc_labels)
                            instance_dict['token_type_ids'].append(turn_type_ids)

                    LOG.info('\n\t\tTotal data instance count : {}'.format(instance_dict['input_ids'].__len__()))
                    return instance_dict
                # END of create_instance_dict

                ## Beginning of ignore domain
                LOG.info("\n\tMaking dataset.....")
                dataset = train_task_iters
                examples = dataset.examples

                #train_domains = ['dialogues/ALARM_SET.txt']
                #eval_domains = ['dialogues/ALARM_SET.txt']
                domains = task.data_handler.train_domains
                eval_domains = ['dialogues/ALARM_SET.txt', 'dialogues/AGREEMENT_BOT.txt', 'dialogues/EDIT_PLAYLIST.txt']
                train_domains = list(set(domains) - set(eval_domains))
                fixed_n_turns = True # if true, consider all dlgs!
                n_turns = 20         # if fixed_n_turns True, meaningless, otherwise, max_length of dlgs
                all_responses = True

                LOG.info("\n\ttrain_domains {}\n\n\teval_domains {}\n\n\tfixed_n_turns {}\n\n\tall_responses {}".format(
                    train_domains, eval_domains, fixed_n_turns, all_responses))
                #LOG.info("\n\t\t Are you sure this options?")
                #import ipdb; ipdb.set_trace()

                train_instance_dict = create_instance_dict(examples, train_domains, fixed_n_turns, n_turns, all_responses)
                eval_instance_dict = create_instance_dict(examples, eval_domains, fixed_n_turns, n_turns, all_responses)
                import pickle
                with open(filename, 'wb') as f:
                    pickle.dump([train_instance_dict, eval_instance_dict], f)
            else:
                import pickle
                with open(filename, 'rb') as f:
                    print("loading {}".format(filename))
                    train_instance_dict, eval_instance_dict = pickle.load(f)
            # Data preprocessing done!
            # only you have to remember is,
            #   "train_instance_dict", and "eval_instance_dict"

            # function definition for input preprocessing for torch model and validation process
            def get_train_batches(modif_train_num, train_input_ids, train_mc_token_ids, train_lm_labels, train_mc_labels, train_token_type_ids):

                # Get train batches [permutation & reshape considersing batch_size]
                perm = np.arange(modif_train_num)
                random.shuffle(perm)
                input_ids = train_input_ids[perm]
                mc_token_ids = train_mc_token_ids[perm]
                lm_labels = train_lm_labels[perm]
                mc_labels = train_mc_labels[perm]
                token_type_ids = train_token_type_ids[perm]
                # Reshaping
                input_ids = input_ids.reshape(-1, this_batch_size, input_ids.shape[1], input_ids.shape[2])
                mc_token_ids = mc_token_ids.reshape(-1, this_batch_size, mc_token_ids.shape[1])
                lm_labels = lm_labels.reshape(-1, this_batch_size, lm_labels.shape[1], lm_labels.shape[2])
                mc_labels = mc_labels.reshape(-1, this_batch_size, mc_labels.shape[1])
                token_type_ids = token_type_ids.reshape(-1, this_batch_size, token_type_ids.shape[1], token_type_ids.shape[2])
                assert(input_ids.shape[0] == mc_token_ids.shape[0] == lm_labels.shape[0] == mc_labels.shape[0] == token_type_ids.shape[0])
                gpt_input_zip = zip(input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids)

                return gpt_input_zip

            def create_eval_input_set(total_eval_num, selected_eval_num, eval_input_ids, eval_mc_token_ids, eval_lm_labels, eval_token_type_ids):

                eval_perm = np.arange(total_eval_num)
                random.shuffle(eval_perm)

                # Select only N cases (eval data)
                eval_perm = eval_perm[:selected_eval_num]

                this_eval_input_ids = np.expand_dims(eval_input_ids[eval_perm][:, 1, :], axis=1) # we need only positive one
                this_eval_mc_token_ids = np.expand_dims(eval_mc_token_ids[eval_perm][:, 1], axis=1)
                this_eval_lm_labels = np.expand_dims(eval_lm_labels[eval_perm][:, 1, :], axis=1)
                this_eval_token_type_ids = np.expand_dims(eval_token_type_ids[eval_perm][:, 1, :], axis=1)

                # Make eval_collect
                eval_set = zip(this_eval_input_ids, this_eval_mc_token_ids, this_eval_lm_labels, this_eval_token_type_ids)
                eval_collect = []
                for eval_ii, eval_mc, eval_ll, eval_tti in eval_set:
                    necessary_idx = np.where(eval_tti != text_embedder.pad_idx)
                    eval_ii = eval_ii.reshape(-1) # (300)
                    eval_ll_for_cut = eval_ll[necessary_idx] # (21, )
                    eval_ll = eval_ll.reshape(-1)
                    eval_tti = eval_tti[necessary_idx] # (21, )
                    # Get border of history and true response
                    ll_idx = np.where(eval_ll_for_cut != -1)
                    border = min(ll_idx[0])
                    #response_final_border = max(ll_idx[0])
                    eval_history = eval_ii[:border]
                    eval_history_token_type = eval_tti[:border]
                    eval_resp_near_m1 = eval_ll[border:]
                    assert(eval_history.shape[0] == eval_history_token_type.shape[0] == eval_ii.shape[0] - eval_resp_near_m1.shape[0])
                    eval_collect.append((eval_history, eval_mc, eval_history_token_type, eval_resp_near_m1))

                return eval_collect

            def valid_process(model, eval_input_zip, tokenizer, byte_decoder):

                reference_matrix = []
                hypotheses = []
                total_valid_losses = 0
                try:
                    model.eval()
                    zip_len = len(list(eval_input_zip))
                    for eval_i, (eval_hist, eval_mc, eval_hist_tt, eval_response_rear_m1) in enumerate(eval_input_zip):
                        if eval_i % 50 == 0:
                            print ("eval_iter : {}/{}".format(eval_i, zip_len))
                        eval_hist = torch.from_numpy(eval_hist).type(torch.LongTensor).to('cuda').unsqueeze(0); # (1, 24)
                        eval_hist_tt = torch.from_numpy(eval_hist_tt).type(torch.LongTensor).to('cuda').unsqueeze(0); # (1, 24)
                        eval_response_rear_m1 = torch.from_numpy(eval_response_rear_m1).type(torch.LongTensor).to('cuda').unsqueeze(0); # (1, 276)
                        #eval_mc = torch.from_numpy(eval_mc).type(torch.LongTensor).to('cuda'); #
                        gpt_input = (eval_hist, None, eval_response_rear_m1, None, eval_hist_tt)

                        input_tokens, token_type_tokens = self.meta_test_query_PRINT_preprocess(gpt_input, tokenizer, byte_decoder)
                        eval_response_no_minus1 = eval_response_rear_m1[eval_response_rear_m1 != -1]
                        eval_resp_no_minus1_tokens = tokenizer.convert_ids_to_tokens(eval_response_no_minus1.tolist(), skip_special_tokens=False)
                        eval_resp_no_minus1_tokens = [self.transform_byte2normal(tokenizer, byte_decoder, token) for token in eval_resp_no_minus1_tokens]
                        #print_toks(input_tokens, token_type_tokens)

                        # Input length checking
                        assert (eval_hist.shape[1] == eval_hist_tt.shape[1] == (self.max_len - eval_response_rear_m1.shape[1]) )

                        try:
                            valid_lm_loss, sentence, sentence_tokens, _, _ = model(text_embedder, *gpt_input, mode='infer')
                        except:
                            print("got u")
                            import ipdb; ipdb.set_trace()
                            valid_lm_loss, sentence, sentence_tokens, _, _ = model(text_embedder, *gpt_input, mode='infer')
                            import ipdb; ipdb.set_trace()
                        total_valid_losses += valid_lm_loss.item()

                        if eval_i == 0:
                            print("eval_i : {}".format(eval_i))
                            print("[input] :", "".join(input_tokens))
                            print("[prdct] :", sentence)
                            print("[label] :", "".join(eval_resp_no_minus1_tokens))
                            score1 = sentence_bleu([eval_resp_no_minus1_tokens], sentence_tokens, smoothing_function=SmoothingFunction().method1)
                            score2 = sentence_bleu([eval_resp_no_minus1_tokens], sentence_tokens, smoothing_function=SmoothingFunction().method2)
                            score3 = sentence_bleu([eval_resp_no_minus1_tokens], sentence_tokens, smoothing_function=SmoothingFunction().method3)
                            score4 = sentence_bleu([eval_resp_no_minus1_tokens], sentence_tokens, smoothing_function=SmoothingFunction().method4)
                            print("[sentence level bleu score] : {:.3f} {:.3f} {:.3f} {:.3f}".format(score1, score2, score3, score4))
                            print('-' * 150)
                            print()

                        reference_matrix.append([eval_resp_no_minus1_tokens])
                        hypotheses.append(sentence_tokens)
                except:
                    import ipdb; ipdb.set_trace()

                score1 = corpus_bleu(reference_matrix, hypotheses, smoothing_function=SmoothingFunction().method1)
                score2 = corpus_bleu(reference_matrix, hypotheses, smoothing_function=SmoothingFunction().method2)
                score3 = corpus_bleu(reference_matrix, hypotheses, smoothing_function=SmoothingFunction().method3)
                score4 = corpus_bleu(reference_matrix, hypotheses, smoothing_function=SmoothingFunction().method4)
                print("[corpus level bleu score] : {:.3f} {:.3f} {:.3f} {:.3f}".format(score1, score2, score3, score4))
                print("eval_i(for debugging)" , eval_i)
                valid_loss = total_valid_losses / (eval_i + 1) # real number
                metric_reporter.add_batch_stats("-ignore-domain-", valid_loss, 1)

                return valid_loss
            # function definition done!

            # Get train batches
            train_input_ids = np.array(train_instance_dict['input_ids'])
            train_mc_token_ids = np.array(train_instance_dict['mc_token_ids'])
            train_lm_labels = np.array(train_instance_dict['lm_labels'])
            train_mc_labels = np.array(train_instance_dict['mc_labels'])
            train_token_type_ids = np.array(train_instance_dict['token_type_ids'])
            this_batch_size = 4
            train_num = train_input_ids.shape[0]
            train_batch_num = train_num // this_batch_size
            modif_train_num = train_batch_num * this_batch_size  # divide batch_size and make it fit.
            LOG.info("\n\tconsidering batch size, train_num {}".format(modif_train_num))

            # Get eval data batches [Select only N cases / make "eval_set"]
            eval_input_ids = np.array(eval_instance_dict['input_ids'])
            eval_mc_token_ids = np.array(eval_instance_dict['mc_token_ids'])
            eval_lm_labels = np.array(eval_instance_dict['lm_labels'])
            eval_token_type_ids = np.array(eval_instance_dict['token_type_ids'])
            total_eval_num = eval_input_ids.shape[0]
            LOG.info("\n\ttotal_eval_num {}".format(total_eval_num))
            # fixed!!

            # CUDA Enable check
            if cuda_utils.CUDA_ENABLED:
                model = model.cuda()
            best_model_path = None

            # TODO Options (ignore-domain)
            # Options [IMPORTANT]
            learning_rate = 0.01
            # Gpt options
            lm_coef = 3.0
            mc_coef = 1.0
            # gradient clipping options
            max_norm = 1.0
            # grad accum option
            gradient_accumulation_steps = 8
            # loading option
            b_model_load = False
            loading_epoch = 0 # If you want to train from initial weights, please change it to 0
            loading_gpt_idx = 0#loading_gpt_idx = 12500
            loading_total_iter = 0#loading_total_iter = 12500
            # training option(do train or not)
            b_train = True
            # eval option
            eval_interval = 1000
            total_eval_num = eval_input_ids.shape[0]
            selected_eval_num = total_eval_num // 10
            # save option
            save_interval = 2500
            LOG.info("\n\tignore domain")
            LOG.info("""\n\tlearning_rate :{}\n\tlm_coef :{}\n\tmc_coef :{}\n\tmax_norm_for_grad_clip :{}\n\tgrad_accums :{}\n\tloading_epoch :{}
                        \n\tmodel_loading :{}\n\tDo Training :{}\n\teval_interval :{}\n\ttotal_eval_num :{}\n\tselect_eval_num :{}
                        \n\tsave_interval :{}""".format(
                            learning_rate, lm_coef, mc_coef, max_norm, gradient_accumulation_steps, loading_epoch,
                            b_model_load, b_train, eval_interval, total_eval_num, selected_eval_num,
                            save_interval
                        ))
            LOG.info("Are you sure with this options? (If you don't need ipdb, please annotate it!)")
            if loading_epoch != 0:
                LOG.info("\n\n\t\tIf you want to train from initial weights, please change it to 0")
                LOG.info("\n\t\tI will load model_epoch{}.pt".format(loading_epoch))

            total_iteration = None
            if b_model_load:
                #Before loading model, release gpu allocation of prev model
                model = model.cpu()
                torch.cuda.empty_cache()
                del model
                intermediate_model_path = os.path.join(
                    train_config.modules_save_dir, "model_epoch{}_gptidx{}_totaliter{}.pt".format(loading_epoch, loading_gpt_idx, loading_total_iter))
                LOG.info('Model loading... {}'.format(intermediate_model_path))
                task, _ = load(intermediate_model_path)
                model = task.model
                LOG.info('Model Loading Success!')
                LOG.info("you need to manage total_iteration !! (tensorboard )")
                total_iteration = loading_total_iter
            else:
                total_iteration = 0

            # Optimizer (Optimizer is different according to gpt version)
            from pytorch_transformers import AdamW
            optimizer = AdamW(model.parameters(), lr=learning_rate)
            optimizer.zero_grad()

            # Considering loading_epoch, Training & Evaluation Phase
            for epoch in range(loading_epoch, 100):
                if b_train:
                    logging.info("Training model on train tasks")
                    gpt_input_zip = get_train_batches(modif_train_num, train_input_ids, train_mc_token_ids, train_lm_labels, train_mc_labels, train_token_type_ids)

                    #[Train] ignore-domain Train Loop
                    loss = None
                    for gpt_input_idx, gpt_input in enumerate(gpt_input_zip):

                        # skip until matching gpt_input_idx == loading_gpt_idx
                        if gpt_input_idx < loading_gpt_idx:
                            continue

                        # Tensoring & get device (cuda)
                        gpt_input = tuple([torch.from_numpy(inp) for inp in gpt_input])
                        gpt_input = tuple([inp.type(torch.LongTensor) for inp in gpt_input])
                        gpt_input = tuple([inp.to('cuda') for inp in gpt_input])

                        # Train
                        model.train()
                        lm_loss, mc_loss, _, _, _ = model(text_embedder, *gpt_input, mode='teacher') # teacher means teacher forcing (for training)
                        loss = (lm_loss * lm_coef + mc_loss * mc_coef) / gradient_accumulation_steps
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

                        if (gpt_input_idx + 1) % gradient_accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                        torch.cuda.empty_cache()

                        # (Tensorboard) ignore-domain metric report
                        #metric_reporter.add_batch_stats('-ignore-train-', loss.item(), 1) # dummy number
                        # Since validation dataset doesn't have mc_loss
                        metric_reporter.add_batch_stats('-ignore-train-', lm_loss.item(), 1) # dummy number
                        if (gpt_input_idx % 50 == 0 and gpt_input_idx != 0) or gpt_input_idx == 1:
                            #metric_reporter.report_metric(Stage.TRAIN, epoch, reset=False)
                            print("epoch : {} | idx : ({}/{}) | loss(not accumed) : {}".format(epoch, gpt_input_idx, train_batch_num, loss.item()))
                            metric_reporter.report_metric(Stage.TRAIN, total_iteration, reset=False) # this loss would be average internally

                        # Validation (eval_interval : 2500)
                        if gpt_input_idx % eval_interval == 0 and gpt_input_idx != 0:
                            logging.info('Evaluating model on eval tasks')
                            eval_input_zip = create_eval_input_set(total_eval_num, selected_eval_num,
                                                                   eval_input_ids, eval_mc_token_ids, eval_lm_labels, eval_token_type_ids)
                            valid_loss = valid_process(model, eval_input_zip, tokenizer, byte_decoder)
                            LOG.info("epoch {} valid_loss {}".format(loading_epoch, valid_loss))
                            # valid_loss -> lm_loss
                            metric_reporter.add_batch_stats('-ignore-valid-', valid_loss, 1)  # dummy number
                            #metric_reporter.report_metric(Stage.EVAL, epoch, reset=False)
                            metric_reporter.report_metric(Stage.EVAL, total_iteration, reset=False)

                        # Save
                        if gpt_input_idx % save_interval == 0 and gpt_input_idx != 0:
                            logging.info("Saving... at {}".format(gpt_input_idx))
                            intermediate_model_path = os.path.join(train_config.modules_save_dir, "model_epoch{}_gptidx{}_totaliter{}.pt".format(
                                epoch, gpt_input_idx, total_iteration))
                            torch.save(model.state_dict(), intermediate_model_path)
                            task.model = model
                            self.save_export_filename(train_config, task, intermediate_model_path)

                        # total_iteration managing
                        total_iteration += 1

                    # one epoch train done
                    # Save
                    LOG.info("Save... one epoch {}".format(epoch))
                    intermediate_model_path = os.path.join(train_config.modules_save_dir, "model_epoch{}_totaliter{}.pt".format(epoch, total_iteration))
                    torch.save(model.state_dict(), intermediate_model_path)
                    task.model = model
                    self.save_export_filename(train_config, task, intermediate_model_path)

                # Validation
                logging.info('Evaluating model on validation tasks')
                eval_input_zip = create_eval_input_set(total_eval_num, selected_eval_num,
                                                       eval_input_ids, eval_mc_token_ids, eval_lm_labels, eval_token_type_ids)
                valid_loss = valid_process(model, eval_input_zip, tokenizer, byte_decoder)
                LOG.info("epoch {} total_iter {} valid_loss {}".format(loading_epoch, total_iteration, valid_loss))

                metric_reporter.add_batch_stats('-ignore-valid-', valid_loss, 1)  # dummy number
                metric_reporter.report_metric(Stage.EVAL, total_iteration, reset=False)

            best_model_path = os.path.join(
                train_config.modules_save_dir, "model.pt"
            )
            torch.save(model.state_dict(), best_model_path)
            train_config.save_snapshot_path = best_model_path
            print("return!")
            return model, None




# 'ignore-domain'
#-----------------------------------------------------------------------------------------------------------------------------------
# meta-learning



        # Upper code is related to "ignore-domain" case
        # Below code is about meta-learning !
        elif learning_op == 'meta-learning':

            # CUDA Enable check
            if cuda_utils.CUDA_ENABLED:
                model = model.cuda()
            best_model_path = None

            # TODO Options (meta-leanring)
            # Options [IMPORTANT]
            # Meta Learning Options
            meta_lr = 0.001 # outer loop learning rate
            update_lr = 0.01 # inner loop learning rate
            update_step = 1 # task-level inner update steps # how many time train from support_sets
            spt_in_batchsz = 1
            qry_in_batchsz = 1
            spt_out_batchsz = 128 // spt_in_batchsz
            qry_out_batchsz = 128 // qry_in_batchsz
            # Gpt options
            lm_coef = 2.0
            mc_coef = 1.0
            # gradient clipping options
            max_norm = 1
            norm_type = 2
            # gradient accum steps
            gradient_accumulation_steps = 1
            # model loading options
            loading_epoch = 0
            b_model_load = False
            # model training options
            b_meta_train = False
            # How many iteration you want in one epoch.
            train_iter_size = 100 # cost = train_iter_size * task_num * (support_set + query_set)
            eval_iter_size = 1    # cost = eval_iter_size * task_num * (support_set + query_set)

            LOG.info("meta-learning method")
            print("meta_lr(outer_lr) : {}\nupdate_lr(inner_lr) : {}\nupdate_step : {}\n \
                  lm_coef : {}\nmc_coef : {}\nmax_norm_for_grad_clip : {}\ngrad_accums : {}\nloading_epoch : {}".format(
                meta_lr, update_lr, update_step, lm_coef, mc_coef, max_norm, gradient_accumulation_steps, loading_epoch))
            print("model_loading : {}\nDo Training : {}\ntrain_iter_size : {}\n eval_iter_size : {}".format(
                b_model_load, b_meta_train, train_iter_size, eval_iter_size))
            LOG.info("Are you sure with this options? (If you don't need ipdb, please annotate it!)")
            if loading_epoch != 0:
                LOG.info("If you want to train from initial weights, please change it to 0")

            # Optimizer
            from pytorch_transformers import AdamW
            meta_optim = AdamW(model.parameters(), lr=meta_lr)
            meta_optim.zero_grad()

            # Create eval_iter_list (because when eval_task_iters finished, it makes errors)
            eval_iter_list = []
            while 1:
                try:
                    next_ = next(eval_task_iters)
                    eval_iter_list.append(next_)
                except:
                    break

            # Start outer loop (meta learner "epochs") #############################################
            if not train_task_iters:
                LOG.warning("Model does not need meta-training")
                model = torch.load('model.pt')
                LOG.info('model loading done!')
            else:
                total_iteration = 0
                if b_model_load:
                    model = model.cpu()
                    del model
                    torch.cuda.empty_cache()
                    intermediate_model_path = os.path.join(
                        train_config.modules_save_dir, "model_epoch{}.pt".format(loading_epoch))

                    LOG.info('model loading... {}'.format(intermediate_model_path))
                    task, _ = load(intermediate_model_path)
                    model = task.model
                    import ipdb; ipdb.set_trace()
                    # TODO
                else:
                    total_iteration = 0

                for epoch in range(loading_epoch+1, 100):

                    if b_meta_train:
                        logging.info("Training model on train tasks")
                        for bidx, (support_query, target, context) in zip(range(train_iter_size), train_task_iters): # N times

                            # TODO "MAML++" paper, what I need to implement next!
                            losses_q = [0 for _ in range(update_step)] # losses_q[i] is the loss on step i
                            model_params = list(model.parameters())
                            # TODO not sure to user model.parameters()
                            # TODO I might think we need to use "load_state_dict()", state_dict()
                            original_params = copy.deepcopy(list(model.parameters()))

                            task_num = len(support_query)
                            for task_i, ((support_set, query_set), _, (s_context, t_context)) in enumerate(zip(support_query, target, context)): # meta_batch_size times

                                # total_iteration update
                                total_iteration += 1

                                s_domain = s_context['domain_id'][0]
                                task_id = t_context['task_id'][0]
                                print("epoch {} b_idx {} task_i ({}/{}) s_domain {}".format(epoch, bidx, task_i+1, task_num, s_domain))

                                support_set = tuple([torch.squeeze(s, dim=1) for s in support_set])
                                query_set = tuple([torch.squeeze(q, dim=1) for q in query_set])

                                # To avoid gpu memory lack, I did a trick like,
                                # support_set (batch_size(128), support_unit) -> (?, small_batch_size, support_unit)
                                # query_set (batch_size(128), support_unit) -> (4, support_unit) # just cut and discard the remain!
                                # TODO spt_batchsz==1, because of gpu memory limit.
                                # TODO Why I make it 1 is, to maximize query_set_size(4)
                                # TODO We need to find optimal setting spt_batch 2 query_batch 2 or....
                                support_sets = self.divide_in_and_out_batchsz(support_set, batchsz=spt_in_batchsz, mode='support', mc_num=2, max_len=self.max_len)
                                #meta_train_query_set = self.meta_train_query_set_preprocess(query_set, batchsz=1)
                                meta_train_query_sets = self.divide_in_and_out_batchsz(query_set, batchsz=qry_in_batchsz, mode='meta-train-query', mc_num=1)

                                model.train()
                                # 1. run the i-th task and compute loss for k=1~K-1
                                for k in range(update_step): #TODO unitl now, I fixed update_step == 1, because of gradient degradation issue
                                                             # I set spt_batchsz=1, (128, 1)
                                                             # this means "128" weight update in inner learning,
                                                             # so I hope to fixed update_step == 1, to avoid gradient degradation issue
                                    # in-train (support)
                                    for si, support_set in enumerate(zip(*support_sets)): # TODO 128 steps (== 128 / spt_batchsz(=1))
                                        lm_loss, mc_loss, _, _, _ = model(text_embedder, *support_set)
                                        loss = (lm_loss * lm_coef + mc_loss * mc_coef) / gradient_accumulation_steps
                                        # 2. compute grad on theta_pi
                                        grad = torch.autograd.grad(loss, model.parameters(), allow_unused=True)
                                        # gradient clipping
                                        new_grad = self._clip_grad_norm(grad[1:], max_norm, norm_type)
                                        grad = tuple([grad[0]] + list(new_grad))
                                        # 3. theta_pi = theta_pi - train_lr * grad
                                        fast_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grad[1:], model_params[1:])))
                                        fast_weights = [model_params[0]] + fast_weights  # model_params[0] is dummy!!!  caution!
                                        for mp_idx, mp in enumerate(model_params):
                                            mp.data = fast_weights[mp_idx]

                                # in-test (query)
                                # Evaluate the model using the target set
                                model.eval()
                                lm_loss_task = 0
                                for qi, query_set in enumerate(zip(*meta_train_query_sets)):
                                    lm_loss, _, _, _, _ = model(text_embedder, *query_set, mode='meta-train-query')
                                    lm_loss_task += lm_loss
                                losses_q[k] += (lm_loss_task / qry_out_batchsz)

                                #model.eval()
                                #lm_loss, _, _, _, _ = model(text_embedder, *meta_train_query_set, mode='meta-train-query')
                                #loss_q = lm_loss / gradient_accumulation_steps
                                #losses_q[k] += loss_q
                                # Release unnecessary gpu allocation
                                torch.cuda.empty_cache()

                                # Restore original params for next tasks
                                for mp_idx, mp in enumerate(model_params):
                                    mp.data = original_params[mp_idx]

                            # end of all tasks
                            # Release unnecessary gpu allocation
                            torch.cuda.empty_cache()
                            # Total loss
                            loss_q = losses_q[-1] / task_num

                            # Report meta-train loss (from meta-query set)
                            metric_reporter.add_batch_stats(task_id, loss_q.item(), s_context['dlg_len'])
                            metric_reporter.report_metric(Stage.TRAIN, total_iteration, reset=False)

                            # print("Meta-train loss_q :", loss_q.item())
                            # backward -> (clip) step -> zero_grad
                            loss_q.backward()
                            # TODO not sure to clipping or not in meta-gradient case
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                            meta_optim.step()
                            meta_optim.zero_grad()

                                        new_grad = self._clip_grad_norm(grad[1:], max_norm, norm_type)
                                        grad = tuple([grad[0]] + list(new_grad))
                                        # 3. theta_pi = theta_pi - train_lr * grad
                                        fast_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grad[1:], model_params[1:])))
                                        fast_weights = [model_params[0]] + fast_weights  # model_params[0] is dummy!!! CAUTION
                                        # 4. Update fast_weights
                                        for mp_idx, mp in enumerate(model_params):
                                            mp.data = fast_weights[mp_idx]

                                # in-test (query)
                                print("Meta_test_query...")
                                model.eval()
                                for q_i, (query_ins, label) in enumerate(zip(meta_test_query_set, labels)):
                                    input_tokens, token_type_tokens = self.meta_test_query_PRINT_preprocess(query_ins, tokenizer, byte_decoder)
                                    print("[q_i]", q_i)
                                    print("[input]", ''.join(input_tokens))
                                    if model.representation.gptmode == 'gpt2':
                                        sentence, _, _, _, _ = model(text_embedder, *query_ins, mode='infer')
                                        print("\n[prdct]", sentence)
                                        print("[label]", label)
                                        print("-" * 200)
                                    else:
                                        lm_loss, mc_loss = model(text_embedder, *query_ins, mode='infer')

                                # restore original params for next tasks
                                for mp_idx, mp in enumerate(model_params):
                                    mp.data = original_params[mp_idx]

                    #metric_reporter.report_metric(Stage.EVAL, epoch, reset=False)

            best_model_path = os.path.join(
                train_config.modules_save_dir, "model.pt"
            )
            torch.save(model.state_dict(), best_model_path)
            train_config.save_snapshot_path = best_model_path
            print("return!")
            return model, None

# Debugging...
# if gpt_input_idx == 0:
#    LOG.info("Debugging gpt input... ")
#    batch_id = 0
#    neg_toks, pos_toks = gptinput2tokens(gpt_input, batch_id=batch_id)
#    print("Neg input example / batch_id {}".format(batch_id))
#    print_toks(*neg_toks)
#    print("Pos input example / batch_id {}".format(batch_id))
#    print_toks(*pos_toks)

