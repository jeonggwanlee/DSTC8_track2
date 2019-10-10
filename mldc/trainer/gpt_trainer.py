import os
import torch
import logging
import copy
import random
from typing import Any, Optional, Tuple
import numpy as np
from itertools import chain
import ipdb

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

from mldc.utils.common import transform_byte2normal, meta_test_query_PRINT_preprocess, gptinput2tokens, print_toks
from mldc.utils.ignore_domain import create_instance_dict, get_train_batches, create_eval_input_set, valid_process
from mldc.utils.meta_learning import divide_in_and_out_batchsz

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
            len_except_pad = not_pad_part.sum().item()  # N
            reverse_iter = len_except_pad - 1
            if text_embedder.sys_idx == qtti[1][len_except_pad - 1].item():
                while qtti[1][reverse_iter].item() == text_embedder.sys_idx: reverse_iter -= 1
            else:
                while qtti[1][reverse_iter].item() == text_embedder.usr_idx: reverse_iter -= 1
            history = qii[1][:reverse_iter + 1]
            history = torch.unsqueeze(history, dim=0)
            label = qii[1][reverse_iter + 1:len_except_pad]
            label = torch.unsqueeze(label, dim=0)
            qtti = qtti[1][:reverse_iter + 1]
            qtti = torch.unsqueeze(qtti, dim=0)
            assert (history.shape == qtti.shape)
            assert (history.shape[1] == qtti.shape[1] == (self.max_len - label.shape[1]))
            query = (history, None, label, None, qtti)
            query_set_.append(query)
            label = label.numpy().reshape(-1).tolist()
            label_tokens = tokenizer.convert_ids_to_tokens(label, skip_special_tokens=False)
            label_transform = [transform_byte2normal(tokenizer, byte_decoder, token) for token in label_tokens]
            labels.append(''.join(label_transform))

        return query_set_, labels

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

        spt_in_batchsz = 1
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
                #support_sets = self.support_sets_transformation(support_set, spt_batchsz=4, manual_size=-1)
                support_sets = divide_in_and_out_batchsz(support_set, batchsz=spt_in_batchsz, mode='support', mc_num=2, max_len=self.max_len)

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
                        input_tokens, token_type_tokens = meta_test_query_PRINT_preprocess(query_ins, tokenizer, byte_decoder)
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
        byte_decoder = tokenizer.byte_decoder
        self.max_len = 300
        pad_token_idx = text_embedder.pad_idx


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
            if not os.path.exists(filename):
                LOG.info("\n\tMaking dataset.....")
                dataset = train_task_iters
                examples = dataset.examples

                domains = task.data_handler.train_domains
                eval_domains = ['dialogues/ALARM_SET.txt', 'dialogues/AGREEMENT_BOT.txt', 'dialogues/EDIT_PLAYLIST.txt']
                train_domains = list(set(domains) - set(eval_domains))
                fixed_n_turns = True # if true, consider all dlgs!
                n_turns = 20         # if fixed_n_turns True, meaningless, otherwise, max_length of dlgs
                all_responses = True

                LOG.info("\n\ttrain_domains {}\n\n\teval_domains {}\n\n\tfixed_n_turns {}\n\n\tall_responses {}".format(
                    train_domains, eval_domains, fixed_n_turns, all_responses))

                train_instance_dict = create_instance_dict(text_embedder, examples, train_domains, fixed_n_turns, n_turns, all_responses, self.max_len, pad_token_idx)
                eval_instance_dict = create_instance_dict(text_embedder, examples, eval_domains, fixed_n_turns, n_turns, all_responses, self.max_len, pad_token_idx)
                import pickle
                with open(filename, 'wb') as f:
                    pickle.dump([train_instance_dict, eval_instance_dict], f)
            else:
                import pickle
                with open(filename, 'rb') as f:
                    print("loading {}".format(filename))
                    train_instance_dict, eval_instance_dict = pickle.load(f)
            # Data preprocessing done!
            # only you have to remember is, "train_instance_dict", and "eval_instance_dict"

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

            # Get eval data batches [Select only N cases / make "eval_set"]
            eval_input_ids = np.array(eval_instance_dict['input_ids'])
            eval_mc_token_ids = np.array(eval_instance_dict['mc_token_ids'])
            eval_lm_labels = np.array(eval_instance_dict['lm_labels'])
            eval_token_type_ids = np.array(eval_instance_dict['token_type_ids'])
            total_eval_num = eval_input_ids.shape[0]
            LOG.info("\n\tconsidering batch size, train_num {}\n\ttotal_eval_num {}".format(modif_train_num, total_eval_num))

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
            loading_gpt_idx = 0 #loading_gpt_idx = 12500
            loading_total_iter = 0 #loading_total_iter = 12500
            # training option(do train or not)
            b_train = True
            # eval option
            eval_interval = 2500
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
                LOG.info("you need to manage total_iteration !! (tensorboard)")
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
                    gpt_input_zip = get_train_batches(modif_train_num, this_batch_size, train_input_ids, train_mc_token_ids, train_lm_labels, train_mc_labels, train_token_type_ids)

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
                            eval_input_zip = create_eval_input_set(text_embedder, total_eval_num, selected_eval_num,
                                                                   eval_input_ids, eval_mc_token_ids, eval_lm_labels, eval_token_type_ids)
                            valid_loss = valid_process(model, eval_input_zip, text_embedder, tokenizer, byte_decoder, self.max_len, metric_reporter)
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
                eval_input_zip = create_eval_input_set(text_embedder, total_eval_num, selected_eval_num,
                                                       eval_input_ids, eval_mc_token_ids, eval_lm_labels, eval_token_type_ids)
                valid_loss = valid_process(model, eval_input_zip, text_embedder, tokenizer, byte_decoder, self.max_len, metric_reporter)
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
            else:
                import ipdb; ipdb.set_trace()
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
            assert(qry_in_batchsz == 1) # TODO because of code below... ## IMPORTANT  would be DEPRECATED!!
            query_real_batch_size = 3
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
            b_meta_train = True
            # How many iteration you want in one epoch.
            ## TODO Warning!!!!!
            train_iter_size = 100 # cost = train_iter_size * task_num * (support_set + query_set)
            eval_iter_size = 1    # cost = eval_iter_size * task_num * (support_set + query_set)

            LOG.info("meta-learning method")
            print("""\n\tmeta_lr(outer_lr):{}\n\tupdate_lr(inner_lr) : {}\n\tupdate_step : {}
                  \tlm_coef : {}\n\tmc_coef : {}\n\tmax_norm_for_grad_clip : {}\n\tgrad_accums : {}\n\tloading_epoch : {}""".format(
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

                                support_set = tuple([torch.squeeze(s, dim=1).to('cuda') for s in support_set])
                                query_set = tuple([torch.squeeze(q, dim=1).to('cuda') for q in query_set])

                                # To avoid gpu memory lack, I did a trick like,
                                # support_set (batch_size(128), support_unit) -> (?, small_batch_size, support_unit)
                                # query_set (batch_size(128), support_unit) -> (4, support_unit) # just cut and discard the remain!
                                # TODO spt_batchsz==1, because of gpu memory limit.
                                # TODO Why I make it 1 is, to maximize query_set_size(4)
                                # TODO We need to find optimal setting spt_batch 2 query_batch 2 or....
                                support_sets = divide_in_and_out_batchsz(support_set, batchsz=spt_in_batchsz, mode='support', mc_num=2, max_len=self.max_len)
                                meta_train_query_sets = divide_in_and_out_batchsz(query_set, batchsz=1, mode='meta-train-query', mc_num=1)

                                model.train()
                                # 1. run the i-th task and compute loss for k=1~K-1
                                for k in range(update_step): #TODO unitl now, I fixed update_step == 1, because of gradient degradation issue
                                    # in-train (support)
                                    ss_len = len(list(zip(*support_sets)))
                                    for si, support_set in enumerate(zip(*support_sets)): # 128 * augmented num
                                        if si % 30 == 0 and si != 0:
                                            print("support sets : ", si, ss_len)
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
                                model.eval()
                                lm_loss_task = 0
                                # To avoid memory lack, cut as query_real_batch_size
                                query_set_cut = list(zip(*meta_train_query_sets))
                                random.shuffle(query_set_cut)
                                query_set_cut = query_set_cut[:query_real_batch_size]

                                for qi, query_set in enumerate(query_set_cut):
                                    lm_loss, _, _, _, _ = model(text_embedder, *query_set, mode='meta-train-query')
                                    lm_loss_task += lm_loss
                                losses_q[k] += (lm_loss_task / len(query_set_cut))

                                # Release unnecessary gpu allocation
                                torch.cuda.empty_cache()

                                # Restore original params for next tasks
                                for mp_idx, mp in enumerate(model_params):
                                    mp.data = original_params[mp_idx]

                            # End of all tasks
                            # Release unnecessary gpu allocation
                            torch.cuda.empty_cache()
                            # Total loss
                            loss_q = losses_q[-1] / task_num

                            # Report meta-train loss (from meta-query set)
                            metric_reporter.add_batch_stats(task_id, loss_q.item(), s_context['dlg_len'])
                            metric_reporter.report_metric(Stage.TRAIN, total_iteration, reset=False)

                            # Backward : backward -> (clip) step -> zero_grad
                            loss_q.backward()
                            # TODO not sure to clipping or not in meta-gradient case
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                            meta_optim.step()
                            meta_optim.zero_grad()
                            torch.cuda.empty_cache()

                        # End of one epoch.
                        metric_reporter.report_metric(Stage.TRAIN, total_iteration, reset=False)
                        intermediate_model_path = os.path.join(train_config.modules_save_dir, "model_epoch{}.pt".format(epoch))
                        torch.save(model.state_dict(), intermediate_model_path)
                        LOG.info("Saving... {}".format(intermediate_model_path))
                        task.model = model
                        self.save_export_filename(train_config, task, intermediate_model_path)
                    # End of one epoch Meta-train

                    # Meta-test
                    logging.info("Evaluating model on eval tasks")
                    with torch.enable_grad():
                        for bidx, (support_query, target, context) in zip(range(eval_iter_size), eval_iter_list):

                            # Copy model parameter for next task
                            model_params = list(model.parameters())
                            original_params = copy.deepcopy(list(model.parameters()))

                            valid_losses = 0
                            # task num == 2
                            task_num = len(support_query)
                            for task_i, ((support_set, query_set), _, (s_context, t_context)) in enumerate(zip(support_query, target, context)):
                                s_domain = s_context['domain_id'][0]
                                task_id = s_context['task_id'][0]
                                print("b_idx", bidx, "task_i", task_i, "task_num", task_num, "s_domain :", s_domain)
                                support_set = tuple([torch.squeeze(s, dim=1).to('cuda') for s in support_set])
                                query_set = tuple([torch.squeeze(q, dim=1).to('cuda') for q in query_set])
                                # To avoid gpu memory lack,
                                support_sets = divide_in_and_out_batchsz(support_set, batchsz=spt_in_batchsz, mode='support', mc_num=2, max_len=self.max_len)
                                query_set_v1 = divide_in_and_out_batchsz(query_set, batchsz=1, mode='meta-train-query', mc_num=1)

                                ## query input setting! TODO make module!
                                # (128, 1, 1, 300)
                                # [None] * 128
                                # (128 ,1, 1, 300)
                                # [None [128]]
                                # (128, 1, 1, 300)
                                qii_list, qll_list, qtti_list = [], [], []
                                for mqi, (qii, _, qll, _, qtti) in enumerate(zip(*query_set_v1)):

                                    nopad_qii = qii[qii != pad_token_idx].unsqueeze(0).unsqueeze(0) # (1, 1, before_response)
                                    nopad_qii_length = nopad_qii.shape[2]
                                    nopad_qtti = qtti[:, :, :nopad_qii_length]
                                    last_turn_token_type = nopad_qtti[0, 0, -1].item()
                                    reverse_iter = nopad_qii_length - 1
                                    if last_turn_token_type == text_embedder.sys_idx: # last turn is sys turn
                                        while nopad_qtti[0, 0, reverse_iter].item() == text_embedder.sys_idx:
                                            reverse_iter -= 1
                                    elif last_turn_token_type == text_embedder.usr_idx:
                                        while nopad_qtti[0, 0, reverse_iter].item() == text_embedder.usr_idx:
                                            reverse_iter -= 1
                                    # now reverse_iter is on the other turn
                                    history_border = reverse_iter + 1
                                    history_qii = qii[:, :, :history_border]
                                    history_qtti = qtti[:, :, :history_border]
                                    response_with_m1 = qll[:, :, history_border:]
                                    qii_list.append(history_qii)
                                    qll_list.append(response_with_m1)
                                    qtti_list.append(history_qtti)

                                query_sets_v2 = (qii_list, [None] * 128, qll_list, [None] * 128, qtti_list)
                                # return query_sets_v2

                                model.train()
                                # 1. run the i-th task and compute loss for k=1~K-1
                                for k in range(update_step):

                                    # in-train(support)
                                    ss_len = len(list(zip(*support_sets)))
                                    for si, support_set in enumerate(zip(*support_sets)):
                                        if si % 30 == 0 and si != 0:
                                            print("support_set: ", si, ss_len)
                                        lm_loss, mc_loss, _, _, _ = model(text_embedder, *support_set, mode='teacher')
                                        loss = (lm_loss * lm_coef + mc_loss * mc_coef) / gradient_accumulation_steps
                                        # 2. compute grad on theta_pi
                                        grad = torch.autograd.grad(loss, model.parameters(), allow_unused=True)
                                        # Gradient clipping
                                        new_grad = self._clip_grad_norm(grad[1:], max_norm, norm_type)
                                        grad = tuple([grad[0]] + list(new_grad))
                                        # 3. theta_pi = theta_pi - train_lr * grad
                                        fast_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grad[1:], model_params[1:])))
                                        fast_weights = [model_params[0]] + fast_weights  # model_params[0] is dummy!!! CAUTION
                                        # 4. Update fast_weights
                                        for mp_idx, mp in enumerate(model_params):
                                            mp.data = fast_weights[mp_idx]

                                # in-test (query)
                                LOG.info("Validation Meta_test_query...")
                                model.eval()
                                total_valid_lm_loss = 0

                                # query sampling! gpt memory is not enough to cover original batch size (128)
                                query_set_cut = list(zip(*query_sets_v2))
                                random.shuffle(query_set_cut)
                                query_set_cut = query_set_cut[:query_real_batch_size]

                                for q_i, query_ins in enumerate(query_set_cut):
                                    if q_i % 1 == 0:
                                        print("query_sets", q_i, len(query_set_cut))
                                    qii, _, qll, _, qtti = query_ins
                                    input_tokens, token_type_tokens = meta_test_query_PRINT_preprocess(query_ins, tokenizer, byte_decoder)
                                    valid_lm_loss, sentence, sentence_tokens, _, _ = model(text_embedder, qii[0], None, qtti[0], None, qtti[0], mode='infer')
                                    if q_i == 0:
                                        print("[q_i]", q_i)
                                        print("[input]", ''.join(input_tokens))
                                        print("\n[prdct]", sentence)
                                        response_with_m1_vector = qll[0, 0, :]
                                        response_no_m1 = response_with_m1_vector[response_with_m1_vector != -1]
                                        response_no_m1_list = response_no_m1.cpu().numpy().tolist()
                                        response_tokens = tokenizer.convert_ids_to_tokens(response_no_m1_list, skip_special_tokens=False)
                                        response = [transform_byte2normal(tokenizer, byte_decoder, token) for token in response_tokens]
                                        print("[label]", "".join(response))
                                        print("-" * 200)
                                    total_valid_lm_loss += valid_lm_loss.item()

                                valid_loss_per_task = total_valid_lm_loss / query_real_batch_size
                                valid_losses += valid_loss_per_task

                                # restore original params for next tasks
                                for mp_idx, mp in enumerate(model_params):
                                    mp.data = original_params[mp_idx]

                                print("really", task_i)
                            # task for loop and
                            valid_loss = valid_losses / task_num # (task_num = 1)
                            metric_reporter.add_batch_stats(task_id, valid_loss, s_context['dlg_len'])
                            metric_reporter.report_metric(Stage.EVAL, total_iteration, reset=False)

                    metric_reporter.report_metric(Stage.EVAL, total_iteration, reset=False)
                    LOG.info('Meta test done {}'.format(valid_loss))
                    # Meta test done

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

