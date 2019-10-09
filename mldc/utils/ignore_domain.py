from itertools import chain
import numpy as np
import logging
import random
import torch

from mldc.utils.common import meta_test_query_PRINT_preprocess, transform_byte2normal

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu

LOG = logging.getLogger("mldc.utils")



# This is create gpt_input_unit function. skip it.
def create_gpt_input_unit(history, pos_resp, neg_resp,
                          history_turn_types, presp_turn_type, nresp_turn_type, max_len, pad_token_idx):
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

    if len(pos_input_ids) > max_len:
        excess = len(pos_input_ids) - max_len
        pos_input_ids = pos_input_ids[-max_len:]
        pos_input_ids2 = pos_input_ids[excess:]
        if pos_input_ids == pos_input_ids2:
            print("real True")
        # print("origin", pos_mc_token_ids)
        pos_mc_token_ids = pos_mc_token_ids - excess
        # print("now", pos_mc_token_ids)
        pos_lm_labels = pos_lm_labels[-max_len:]
        pos_token_types = pos_token_types[-max_len:]
    else:
        num_pad = max_len - len(pos_input_ids)
        pos_input_ids = pos_input_ids + [pad_token_idx] * num_pad
        pos_lm_labels = pos_lm_labels + [-1] * num_pad
        pos_token_types = pos_token_types + [pad_token_idx] * num_pad

    assert (len(pos_input_ids) == max_len)
    assert (pos_mc_token_ids < max_len)
    assert (len(pos_lm_labels) == max_len)
    assert (len(pos_token_types) == max_len)

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

    if len(neg_input_ids) > max_len:
        excess = len(neg_input_ids) - max_len
        neg_input_ids = neg_input_ids[-max_len:]
        neg_mc_token_ids = neg_mc_token_ids - excess
        neg_lm_labels = neg_lm_labels[-max_len:]
        neg_token_types = neg_token_types[-max_len:]
    else:
        num_pad = max_len - len(neg_input_ids)
        neg_input_ids = neg_input_ids + [pad_token_idx] * num_pad
        neg_lm_labels = neg_lm_labels + [-1] * num_pad
        neg_token_types = neg_token_types + [pad_token_idx] * num_pad

    assert (len(neg_input_ids) == max_len)
    assert (neg_mc_token_ids < max_len)
    assert (len(neg_token_types) == max_len)
    assert (len(neg_lm_labels) == max_len)

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
def create_instance_dict(text_embedder, examples, domains, fixed_n_turns, n_turns, all_responses):
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

        if not example.domain_id in new_domains:  # This means We will pick example only related to domain
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
        history_turns = [turn.tolist()[text_embedder.pieces_slice] for turn in pos_turns[startpoint:endpoint]]
        raw_history_turns = [turn[text_embedder.pieces_slice] for turn in raw_pos_turns[startpoint:endpoint]]
        history_mat.append(history_turns)
        raw_history_mat.append(raw_history_turns)
        input_token_type_id = []
        for turn in history_turns:
            speaker = text_embedder.decode_ids_as_text([turn[0]])
            if speaker == '<system>':
                input_token_type_id.append([text_embedder.sys_idx] * len(turn))
            elif speaker == '<user>':
                input_token_type_id.append([text_embedder.usr_idx] * len(turn))
        history_turn_type_mat.append(input_token_type_id)

        # TARGET
        if all_responses:  # answers to all input turns
            ep_slice = slice(startpoint + 1, endpoint + 1, 2)
        else:  # answer to last input turn
            ep_slice = slice(endpoint, endpoint + 1)

        pos_resps = [turn.tolist()[text_embedder.pieces_slice] for turn in pos_turns[ep_slice]]
        pos_resps_raw = [turn[text_embedder.pieces_slice] for turn in raw_pos_turns[ep_slice]]
        pos_resp_mat.append(pos_resps)
        raw_pos_resp_mat.append(pos_resps_raw)
        target_token_type_id = []
        for resp in pos_resps:
            speaker = text_embedder.decode_ids_as_text([resp[0]])
            if speaker == '<system>':
                target_token_type_id.append([text_embedder.sys_idx] * len(resp))
            elif speaker == '<user>':
                target_token_type_id.append([text_embedder.usr_idx] * len(resp))
        pos_resp_turn_type_mat.append(target_token_type_id)

        neg_resps = [turn.tolist()[text_embedder.pieces_slice] for turn in neg_turns[ep_slice]]
        neg_resps_raw = [turn[text_embedder.pieces_slice] for turn in raw_neg_turns[ep_slice]]
        raw_neg_resp_mat.append(neg_resps_raw)
        neg_resp_mat.append(neg_resps)
        neg_tar_token_type_id = []
        for resp in neg_resps:
            speaker = text_embedder.decode_ids_as_text([resp[0]])
            if speaker == '<system>':
                neg_tar_token_type_id.append([text_embedder.sys_idx] * len(resp))
            elif speaker == '<user>':
                neg_tar_token_type_id.append([text_embedder.usr_idx] * len(resp))
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


def get_train_batches(modif_train_num, this_batch_size, train_input_ids, train_mc_token_ids, train_lm_labels, train_mc_labels, train_token_type_ids):
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
    assert (input_ids.shape[0] == mc_token_ids.shape[0] == lm_labels.shape[0] == mc_labels.shape[0] == token_type_ids.shape[0])
    gpt_input_zip = zip(input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids)

    return gpt_input_zip


def create_eval_input_set(text_embedder, total_eval_num, selected_eval_num,
                          eval_input_ids, eval_mc_token_ids, eval_lm_labels, eval_token_type_ids):
    eval_perm = np.arange(total_eval_num)
    random.shuffle(eval_perm)

    # Select only N cases (eval data)
    eval_perm = eval_perm[:selected_eval_num]

    this_eval_input_ids = np.expand_dims(eval_input_ids[eval_perm][:, 1, :], axis=1)  # we need only positive one
    this_eval_mc_token_ids = np.expand_dims(eval_mc_token_ids[eval_perm][:, 1], axis=1)
    this_eval_lm_labels = np.expand_dims(eval_lm_labels[eval_perm][:, 1, :], axis=1)
    this_eval_token_type_ids = np.expand_dims(eval_token_type_ids[eval_perm][:, 1, :], axis=1)

    # Make eval_collect
    eval_set = zip(this_eval_input_ids, this_eval_mc_token_ids, this_eval_lm_labels, this_eval_token_type_ids)
    eval_collect = []
    for eval_ii, eval_mc, eval_ll, eval_tti in eval_set:
        necessary_idx = np.where(eval_tti != text_embedder.pad_idx)
        eval_ii = eval_ii.reshape(-1)  # (300)
        eval_ll_for_cut = eval_ll[necessary_idx]  # (21, )
        eval_ll = eval_ll.reshape(-1)
        eval_tti = eval_tti[necessary_idx]  # (21, )
        # Get border of history and true response
        ll_idx = np.where(eval_ll_for_cut != -1)
        border = min(ll_idx[0])
        # response_final_border = max(ll_idx[0])
        eval_history = eval_ii[:border]
        eval_history_token_type = eval_tti[:border]
        eval_resp_near_m1 = eval_ll[border:]
        assert (eval_history.shape[0] == eval_history_token_type.shape[0] == eval_ii.shape[0] - eval_resp_near_m1.shape[0])
        eval_collect.append((eval_history, eval_mc, eval_history_token_type, eval_resp_near_m1))

    return eval_collect


def valid_process(model, eval_input_zip, text_embedder, tokenizer, byte_decoder, max_len, metric_reporter):
    reference_matrix = []
    hypotheses = []
    total_valid_losses = 0
    try:
        model.eval()
        zip_len = len(list(eval_input_zip))
        for eval_i, (eval_hist, eval_mc, eval_hist_tt, eval_response_rear_m1) in enumerate(eval_input_zip):
            if eval_i % 50 == 0:
                print("eval_iter : {}/{}".format(eval_i, zip_len))
            eval_hist = torch.from_numpy(eval_hist).type(torch.LongTensor).to('cuda').unsqueeze(0);  # (1, 24)
            eval_hist_tt = torch.from_numpy(eval_hist_tt).type(torch.LongTensor).to('cuda').unsqueeze(0);  # (1, 24)
            eval_response_rear_m1 = torch.from_numpy(eval_response_rear_m1).type(torch.LongTensor).to('cuda').unsqueeze(0);  # (1, 276)
            # eval_mc = torch.from_numpy(eval_mc).type(torch.LongTensor).to('cuda'); #
            gpt_input = (eval_hist, None, eval_response_rear_m1, None, eval_hist_tt)

            input_tokens, token_type_tokens = meta_test_query_PRINT_preprocess(gpt_input, tokenizer, byte_decoder)
            eval_response_no_minus1 = eval_response_rear_m1[eval_response_rear_m1 != -1]
            eval_resp_no_minus1_tokens = tokenizer.convert_ids_to_tokens(eval_response_no_minus1.tolist(), skip_special_tokens=False)
            eval_resp_no_minus1_tokens = [transform_byte2normal(tokenizer, byte_decoder, token) for token in eval_resp_no_minus1_tokens]
            # print_toks(input_tokens, token_type_tokens)

            # Input length checking
            assert (eval_hist.shape[1] == eval_hist_tt.shape[1] == (max_len - eval_response_rear_m1.shape[1]))

            try:
                valid_lm_loss, sentence, sentence_tokens, _, _ = model(text_embedder, *gpt_input, mode='infer')
            except:
                print("got u")
                import ipdb;
                ipdb.set_trace()
                valid_lm_loss, sentence, sentence_tokens, _, _ = model(text_embedder, *gpt_input, mode='infer')
                import ipdb;
                ipdb.set_trace()
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
        import ipdb;
        ipdb.set_trace()

    score1 = corpus_bleu(reference_matrix, hypotheses, smoothing_function=SmoothingFunction().method1)
    score2 = corpus_bleu(reference_matrix, hypotheses, smoothing_function=SmoothingFunction().method2)
    score3 = corpus_bleu(reference_matrix, hypotheses, smoothing_function=SmoothingFunction().method3)
    score4 = corpus_bleu(reference_matrix, hypotheses, smoothing_function=SmoothingFunction().method4)
    print("[corpus level bleu score] : {:.3f} {:.3f} {:.3f} {:.3f}".format(score1, score2, score3, score4))
    print("eval_i(for debugging)", eval_i)
    valid_loss = total_valid_losses / (eval_i + 1)  # real number
    metric_reporter.add_batch_stats("-ignore-domain-", valid_loss, 1)

    return valid_loss
# function definition done!

