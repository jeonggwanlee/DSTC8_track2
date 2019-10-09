


def transform_byte2normal(tokenizer, byte_decoder, token):
    if token is None:
        return None
    temp = []
    for tok in token:
        temp.append(byte_decoder[tok])
    temp2 = bytearray(temp).decode('utf-8', errors=tokenizer.errors)
    return temp2


def meta_test_query_PRINT_preprocess(query_ins, tokenizer, byte_decoder):

    input_ids = query_ins[0][0].cpu().numpy().tolist()
    token_type_ids = query_ins[4][0].cpu().numpy().tolist()
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
    token_type_tokens = tokenizer.convert_ids_to_tokens(token_type_ids, skip_special_tokens=False)
    input_tokens = [transform_byte2normal(tokenizer, byte_decoder, token) for token in input_tokens]
    token_type_tokens = [transform_byte2normal(tokenizer, byte_decoder, token) for token in token_type_tokens]

    return input_tokens, token_type_tokens


def gptinput2tokens(tokenizer, byte_decoder, support_set, batch_id=0):
    """ Until now, this function only support "support set", not query set """
    """ Debugging purpose """

    def gputensor2tokens(tensor_gpu):
        ids = tensor_gpu.cpu().numpy().tolist()
        byte_tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
        tokens = [transform_byte2normal(tokenizer, byte_decoder, token) for token in byte_tokens]
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
