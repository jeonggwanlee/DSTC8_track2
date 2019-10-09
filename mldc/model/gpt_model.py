import torch
from pytext.config import ConfigBase
from pytext.models.embeddings import EmbeddingBase, EmbeddingList
from pytext.models.model import Model
from pytext.models.representations.representation_base import RepresentationBase
from pytext.models.decoders import DecoderBase
from pytext.models.output_layers import OutputLayerBase
from typing import Optional, List, Dict, Tuple, Any
import copy
import numpy as np

from pytorch_transformers import GPT2DoubleHeadsModel
from pytorch_pretrained_bert import OpenAIGPTDoubleHeadsModel
from mldc.preprocessing.input_embedding import SPECIAL_TOKENS
from mldc.preprocessing.input_embedding import GPT2Embed

class RetrievalOutputLayer(OutputLayerBase):
    class Config(ConfigBase):
        pass

    @classmethod
    def from_config(cls, config: Config, meta: Any):
        return cls(config)


class RetrievalRepresentation(RepresentationBase):
    class Config(RepresentationBase.Config):
        pass

    def __init__(self, config: Config, embed_dim: int) -> None:
        super().__init__(config)
        self.representation_dim = embed_dim

    def forward(
            self,
            seq: torch.Tensor,
            seq_lengths: torch.Tensor,
            word_lengths: torch.Tensor,
            *args,
    ) -> torch.Tensor:

        for dlg_i, dlg_n_turns in enumerate(seq_lengths):
            for turn_i, n_words in zip(range(dlg_n_turns), word_lengths[dlg_i]):
                seq[dlg_i, turn_i, 0] = seq[dlg_i, turn_i, :n_words].sum(dim=0)
        seq_embed = seq[:, :, 0]
        seq_embed = torch.cumsum(seq_embed, dim=1)

        return seq_embed


class RetrievalDecoder(DecoderBase):
    class Config(ConfigBase):
        pass

    @classmethod
    def from_config(cls, config: Config, in_dim: int, out_dim: int):
        return cls(config)

    def forward(self, dlg_states: Tuple[torch.Tensor, ...],
                encoder_inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass


class GPT(RepresentationBase):
    class Config(RepresentationBase.Config):
        num_special_tokens = len(SPECIAL_TOKENS)

    def __init__(self, config: Config, embed_dim: int, *args, **kwargs) -> None:
        super().__init__(config)
        self.representation_dim = embed_dim
        self.gptmode = 'gpt2'
        #self.gptmode = 'openai-gpt'
        if self.gptmode == 'gpt2':
            self.model = GPT2DoubleHeadsModel.from_pretrained('gpt2')
            self.model.resize_token_embeddings(self.model.config.vocab_size + config.num_special_tokens)
        else:
            self.model = OpenAIGPTDoubleHeadsModel.from_pretrained('openai-gpt')
            self.model.set_num_special_tokens(len(SPECIAL_TOKENS))

        self.temperature = 0.9
        self.top_k = 0
        self.top_p = 0.7
        self.min_length = 1
        self.max_length = 300
        self.no_sample = True

    def build_input(self, input_ids, token_type_ids, sys_utt):
        if sys_utt == []:
            return input_ids, token_type_ids

        device = input_ids.device

        input_ids_list = list(input_ids.cpu().numpy().reshape(-1))
        last_ids = sys_utt[-1]
        input_ids_list.append(last_ids)
        #print("build_input", self.text(input_ids_list))
        input_ids = torch.tensor(input_ids_list, device=device).unsqueeze(0)
        #mc_token_ids[0][0] += 1

        if self.bsys_last_turn:
            token_type_ids = torch.cat([token_type_ids, torch.tensor([self.text_embedder.usr_idx], device=device).unsqueeze(0)], dim=1)
        else:
            token_type_ids = torch.cat([token_type_ids, torch.tensor([self.text_embedder.sys_idx], device=device).unsqueeze(0)], dim=1)

        return input_ids, token_type_ids

    def print_toks(self, input_tokens, token_type_labels, lm_labels=None, mc_token_ids=None, max_print_length=120, history_len=-1):
        """print tokens with pretty way"""
        num_toks_per_line = 20  # number of tokens per one line
        one_word_size = 8
        total_length = len(input_tokens)
        b_end = False

        frommin2max = list(range(0, max_print_length + 1, num_toks_per_line))  # +1 is just trick
        min_maxs = []
        for _ in range(max_print_length // num_toks_per_line):
            min_maxs.append((frommin2max[_], frommin2max[_ + 1]))

        for mini, maxi in min_maxs:
            for num_idx in range(mini, maxi):  # number
                if num_idx >= total_length:
                    b_end = True
                    break
                if num_idx != history_len:
                    print('{:8d}'.format(num_idx), end='')
                else:
                    print(' **{:3d}**'.format(num_idx), end='')
            print()

            for idx in range(mini, maxi):
                if idx >= total_length:  # cut, if over then total length
                    break
                nii = input_tokens[idx]
                if idx == max_print_length:
                    break
                print('{}'.format(nii[:one_word_size].rjust(one_word_size)), end='')
            print()

            for idx in range(mini, maxi):
                if idx >= total_length:  # cut, if over then total length
                    break
                ntl = token_type_labels[idx]
                if idx == max_print_length:
                    break
                print('{}'.format(ntl[:one_word_size].rjust(one_word_size)), end='')
            print()

            if lm_labels:
                for idx in range(mini, maxi):
                    if idx >= total_length:  # cut, if over then total length
                        break
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
                    if idx >= total_length:  # cut, if over then total length
                        break
                    if mc_token_ids == idx:
                        print('{}'.format('<mctok>'.rjust(one_word_size)), end='')
                    else:
                        print('{}'.format('-nomc-'.rjust(one_word_size)), end='')
                print()

            print('-' * (one_word_size * num_toks_per_line))
            if b_end:
                break

    def transform_byte2normal(self, tokenizer, byte_decoder, token):
        if token is None:
            return None
        temp = []
        for tok in token:
            temp.append(byte_decoder[tok])
        temp2 = bytearray(temp).decode('utf-8', errors=tokenizer.errors)
        return temp2

    def forward(self,
                text_embedder,
                input_ids: torch.Tensor,
                mc_token_ids: Optional[torch.Tensor],
                lm_labels: Optional[torch.Tensor],
                mc_labels: Optional[torch.Tensor],
                token_type_ids: Optional[torch.Tensor],
                mode='teacher'
                ) -> List[torch.Tensor]:

        if text_embedder is not None:
            self.text_embedder = text_embedder
            self.text = self.text_embedder.tokenizer.decode
            self.eos = self.text_embedder.eos_idx
            self.usr = self.text_embedder.usr_idx
            self.sys = self.text_embedder.sys_idx
            self.pad_idx = self.text_embedder.pad_idx

        if mode == 'teacher':
            lm_loss, mc_loss, lm_logits, mc_logits, pres = self.model(input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids)
            # lm_loss () mc_loss () lm_logits (4, 2, 300, 50270) mc_logits (4, 2) pres.__len__() 12
            return lm_loss, mc_loss, lm_logits, mc_logits, pres

        elif mode == 'meta-train-query':
            lm_loss, lm_logits, mc_logits, pres = self.model(input_ids, lm_labels=lm_labels, token_type_ids=token_type_ids)
            return lm_loss, None, None, None, None

        # Inference
        elif mode == 'infer':

            # check last_turn is finished at system turn
            # last turn system -> next token type ids ==> user
            # last turn user -> next token type ids ==> system
            self.bsys_last_turn = self.text_embedder.sys_idx == token_type_ids[-1][-1].item()
            b_finish_eos = False
            response_rear_m1 = lm_labels

            # inference start!
            infer_iter = 0
            sys_utt = []
            history_len = input_ids[0].shape[0]
            # origin ii, tti
            origin_input_ids = copy.deepcopy(input_ids)
            origin_token_type_ids = copy.deepcopy(token_type_ids)

            # Inference while loop
            while infer_iter <= self.max_length-1:
                input_ids, token_type_ids = self.build_input(input_ids, token_type_ids, sys_utt)

                lm_logits, mc_logits, _ = self.model(input_ids, token_type_ids=token_type_ids)

                logits = lm_logits # (1, 46, 50270)
                logits = logits[0, -1, :] / self.temperature # (50270)
                logits = self.top_filtering(logits)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                prev = torch.topk(probs, 1)[1] if self.no_sample else torch.multinomial(probs, 1)

                if infer_iter < self.min_length and prev.item() == self.eos:
                    while prev.item() == self.eos:
                        prev = torch.multinomial(probs, num_samples=1)

                if prev.item() == self.eos:
                    sys_utt.append(prev.item())
                    b_finish_eos = True
                    break

                infer_iter += 1
                sys_utt.append(prev.item())
            # End of inference

            b_full = False
            if b_finish_eos:
                # add eos
                input_ids, token_type_ids = self.build_input(input_ids, token_type_ids, sys_utt)
            else:
                b_full = True

            # valid_lm_loss
            valid_lm_loss = None
            try:
                if response_rear_m1 is not None:
                    # For gaining validation loss!
                    # 1. make solution if input_ids, toten_type_ids, lm_labels
                    # 2. forward and get lm_loss!

                    # solution_token_type_ids
                    if b_full:
                        gt_token_type_ids_final = token_type_ids[0, :self.max_length].unsqueeze(0)
                    else:
                        gt_response_no_m1 = response_rear_m1[response_rear_m1 != -1] # [1, 2, 3, -1, -1, -1]-> [1, 2, 3]
                        sys_token_type_num = gt_response_no_m1.shape[0] # 4

                        assert(sys_token_type_num >= 0)
                        if self.bsys_last_turn:
                            gt_resp_token_types = torch.ones((sys_token_type_num), device='cuda', dtype=torch.int64) * self.text_embedder.usr_idx
                        else:
                            gt_resp_token_types = torch.ones((sys_token_type_num), device='cuda', dtype=torch.int64) * self.text_embedder.sys_idx
                        gt_resp_token_types = gt_resp_token_types.unsqueeze(0)

                        gt_token_type_ids = torch.cat((origin_token_type_ids, gt_resp_token_types), dim=1) # generated
                        assert(gt_token_type_ids.shape[1] <= self.max_length)

                        token_type_pad_num = self.max_length - gt_token_type_ids.shape[1]
                        assert(token_type_pad_num >= 0)
                        token_type_pad = torch.ones((token_type_pad_num), device='cuda', dtype=torch.int64) * self.text_embedder.pad_idx
                        token_type_pad = token_type_pad.unsqueeze(0)
                        gt_token_type_ids_final = torch.cat((gt_token_type_ids, token_type_pad), dim=1)
                        assert(gt_token_type_ids_final.shape[1] <= self.max_length)

                    assert(gt_token_type_ids_final.shape[1] == self.max_length)

                    # gt_lm_labels
                    response_front_m1_num = self.max_length - response_rear_m1.shape[1]
                    assert(response_front_m1_num >= 0)
                    resp_front_m1 = torch.ones((response_front_m1_num), device='cuda', dtype=torch.int64) * -1
                    resp_front_m1 = resp_front_m1.unsqueeze(0)
                    gt_lm_labels = torch.cat((resp_front_m1, response_rear_m1), dim=1)
                    assert(gt_lm_labels.shape[1] == self.max_length)

                    # my_models' predict_input_ids
                    if b_full:
                        predict_input_ids = input_ids[0, :self.max_length].unsqueeze(0)
                    else:
                        if input_ids.shape[1] > self.max_length:
                            input_ids = input_ids[:, :self.max_length]

                        input_ids_pad_num = self.max_length - input_ids.shape[1]

                        assert(input_ids_pad_num >= 0)
                        input_ids_pad = torch.ones((input_ids_pad_num), device='cuda', dtype=torch.int64) * self.text_embedder.pad_idx
                        input_ids_pad = input_ids_pad.unsqueeze(0)
                        predict_input_ids = torch.cat((input_ids, input_ids_pad), dim=1)
                    assert(predict_input_ids.shape[1] == self.max_length)

                    assert (predict_input_ids.shape == gt_token_type_ids_final.shape == gt_lm_labels.shape)
                    valid_lm_loss, lm_logits, mc_logits, pres = self.model(predict_input_ids, token_type_ids=gt_token_type_ids_final, lm_labels=gt_lm_labels)
            except:
                import ipdb; ipdb.set_trace()

            # sentence
            sentence = self.text_embedder.tokenizer.decode(sys_utt)

            # resp_tokens
            #input_ids_list = input_ids[0].cpu().numpy().tolist()
            #input_tokens = text_embedder.tokenizer.convert_ids_to_tokens(input_ids_list, skip_special_tokens=False)
            #input_tokens = [self.transform_byte2normal(text_embedder.tokenizer, text_embedder.tokenizer.byte_decoder, token) for token in input_tokens]
            #token_type_ids_list = token_type_ids[0].cpu().numpy().tolist()
            #token_type_tokens = text_embedder.tokenizer.convert_ids_to_tokens(token_type_ids_list, skip_special_tokens=False)
            #token_type_tokens = [self.transform_byte2normal(text_embedder.tokenizer, text_embedder.tokenizer.byte_decoder, token) for token in
            #                     token_type_tokens]
            #self.print_toks(input_tokens, token_type_tokens, history_len=history_len)
            #resp_idx = len(input_tokens) - 1 - input_tokens[::-1].index('<user>')
            #resp_tokens = input_tokens[resp_idx:]
            resp_tokens = text_embedder.tokenizer.convert_ids_to_tokens(sys_utt)
            resp_tokens = [self.transform_byte2normal(text_embedder.tokenizer, text_embedder.tokenizer.byte_decoder, token) for token in resp_tokens]


            return valid_lm_loss, sentence, resp_tokens, None, None


    def top_filtering(self, logits, threshold=-float('Inf'), filter_value=-float('Inf')):

        """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
                top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                    whose total probability mass is greater than or equal to the threshold top_p.
                    In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                    the threshold top_p.
                threshold: a minimal threshold to keep logits
        """
        assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
        self.top_k = min(self.top_k, logits.size(-1))
        if self.top_k > 0:
            # Remove all tokens with a probability less than the last token in the top-k tokens
            indices_to_remove = logits < torch.topk(logits, self.top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if self.top_p > 0.0:
            # Compute cumulative probabilities of sorted tokens
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probabilities = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probabilities > self.top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Back to unsorted indices and set them to -infinity
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value

        indices_to_remove = logits < threshold
        logits[indices_to_remove] = filter_value

        return logits


class GptModel(Model):
    class Config(ConfigBase):
        representation: GPT.Config = GPT.Config()
        decoder: RetrievalDecoder.Config = RetrievalDecoder.Config()
        output_layer: RetrievalOutputLayer.Config = RetrievalOutputLayer.Config()


    def forward(self,
                text_embedder,
                input_ids: torch.Tensor,
                mc_token_ids: Optional[torch.Tensor],
                lm_labels: Optional[torch.Tensor],
                mc_labels: Optional[torch.Tensor],
                token_type_ids: Optional[torch.Tensor],
                mode='teacher') -> List[torch.Tensor]:
        lm_loss, mc_loss, lm_logits, mc_logits, pres = self.representation(text_embedder, input_ids, mc_token_ids,
                                                                           lm_labels, mc_labels, token_type_ids, mode)
        return lm_loss, mc_loss, lm_logits, mc_logits, pres


