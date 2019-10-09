import itertools
import json
import numpy as np
import os
import torch
import _pickle as pickle
import warnings
import logging
from .batch import right_pad_fixed_shape

from array import array
from collections import defaultdict
from copy import deepcopy
from hashlib import md5
from collections import Iterator

from mldc.data.config import ModelInput, ModelOutput, ModelOutputConfig, ModelInputConfig
from pytext.data import CommonMetadata
from pytext.data.data_handler import DataHandler, BatchIterator
from pytext.common.constants import DatasetFieldName, BatchContext
from pytext.fields import (Field, RawField, FieldMeta)
from pytext.utils import cuda_utils
from torchtext import data as textdata
from typing import Dict, Any, List, Type, Iterable, Optional, Union
from fairseq.data.dictionary import Dictionary as FairseqDict
from mldc.data.schema import MetaDlgDataDialog
from mldc.preprocessing.stream import stream_dlgs_many
from mldc.preprocessing.featurizer import TokenIdFeaturizer
from mldc.preprocessing.input_embedding import EmbedderInterface
from torch import LongTensor
from pydantic import BaseModel
from itertools import chain

from mldc.data.batchqueue import BatchProcessor, BatchQueue


LOG = logging.getLogger("mldc.data.data_handler")


def ensure_random_state(random_state):
    if random_state is None:
        return np.random.RandomState()
    if isinstance(random_state, int):
        return np.random.RandomState(random_state)
    assert isinstance(random_state, np.random.RandomState), "random_state must be None, an int, or a RandomState object"
    return random_state


class MetaBatch(list):
    def __init__(self, *args, max_n_turns=2):
        super().__init__(*args)
        setattr(self, ModelInput.DLG_LEN, max_n_turns)


class MetaSpec(BaseModel):
    target_dlg: str    # ID of the target dialogue
    support_dlgs: List[str]  # IDs of the support dialogues
    predict_turn: int  # a list of turns that can be predicted


RAW_TEXT = 'orig_text'
NEG_RAW_TEXT = 'neg_orig_text'

# hack to get around torchtext's stored reference to the dataset
class DatasetLike:
    def __init__(self, ds):
        self.name = ds.name if hasattr(ds, 'name') and isinstance(ds.name, str) else ''


class BatchLike:
    def __init__(self, batch, updated_fields={}, fields_to_cuda=[]):
        self.fields = batch.fields
        fields_to_cuda = set(fields_to_cuda)

        for f in batch.fields:
            val = updated_fields[f] if f in updated_fields else getattr(batch, f)

            if cuda_utils.CUDA_ENABLED and f in fields_to_cuda:
                if type(val) == torch.Tensor:
                    val = val.cuda()
                elif hasattr(val, '__iter__'):
                    val = tuple(v.cuda() if type(v) == torch.Tensor else v for v in val)

            setattr(self, f, val)


class BatchPreparationPipeline(Iterator):
    """
    Handles access to the multi-threaded batch preparation pipeline.

    Behaves like an iterator, but can be closed, which will send an
    end signal to the batch generator and join all workers.
    """

    def __init__(self, data_iterator, batch_queue, processor, is_train: bool, flush: bool = True):
        self.data_iterator = data_iterator
        self.batch_queue = batch_queue
        self.processor = processor
        self._iter = iter(batch_queue) # BatchQueue
        self.is_train = is_train
        self.flush = flush

    @property
    def max_n_turns(self):
        return self.data_iterator.max_n_turns

    @max_n_turns.setter
    def max_n_turns(self, max_n_turns):
        """
        takes effect when next epoch has started.
        Note that since the queue runs asynchronously, it might be started already even if training hasn't.
        """
        self.data_iterator.max_n_turns = max_n_turns

    def close(self):
        self.batch_queue.close()

    def __next__(self):
        while True:
            # ensure that the returned batch follows max_n_turns
            batch = next(self._iter)
            #  batch[0][0].sequence[0][0][0][0]
            #  tensor([50262,  1996,   279,   261,   319,   336,  6442,    13,  6756,   319,
            #         27635,    13,  1355,  8161,    13, 50261, 31442,  1392,   340,    13,
            #         50262,    33,  9437,  1660,    13, 50261, 19184,   318, 24372,    13,
            #         50262,  2617, 19781,   277,   313, 50267,  1084,  1769, 50262,   268,
            #          2633,  9653,    13, 50259, 50259, 50259, 50259, 50259, 50259, 50259,
            #         50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259, 50259,
            # batch[0][0].sequence[0].shape (128, 1, 2, 300)
            if not self.flush or getattr(batch, ModelInput.DLG_LEN) == self.max_n_turns:
                break

        input, target, context = self.processor( # _to_cuda_postprocess
            batch,
            include_input=True,
            include_target=True,
            include_context=True,
            is_train=self.is_train,
        )
        return (input, target, context)

    def __len__(self):
        return len(self.batch_queue)


class DataIterator:
    def __init__(self, dataset, batch_size: int, support_batch_size: int = 0, repeat: bool = False, shuffle: bool = False,
                 disjunct_tasks: bool = False, random_state: Optional[int] = None, allow_incomplete: bool = False,
                 meta_batch_size: int = 1, meta_batch_spec_file: Optional[str] = None, max_n_turns: int = 4):
        """
        args:
          - dataset: pytorch Dataset class, containing a list of example instances
          - batch_size: length of batch produced (target batch in case of meta-learning)
          - support_batch_size: number of support batch samples (meta-learning only)
          - disjunct_tasks: if True, support and target set have disjunct tasks (meta-learning only)
          - allow_incomplete: if the dataset size isn't divisible by batch size, the last batch will be smaller.
          - meta_batch_size: number of domains in a single meta-batch
          - meta_batch_spec_file: if given, support set and target is chosen according to the data in the file
          - max_n_turns: sent downstream to workers for dialogue cutoff (except for predict iterators)
        """
        self._dataset = dataset
        self._batch_size = batch_size                    # 128
        self._support_batch_size = support_batch_size    # 128
        self._repeat = repeat
        self._shuffle = shuffle
        self._disjunct_tasks = disjunct_tasks
        self._allow_incomplete = allow_incomplete
        self._meta_batch_size = meta_batch_size
        self._rng = ensure_random_state(random_state)
        self._update_dataset_info()
        self._meta_specs: List[MetaSpec] = []
        self.max_n_turns = max_n_turns

        if meta_batch_spec_file:
            with open(meta_batch_spec_file, 'rt') as f:
                for line in f:
                    self._meta_specs.append(MetaSpec(**json.loads(line)))

    @staticmethod
    def _wrap_batch(batch, max_n_turns):
        """
        make batch serializable so it can be passed on to workers
        """
        batch.dataset = DatasetLike(batch.dataset)
        setattr(batch, ModelInput.DLG_LEN, max_n_turns)
        return batch

    def _update_dataset_info(self):
        self._tasks = list({getattr(e, ModelInput.TASK_ID) for e in self._dataset.examples})
        self._domains = list({getattr(e, ModelInput.DOMAIN_ID) for e in self._dataset.examples})
        self._task_dlgs = defaultdict(lambda: defaultdict(lambda: array('I')))   # x[domain][task] -> indices
        self._domain_dlgs = defaultdict(lambda: array('I'))   # x[domain] -> indices
        for eid, example in enumerate(self._dataset.examples):
            self._task_dlgs[getattr(example, ModelInput.DOMAIN_ID)][getattr(example, ModelInput.TASK_ID)].append(eid)
            self._domain_dlgs[getattr(example, ModelInput.DOMAIN_ID)].append(eid)

    @staticmethod
    def n_batches(n_examples: int, batch_size: int, allow_incomplete: bool):
        if not allow_incomplete:
            return n_examples // batch_size
        return (n_examples + batch_size) // batch_size

    @staticmethod
    def grouper(values, n: int, allow_incomplete: bool):
        args = [iter(values)] * n
        if allow_incomplete:
            return itertools.zip_longest(*args)
        return zip(*args)

    def create_batches(self):
        """
        Returns a list of batches or meta-batches.

        The way batches are constructed differs:
        - if meta_specs was provided, returns meta batches with a single domain and the specified support/target set.
        - if support batch size is >0, return random meta batches
        - else return regular batches
        """
        if self._meta_specs:
            # meta batches according taking dialogue ids from a file
            if self._support_batch_size > 0:
                return self._create_meta_batches_from_spec(self._meta_specs)
            else:
                return self._create_batches_from_spec()

        if self._support_batch_size > 0:
            # create meta batches (>=1 domain, support set, target set)
            temp = self._create_meta_batches()
            return temp

        self._idx = np.arange(len(self._dataset))
        if self._shuffle:
            self._rng.shuffle(self._idx)
        # no bucketing for now
        return self.grouper(self._idx, self._batch_size, allow_incomplete=self._allow_incomplete)

    def _create_batches_from_spec(self):
        """
        Creates batches for prediction from spec file, given no support set
        i.e. zero-shot prediction
        """
        id2dlg = {dlg.dlg_id: i for i, dlg in enumerate(self._dataset)}
        batches = []
        for spec in self._meta_specs:
            if spec.target_dlg not in id2dlg:
                # only load batches which are part of the loaded dataset
                continue
            target_example = id2dlg[spec.target_dlg]
            mb = MetaBatch([([], [target_example], spec.predict_turn)],
                           max_n_turns=self.max_n_turns)
            batches.append(mb)
        return batches

    def _create_meta_batches_from_spec(self, meta_specs):
        """
        Create meta batches for a single domain, given a specification that contains:

         - a list of support dialogue ids
         - a single target dialogue id
         - the target turn in the target dialogue
        """
        assert self._batch_size == 1, "Need batch_size==1 to create batches from spec file."
        assert self._support_batch_size > 0, "spec file only makes sense for support_batch_size > 0"
        assert self._meta_batch_size == 1, "spec file only makes sense for meta_batch_size == 1"
        id2dlg = {dlg.dlg_id: i for i, dlg in enumerate(self._dataset)}
        meta_batches = []
        for spec in self._meta_specs:
            if spec.target_dlg not in id2dlg:
                # only load batches which are part of the loaded dataset
                continue
            support_dlgs = spec.support_dlgs
            if len(support_dlgs) < self._support_batch_size:
                warnings.warn("Inconsistency: support batch size from spec file less than requested support batch size",
                              RuntimeWarning)
            elif len(support_dlgs) > self._support_batch_size:
                # sample subset of dialogues
                support_dlgs = list(support_dlgs)  # copy list
                support_dlgs = self._rng.choice(support_dlgs, replace=False, size=self._support_batch_size)
                warnings.warn(
                    "Inconsistency: support batch size from spec file larger than requested support batch size. Subsampling.",
                    RuntimeWarning)
            support_examples = [id2dlg[id] for id in spec.support_dlgs]
            target_example = id2dlg[spec.target_dlg]
            mb = MetaBatch([(support_examples, [target_example], spec.predict_turn)],
                           max_n_turns=self.max_n_turns)
            meta_batches.append(mb)
        return meta_batches

    def _create_meta_batches(self):
        """
        return meta batches. A meta batch contains meta_batch_size pairs of support set and a target set. Each pair is from the same
        domain. All pairs are from different domains.
        """
        # [JG_eureka]
        batches = defaultdict(list)
        for didx, domain in enumerate(self._domains):
            domain_dlgs = np.asarray(self._domain_dlgs[domain])
            if self._shuffle:
                self._rng.shuffle(domain_dlgs)
            n_domain_examples = len(domain_dlgs)
            n_domain_batches = self.n_batches(n_domain_examples, self._batch_size, self._allow_incomplete)
            mask = np.empty(n_domain_examples)
            for bidx in range(n_domain_batches):
                target_slice = slice(bidx * self._batch_size, (bidx + 1) * self._batch_size)
                target_examples = domain_dlgs[target_slice]
                mask[:] = 1 / (n_domain_examples - len(target_examples))
                mask[target_slice] = 0.
                support_examples = self._rng.choice(domain_dlgs, replace=False, p=mask, size=self._support_batch_size)
                predict_turn = None
                batches[didx].append((support_examples, target_examples, predict_turn))

        # so far we only shuffled within domains, so we may need to shuffle again
        meta_batches = []
        for bidx in itertools.count():
            if len(batches) < self._meta_batch_size:
                if len(batches) != 0:
                    if bidx == 0:
                        raise RuntimeError("Not enough domains to create meta-batches of size %d, only have domains: %s" %
                                           (self._meta_batch_size, ", ".join(self._domains)))
                    warnings.warn("Discarding domain-incomplete meta-batches")
                break

            # sample mbs *different* domains
            metabatch_domains = self._rng.choice(list(batches.keys()), self._meta_batch_size, replace=False)
            meta_batch = []
            # pick 1st batch from each domain for meta-batch
            for domain in metabatch_domains:
                meta_batch.append(batches[domain][0])

                # remove 1st batch from each domain, remove empty domains
                if len(batches[domain]) == 1:
                    del batches[domain]
                else:
                    batches[domain] = batches[domain][1:]
            meta_batches.append(meta_batch)
        return meta_batches

    def _examples_to_batch(self, batch, max_n_turns):

        def truncate_target(target_ex, predict_turn):
            # cut off dialogue after predict_turn.
            # downstream, we'll just predict the last one.
            target_ex[0] = deepcopy(target_ex[0])

            turns = getattr(target_ex[0], ModelInput.SEQ)[:predict_turn + 1]
            setattr(target_ex[0], ModelInput.SEQ, turns)

            turns = getattr(target_ex[0], ModelOutput.TOK)[:predict_turn + 1]
            setattr(target_ex[0], ModelOutput.TOK, turns)

        if self._support_batch_size == 0:
            if self._meta_specs:
                batch, = batch
                _, ex, predict_turn = batch
                ex = [self._dataset[i] for i in ex]
                if predict_turn is not None:
                    truncate_target(ex, predict_turn)
                return self._wrap_batch(textdata.Batch(ex, self._dataset), max_n_turns)

            else:
                return self._wrap_batch(
                    textdata.Batch([self._dataset[i] for i in batch if i is not None], self._dataset),
                    max_n_turns)

        mb = MetaBatch(max_n_turns=max_n_turns)
        for domain_batch in batch:
            support_ex, target_ex, predict_turn = domain_batch
            support_ex = [self._dataset[i] for i in support_ex]
            target_ex = [self._dataset[i] for i in target_ex]
            if predict_turn is not None:
                truncate_target(target_ex, predict_turn)

            mb.append((self._wrap_batch(textdata.Batch(support_ex, self._dataset), max_n_turns),
                       self._wrap_batch(textdata.Batch(target_ex, self._dataset), max_n_turns)))
        return mb

    def __iter__(self):
        """
        loop over batches in the dataset
        """
        while True:
            batches = self.create_batches()  #(127)
            for batch in batches: # (2, 3)
                iter_yield = self._examples_to_batch(batch, self.max_n_turns) # MetaBatch
                # iter_yield[0][0]
                batch2 = iter_yield[0][0]
                yield iter_yield # 2 because of support and target
            if not self._repeat:
                break

    def __len__(self):
        """ Returns how many batches are in one pass of the dataset """
        if self._support_batch_size == 0 and not self._meta_specs:
            # plain batches
            return self.n_batches(len(self._dataset.examples), self._batch_size, self._allow_incomplete)
        if self._meta_specs:
            # meta batches with target set size 1 specified in _meta_specs
            id2dlg = {dlg.dlg_id: i for i, dlg in enumerate(self._dataset)}
            return sum(1 for dlg in self._meta_specs if dlg.target_dlg in id2dlg)
        n_examples = min([len(self._domain_dlgs[domain]) for domain in self._domains])
        return self.n_batches(n_examples, self._batch_size, False)


class BPEField(RawField):

    def __init__(self, text_embedder: EmbedderInterface, is_target=False):
        self.is_target = is_target
        self.meta = FieldMeta()
        self.meta.vocab_size = text_embedder.n_vocab
        self.meta.pad_token_idx = text_embedder.pad_idx  # This is set to 0 in SPM train
        self.meta.unk_token_idx = text_embedder.unk_idx
        self.meta.bos_token_idx = text_embedder.bos_idx
        self.meta.eos_token_idx = text_embedder.eos_idx
        self.use_vocab = False
        self.postprocessing = None

    def load_meta(self, meta):
        self.meta = meta

    def preprocessing(self, dlg):
        return dlg

    def get_meta(self) -> FieldMeta:
        return self.meta


class CustomBatchProcessor(BatchProcessor):
    """
    Runs in a separate thread and does some preprocessing on the example dialogues

    - selecting how many turns the dialogues passed to the network should contain
    - cut off turns that are too long
    - pad dialogues on the turn axis and the word axis
    - embed dialogues using the text_embedder (e.g. fasttext, BERT)

    """

    def __init__(self, embedder_cfg: EmbedderInterface.Config,
                 fixed_n_turns: bool = False,
                 all_responses: bool = False):
        # Common setup in the process for it's lifetime
        self.text_embedder = EmbedderInterface.from_config(embedder_cfg)
        self.pad_token_idx = self.text_embedder.pad_idx
        self.unk_token_idx = self.text_embedder.unk_idx
        self.bos_token_idx = self.text_embedder.bos_idx
        self.eos_token_idx = self.text_embedder.eos_idx
        self.fixed_n_turns = fixed_n_turns
        self.all_responses = all_responses
        self.max_len = 300

    def process_batch(self, batch: Union[textdata.Batch, MetaBatch]):
        """
        Processes a batch. If it is a meta-batch, independently process support and target sets.
        """
        if isinstance(batch, (textdata.Batch, BatchLike)):
            return self.process_batch_nometa(batch, self.fixed_n_turns, False)

        meta_batch = MetaBatch(  # batch : (1) batch[0] : (2)   class MetaBatch(list): def __init__(self, *args, max_n_turns=2): super().__init__(*args) setattr(self, ModelInput.DLG_LEN, max_n_turns)
            [(self.process_batch_nometa(domain[0], False, self.all_responses), # self.all_responses == True
              self.process_batch_nometa(domain[1], self.fixed_n_turns, False)) for domain in batch], # self.fixed_n_turns == False  ## self.process_batch_nometa ==> BatchLike
            max_n_turns=getattr(batch, ModelInput.DLG_LEN)
        )
        return meta_batch

    def process_batch_nometa(self, batch: Union[textdata.Batch, BatchLike], fixed_n_turns: bool, all_responses: bool):
        """
        This does what `BPEField.postprocess` used to do, with some caveats
        - returns a `Batch`-like object instead of raw tensors, since `postprocess` should be called during `Batch`
          creation. The `Batch` is created before this call, so we must operate on it.
        - nothing is allocated to the GPU here since `BatchQueue`'s dependency isn't implemented with
          `torch.multiprocessing`.
        """

        def make_batch_tensors(dlgs, is_input=True):
            # there's an extra dimension to accommodate the retrieval case
            n_seqs = [len(dlg) for dlg in dlgs]
            n_words = [[len(t) for t in turns] for turns in dlgs]

            max_n_turns = max(n_seqs)
            max_n_tokens = max([max(dlg) for dlg in n_words])
            embed_dim = self.text_embedder.embed_dim

            max_shape = (batchsz, max_n_turns, max_n_tokens, embed_dim)

            padded_n_words = right_pad_fixed_shape(n_words, max_shape=max_shape[:2], dtype=int)
            padded_turns = right_pad_fixed_shape(dlgs, max_shape=max_shape[:3], dtype=int, value=self.pad_token_idx)

            dlgs_length = []
            for dlg_i, dlg in enumerate(dlgs):
                dlg_length = []
                for dlg_each_i, dlg_each in enumerate(dlg):
                    dlg_length.append(len(dlg_each))
                dlgs_length.append(dlg_length)

            emb = None
            if is_input:
                emb = self.text_embedder.embed_ids_batch(padded_turns.reshape(-1, max_n_tokens)).reshape(*max_shape)

            # [JG_IMPORTANT]
            return LongTensor(padded_turns), LongTensor(n_seqs), LongTensor(padded_n_words), emb

        def make_batch_target_tensors(dlgs):
            padded_turns, _, n_words, _ = make_batch_tensors(dlgs, is_input=False)
            if not all_responses:
                # remove turn dimension
                padded_turns = torch.squeeze(padded_turns, 1)
                n_words = torch.squeeze(n_words, 1)
            return padded_turns, n_words

        def make_sequence_tensor(instance_dict):
            batchsz = instance_dict['input_ids'].__len__()
            input_ids = np.array(instance_dict['input_ids']).reshape(batchsz,-1,2,self.max_len)
            mc_token_ids = np.array(instance_dict['mc_token_ids']).reshape(batchsz,-1,2)
            lm_labels = np.array(instance_dict['lm_labels']).reshape(batchsz,-1,2,self.max_len)
            mc_labels = np.array(instance_dict['mc_labels']).reshape(batchsz,-1)
            token_type_ids = np.array(instance_dict['token_type_ids']).reshape(batchsz,-1,2,self.max_len)
            return LongTensor(input_ids), LongTensor(mc_token_ids), LongTensor(lm_labels), LongTensor(mc_labels), LongTensor(token_type_ids)

        def create_gpt_input_unit(history, raw_history, pos_resp, rpos_resp, neg_resp, rneg_resp,
                                  history_turn_types, presp_turn_type, nresp_turn_type):
            """

            :param history: [[], [], [], ... []]
            :param raw_history:
            :param pos_resp:
            :param rpos_resp:
            :param neg_resp:
            :param rneg_resp:
            :param history_turn_types:
            :param presp_turn_type:
            :param nresp_turn_type:
            :return:
            """
            assert(len(list(chain(*history))) == len(list(chain(*history_turn_types))))
            assert(len(pos_resp) == len(presp_turn_type))
            assert(len(neg_resp) == len(nresp_turn_type))

            input_ids, mc_token_ids, lm_labels, mc_labels, turn_type_ids = [], [], [], [], []

            pos_seq = history + [pos_resp]
            pos_turn_types = history_turn_types + [presp_turn_type]

            pos_input_ids = list(chain(*pos_seq))
            pos_mc_token_ids = len(pos_input_ids) - 1
            pos_lm_labels = [-1] * len(list(chain(*history))) + pos_resp #
            pos_token_types = list(chain(*pos_turn_types))
            try:
                assert(len(pos_input_ids) == len(pos_lm_labels))
                assert(len(pos_input_ids) == len(pos_token_types))
            except:
                print("len(pos_input_ids): ", len(pos_input_ids))
                print("len(pos_lm_labels): ", len(pos_lm_labels))
                print("len(pos_token_types): ", len(pos_token_types))

            if len(pos_input_ids) > self.max_len:
                excess = len(pos_input_ids) - self.max_len
                pos_input_ids = pos_input_ids[-self.max_len:]
                pos_mc_token_ids = pos_mc_token_ids - excess
                pos_lm_labels = pos_lm_labels[-self.max_len:]
                pos_token_types = pos_token_types[-self.max_len:]
            else:
                num_pad = self.max_len - len(pos_input_ids)
                pos_input_ids = pos_input_ids + [self.pad_token_idx] * num_pad
                pos_lm_labels = pos_lm_labels + [-1] * num_pad
                pos_token_types = pos_token_types + [self.pad_token_idx] * num_pad

            assert(len(pos_input_ids) == self.max_len)
            assert(pos_mc_token_ids < self.max_len)
            assert(len(pos_lm_labels) == self.max_len)
            assert(len(pos_token_types) == self.max_len)

            neg_seq = history + [neg_resp]
            neg_turn_types = history_turn_types + [nresp_turn_type]

            neg_input_ids = list(chain(*neg_seq))
            neg_mc_token_ids = len(neg_input_ids) - 1
            neg_lm_labels = [-1] * len(neg_input_ids)
            neg_token_types = list(chain(*neg_turn_types))
            try:
                assert(len(neg_input_ids) == len(neg_lm_labels))
                assert(len(neg_input_ids) == len(neg_token_types))
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

            assert(len(neg_input_ids) == self.max_len)
            assert(neg_mc_token_ids < self.max_len)
            assert(len(neg_token_types) == self.max_len)
            assert(len(neg_lm_labels) == self.max_len)

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

        # [JG_INFO] batch attr; batch_size, dataset, dlg_id, dlg_len, domain_id, fields (['seq_word_feat', 'orig_text', 'dlg_len ...']), index, input_fields, orig_text, out_tokens, seq_word_feat, target_fields, task_id
        n_turns = getattr(batch, ModelInput.DLG_LEN)
        history_mat, pos_resp_mat, neg_resp_mat = [], [], []
        history_turn_type_mat, pos_resp_turn_type_mat, neg_resp_turn_type_mat = [], [], []
        raw_history_mat, raw_pos_resp_mat, raw_neg_resp_mat = [], [], []

        for pos_turns, neg_turns, raw_pos_turns, raw_neg_turns in zip(getattr(batch, ModelInput.SEQ),
                                                              getattr(batch, ModelInput.NEG_SEQ),
                                                              getattr(batch, RAW_TEXT),
                                                              getattr(batch, NEG_RAW_TEXT)):
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

        batchsz = len(batch)
        matrix_iterator = zip(history_mat, pos_resp_mat, neg_resp_mat, raw_history_mat, raw_pos_resp_mat, raw_neg_resp_mat,
                              history_turn_type_mat, pos_resp_turn_type_mat, neg_resp_turn_type_mat)
        instance_dict = {'input_ids': [], "mc_token_ids": [], "lm_labels": [], "mc_labels": [], "token_type_ids": []}
        for history, pos_resps, neg_resps, raw_history, raw_pos_resps, raw_neg_resps, \
            history_turn_types, pos_resp_turn_types, neg_resp_turn_types in matrix_iterator:

            if all_responses:
                resps_zip = zip(pos_resps, raw_pos_resps, neg_resps, raw_neg_resps, pos_resp_turn_types, neg_resp_turn_types)
                for resp_idx, (pos_resp, rpos_resp, neg_resp, rneg_resp, presp_turn_type, nresp_turn_type) in enumerate(resps_zip):
                    history_idx = 2 * resp_idx + 1
                    this_history = history[:history_idx]
                    this_raw_history = raw_history[:history_idx]
                    this_history_turn_types = history_turn_types[:history_idx]
                    input_ids, mc_token_ids, lm_labels, mc_labels, turn_type_ids = create_gpt_input_unit(
                        this_history, this_raw_history, pos_resp, rpos_resp, neg_resp, rneg_resp,
                        this_history_turn_types, presp_turn_type, nresp_turn_type)
                    instance_dict['input_ids'].append(input_ids)
                    instance_dict['mc_token_ids'].append(mc_token_ids)
                    instance_dict['lm_labels'].append(lm_labels)
                    instance_dict['mc_labels'].append(mc_labels)
                    instance_dict['token_type_ids'].append(turn_type_ids)
            else:
                assert(len(pos_resps) == 1)
                resp_tuple = next(zip(pos_resps, raw_pos_resps, neg_resps, raw_neg_resps, pos_resp_turn_types, neg_resp_turn_types))
                pos_resp, rpos_resp, neg_resp, rneg_resp, presp_turn_type, nresp_turn_type = resp_tuple
                input_ids, mc_token_ids, lm_labels, mc_labels, turn_type_ids = create_gpt_input_unit(
                    history, raw_history, pos_resp, rpos_resp, neg_resp, rneg_resp,
                    history_turn_types, presp_turn_type, nresp_turn_type)
                instance_dict['input_ids'].append(input_ids)
                instance_dict['mc_token_ids'].append(mc_token_ids)
                instance_dict['lm_labels'].append(lm_labels)
                instance_dict['mc_labels'].append(mc_labels)
                instance_dict['token_type_ids'].append(turn_type_ids)

        fields = {
            #ModelInput.SEQ: make_batch_tensors(history_mat),
            ModelInput.SEQ: make_batch_tensors(history_mat, is_input=False), # 4 outputs
            ModelOutput.TOK: make_batch_target_tensors(pos_resp_mat), # ModelOutput {TOK : 'out_tokens'}
            ModelOutput.NEG_TOK: make_batch_target_tensors(neg_resp_mat), #
            ModelInput.SEQUENCE: make_sequence_tensor(instance_dict),
        }

        return BatchLike(batch, updated_fields=fields)


class MetaData(CommonMetadata):
    source_dict: FairseqDict
    target_dict: FairseqDict


class DialogueDataHandler(DataHandler):
    class Config(DataHandler.Config):
        # determines by which field data is sorted
        text_feature_name: str = ModelInput.SEQ
        # shuffle: bool = False
        # sort_within_batch: bool = False
        max_turns: int = 12
        n_workers: int = 4
        max_load_workers: int = 4
        seed: int = 42
        # TODO featurized_cache_dir !!!
        featurized_cache_dir: str = './feature_cache'
        train_domains: List[str] = []
        eval_domains: List[str] = []
        test_domains: List[str] = []
        # dictates how many samples go to a process, and determines the lifetime of the process
        # make this larger for larger datasets
        preproc_chunksize: int = 1000
        all_responses: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.metadata_cls: Type = MetaData
        self.metadata: MetaData = MetaData()

        # hint for mypy (otherwise it assumes pytext's class Featurizer from parent
        # class)
        self.featurizer: TokenIdFeaturizer = kwargs['featurizer']
        self.n_workers: int = kwargs['n_workers']
        self.max_load_workers: int = kwargs['max_load_workers']

    @classmethod
    def from_config(cls, config: Config,
                    feature_config: ModelInputConfig,
                    target_config: ModelOutputConfig,
                    text_embedder_config: EmbedderInterface.Config,
                    **kwargs):

        gptmode = 'gpt2'
        #gptmode = 'openai-gpt'
        if gptmode == 'gpt2':
            config.featurized_cache_dir = './feature_cache_gpt2'
        else:
            config.featurized_cache_dir = './feature_cache_gpt'

        text_embedder: EmbedderInterface = EmbedderInterface.from_config(text_embedder_config)
        features: Dict[str, Field] = {
            ModelInput.SEQ: BPEField(text_embedder)
        }
        assert len(features)

        targets: Dict[str, Field] = {
            ModelOutputConfig._name: BPEField(text_embedder, is_target=True),
        }

        extra_fields = {
            RAW_TEXT: RawField(),
            NEG_RAW_TEXT: RawField(),
            ModelInput.DLG_LEN: RawField(),
            ModelInput.DLG_ID: RawField(),
            ModelInput.DOMAIN_ID: RawField(),
            ModelInput.TASK_ID: RawField(),
            ModelInput.NEG_SEQ: RawField(),
            ModelInput.ITTI: RawField(),
            ModelInput.TTTI: RawField(),
            ModelInput.NTTTI: RawField(),
            ModelInput.SEQUENCE: RawField(),
            ModelInput.HISTORY: RawField(),
            ModelOutputConfig._neg: BPEField(text_embedder, is_target=True),
            ModelInput.SEQUENCE: RawField(),
        }

        kwargs.update(config.items())
        self = cls(
            raw_columns=[],  # ignored in our read function
            features=features,
            labels=targets,
            extra_fields=extra_fields,
            **kwargs,
        )
        self.max_turns = config.max_turns
        self.text_embedder_cfg = text_embedder_config
        self.all_responses = config.all_responses
        self.preproc_chunksize = config.preproc_chunksize
        self.train_domains = config.train_domains
        self.eval_domains = config.eval_domains
        self.featurized_cache_dir = config.featurized_cache_dir
        self.test_domains = config.test_domains
        self.text_embedder = text_embedder
        self.seed = config.seed
        return self



    def preprocess_row(self, row_data: MetaDlgDataDialog) -> Dict[str, Any]:
        featurized = self.featurizer.featurize(row_data)

        res = {
            # features
            ModelInput.SEQ: featurized.token_ids,

            # target
            ModelOutputConfig._name: featurized.token_ids,

            RAW_TEXT: featurized.token_ids,
        }
        return res

    def _get_batch_iter(
            self,
            dataset: textdata.Dataset, # 'examples', 'fields'
            batch_size: int,          # 128
            meta_batch_size: int = 1, # 2
            rank: int = 0,
            world_size: int = 1, # 1
            repeat: bool = True,
            n_workers: int = 4,
            is_train: bool = True,
            is_predict: bool = False,
            **kwargs
    ) -> BatchIterator:
        if world_size > 1 and kwargs.get("meta_batch_spec_file"):
            raise RuntimeError("sharding not supported if meta_batch_spec_file is given")
        dataset_shard, max_num_examples = self._get_dataset_shard( # max_num_examples = 35665
            dataset, rank, world_size
        )
        assert not (is_train and is_predict)

        # Compute the per-worker batch size
        assert (
                batch_size >= world_size
        ), "batch size needs to be >= the distributed world size"
        # TODO should we modify meta_batch_size here?
        batch_size = batch_size // world_size

        if 'max_n_turns' in kwargs.keys():
            diter = DataIterator(
                dataset_shard,
                batch_size=batch_size,
                repeat=repeat, # True
                shuffle=self.shuffle,
                allow_incomplete=not repeat, # False
                meta_batch_size=meta_batch_size,
                random_state=self.seed,
                **kwargs
            )  # yields either textdata.Batch or MetaBatch (containing textdata.Batch objects)
        else:
            diter = DataIterator(
                dataset_shard,
                batch_size=batch_size,
                repeat=repeat, # True
                shuffle=self.shuffle,
                allow_incomplete=not repeat, # False
                meta_batch_size=meta_batch_size,
                random_state=self.seed,
                max_n_turns=20, ############# 3
                **kwargs
            )  # yields either textdata.Batch or MetaBatch (containing textdata.Batch objects)



        n_batches = len(diter) # 2

        # enqueues BatchLike or a nested structure of BatchLike, because Batch is not pickleable
        # CustomBatchProcessor produces the same.
        bq = BatchQueue(
            diter,
            n_batches, # 2
            CustomBatchProcessor,
            n_workers=n_workers,
            qcap=3, # [JG] 3
            embedder_cfg=self.text_embedder_cfg,
            fixed_n_turns=is_predict,
            all_responses=self.all_responses,
        )

        print("BatchQueue done!")
        return BatchPreparationPipeline(
            diter, bq, processor=self._to_cuda_postprocess_batch, is_train=is_train)

    def gen_dataset(
            self, data: Iterable[MetaDlgDataDialog], include_label_fields: bool = True,
            featurized_path: str = ''
    ) -> textdata.Dataset:
        """
        Generate torchtext Dataset from raw in memory data.
        Returns:
            dataset(TorchText.Dataset)

        *NOTE*: order will vary between `data` and what's loaded from `featurized_path`, or generated from
        `parallel_featurize_batch`. This is fine since the featurized data encompasses everything needed for the
        torchtext Dataset anyway.
        """
        to_process = {}
        to_process.update(self.features)
        to_process.update(self.extra_fields) # line 625 self.extra_fields
        if include_label_fields:
            to_process.update(self.labels)
        fields = {name: (name, field) for name, field in to_process.items()}

        # Optimizations for parallel preproc and cached preproc
        if featurized_path and os.path.exists(featurized_path):
            with open(featurized_path, 'rb') as f:
                featurized_data = pickle.load(f)
        else:
            # bypass the preprocess... methods in the datahandler to parallelize
            featurized_data = TokenIdFeaturizer.parallel_featurize_batch(
                data,
                text_embedder_cfg=self.text_embedder_cfg,
                chunksize=self.preproc_chunksize, # 1000
                max_workers=self.max_load_workers, # 4
            )
            if featurized_path:
                with open(featurized_path, 'wb') as f:
                    pickle.dump(featurized_data, f)

        examples = []

        for idx, featurized in enumerate(featurized_data): # featurized_data : <itertools.chain> # 35000~
            row = { # index, seq_word_feat, task_id, domain_id, dlg_id, out_tokens, orig_text, dlg_len
                BatchContext.INDEX: idx, # 'index'
                ModelInput.SEQ: featurized.token_ids, # 'seq_word_feat'
                ModelInput.TASK_ID: featurized.task_id, # 'task_id'
                ModelInput.DOMAIN_ID: featurized.domain_id, # 'domain_id'
                ModelInput.DLG_ID: featurized.dlg_id,  # 'dlg_id'
                ModelOutputConfig._name: featurized.token_ids, #'out_tokens'
                #RAW_TEXT: featurized.token_ids, # 'orig_text'
                RAW_TEXT: featurized.turns, # 'orig_text' # [JG_edit]
                NEG_RAW_TEXT: featurized.neg_turns,
                ModelInput.DLG_LEN: 2, # 'dlg_len'
                ModelInput.NEG_SEQ: featurized.neg_token_ids,
                ModelInput.ITTI: [],
                ModelInput.TTTI: [],
                ModelInput.NTTTI: [],
                ModelInput.HISTORY: [],
                ModelInput.SEQUENCE: [],
                ModelOutputConfig._neg: featurized.neg_token_ids,
            } # featurized : dlg_id, domain_id, task_id, token_ids (11)
            example = textdata.Example.fromdict(row, fields) # example  # 'dlg_id' : '77e3846c' # 'dlg_len' : 2 # 'domain_id' : 'WEATHER_CHECK'# index = 0
            examples.append(example)

        dts = textdata.Dataset(examples, to_process) # 'examples', 'fields'
        return dts

    def get_trainset(self, dataset):
        pass

    def gen_dataset_from_path(self, path: str, include_label_fields: bool = True, use_cache: bool = True, domains: List = [],
                              batch_size=None, support_batch_size=None):
        """
        load from json instead of csv
        """

        LOG.debug("Generating dataset from path: %s, domains: %s", path, ', '.join(tuple(domains[:3]) + ('...',)))
        featurized_path = ''
        if self.featurized_cache_dir:
            os.makedirs(self.featurized_cache_dir, exist_ok=True)
            # Featurized filename is just the hash of the filename + sorted domains
            feathash = md5(f"{path}'-'{'-'.join(sorted(domains))}".encode('utf-8')).hexdigest()
            featurized_path = os.path.join(self.featurized_cache_dir, feathash + '.pkl')

        min_domain_size = None
        if support_batch_size:
            min_domain_size = batch_size + support_batch_size

        if not domains:
            raise RuntimeError(f'No files specified in .zip archive {path}!')

        data_key = f"{path}/{','.join(sorted(domains))}"
        if use_cache and data_key in self._data_cache:
            LOG.debug("Found existing cache, loading from there.")
            return self._data_cache[data_key]

        dlgs = stream_dlgs_many(path, domains, min_domain_size=min_domain_size) # dlgs <generator object stream_dlgs_many>
        res = self.gen_dataset(dlgs, include_label_fields=include_label_fields, featurized_path=featurized_path) # res.examples : (35,565) # [JG_INFO]
        # INFO:mldc.preprocessing.featurizer:Loading dialogues in chunks of size 1000: 0 chunks [00:00, ? chunks/s]
        # INFO:mldc.preprocessing.featurizer:Loading dialogues in chunks of size 1000: 36 chunks [00:02, 13.67 chunks/s]
        # INFO:mldc.preprocessing.featurizer:Processing dialogues in chunks of size 1000:   0%|          | 0/36 [00:00<?, ? chunks/s]
        # INFO:mldc.preprocessing.featurizer:Processing dialogues in chunks of size 1000:  31%|###       | 11/36 [00:31<01:12,  2.90s/ chunks]
        # INFO:mldc.preprocessing.featurizer:Processing dialogues in chunks of size 1000:  31%|###       | 11/36 [00:50<01:12,  2.90s/ chunks]
        # INFO:mldc.preprocessing.featurizer:Processing dialogues in chunks of size 1000:  69%|######9   | 25/36 [01:04<00:30,  2.73s/ chunks]
        # INFO:mldc.preprocessing.featurizer:Processing dialogues in chunks of size 1000:  69%|######9   | 25/36 [01:20<00:30,  2.73s/ chunks]
        # INFO:mldc.preprocessing.featurizer:Processing dialogues in chunks of size 1000: 100%|##########| 36/36 [01:30<00:00,  2.52s/ chunks]

        self._data_cache[data_key] = res # <torchtext.data.dataset.Dataset>  'examples', 'fields'
        LOG.debug("Done loading dataset")
        return res

    # DialogDataHandler
    def get_train_iter_from_path(
            self, train_path: str, batch_size: int, rank: int = 0, world_size: int = 1, domains: List = [],
            **kwargs
    ):
        #-> BatchIterator:
        if rank == 0:
            return self._get_batch_iter(
                self.gen_dataset_from_path(train_path, domains=domains, batch_size=batch_size,
                                          support_batch_size=kwargs.get('support_batch_size')), # res
                batch_size, rank=rank, world_size=world_size, n_workers=self.n_workers, **kwargs)
        elif rank == 1:
            temp = self.gen_dataset_from_path(train_path, domains=domains, batch_size=batch_size,
                                              support_batch_size=kwargs.get('support_batch_size'))
            return temp
            # temp.examples.__len__()


    def get_eval_iter_from_path(
            self, eval_path: str, batch_size: int, rank: int = 0, world_size: int = 1, domains: List = [],
            **kwargs
    ) -> BatchIterator:
        return self._get_batch_iter(
            self.gen_dataset_from_path(eval_path, domains=domains, batch_size=batch_size,
                                       support_batch_size=kwargs.get('support_batch_size')),
            batch_size, rank=rank, world_size=world_size, n_workers=self.n_workers, **kwargs)

    def get_test_iter_from_path(
            self, test_path: str, batch_size: int, rank: int = 0, world_size: int = 1, domains: List = [],
            **kwargs
    ) -> BatchIterator:
        kwargs['repeat'] = False
        kwargs['is_train'] = False ####### [JG_WARNING]
        return self._get_batch_iter(
            self.gen_dataset_from_path(test_path, domains=domains), batch_size, n_workers=1, **kwargs)

    def get_predict_iter(self, data: List[Dict[str, Any]]):
        """ Used by default task.predict """
        ds = self.gen_dataset(data)
        it = self._get_batch_iter(
            ds,
            batch_size=len(ds),
            n_workers=1,
            repeat=False,
            is_train=False,
        )

        input, _, context = next(it)
        it.close()
        return input, context

    def get_predict_iter_from_path(
            self, predict_path: str, batch_size: int, rank: int = 0, world_size: int = 1, domains: List = [],
            **kwargs
    ) -> BatchIterator:
        kwargs['repeat'] = False
        kwargs['is_predict'] = True
        return self._get_batch_iter(
            self.gen_dataset_from_path(predict_path, domains=domains), batch_size,
            n_workers=self.n_workers, **kwargs)

    def get_test_iter(self, rank: int = 0, world_size: int = 1, **kwargs):
        return self.get_test_iter_from_path(self.test_path, self.test_batch_size, domains=self.test_domains,
                                            repeat=False, **kwargs)

    def get_train_iter(self, rank: int = 0, world_size: int = 1, **kwargs):
        if rank == 0:
            return self.get_train_iter_from_path(self.train_path, self.train_batch_size, domains=self.train_domains, **kwargs)
        elif rank == 1:
            return self.get_train_iter_from_path(self.train_path, self.train_batch_size, rank=1, domains=self.train_domains, **kwargs)

    def get_eval_iter(self, **kwargs):
        return self.get_eval_iter_from_path(self.eval_path, self.eval_batch_size, domains=self.eval_domains, **kwargs)

    def _gen_extra_metadata(self) -> None:
        self.metadata.source_dict = self._field_to_fairseq_dict(self.features[ModelInput.SEQ])
        self.metadata.target_dict = self._field_to_fairseq_dict(self.labels[ModelOutput.TOK])

    def init_metadata(self):
        """
        Since we have use_vocab=False on all fields, we don't need to use our datasets to
        compute vocab or other metadata.
        """
        self.metadata.features = {}
        for name, feat in self.features.items():
            meta = feat.get_meta()
            meta.pretrained_embeds_weight = None
            self.metadata.features[name] = meta

        self.metadata.target = [field.get_meta() for field in self.labels.values()]
        if len(self.metadata.target) == 1:
            [self.metadata.target] = self.metadata.target

        self._gen_extra_metadata()

    def _to_cuda_postprocess_batch(self, batch, include_input, include_target, include_context, is_train):
        def inner(batch, is_train):
            #new_batch = BatchLike(batch, fields_to_cuda=[ModelInput.SEQ, ModelOutput.TOK])
            new_batch = BatchLike(batch, fields_to_cuda=[ModelInput.SEQ, ModelOutput.TOK, ModelOutput.NEG_TOK, ModelInput.SEQUENCE])
            return self._postprocess_batch(new_batch, include_input, include_target, include_context, is_train=is_train)

        if isinstance(batch, (textdata.Batch, BatchLike)):
            input, target, ctx = inner(batch, is_train=is_train)
        else:
            # meta batch

            # debug
            #input_ids = batch[0][1].sequence[0]
            #instance = input_ids[0][0][1]

            input, target, ctx = [], [], []
            for b_i, domain in enumerate(batch): # 2
                d0_input, d0_target, d0_ctx = inner(domain[0], is_train=True)      # support set
                d1_input, d1_target, d1_ctx = inner(domain[1], is_train=is_train)  # target set
                input.append((d0_input, d1_input))
                target.append((d0_target, d1_target))
                ctx.append((d0_ctx, d1_ctx))

        if not include_input:
            input = None
        if not include_target:
            target = None
        if not include_context:
            ctx = None

        return input, target, ctx

    def _make_teacher_forcing(self, teacher_forcing_input, teacher_forcing_lens):
        # remove final </s>, doesn't make sense to predict it particularly if it's in the middle of a sentence
        # insert </s> at the *beginning* 'cause that's how fairseq wants it for some reason.
        # teacher_foring_input.shape (128, 1, 34)
        tf = teacher_forcing_input[..., :-1].clone()
        tf[..., 0] = self.metadata.target_dict.eos_index  # this is what fairseq uses for some reason
        return tf, teacher_forcing_lens - 1

    def _train_input_from_batch(self, batch):
        seq_input = getattr(batch, ModelInput.SEQ) # seq_input (4) # (128, 5, 35), (128) n seqs, (128, 5) n words per seq, None
        #target = getattr(batch, ModelOutput.TOK) # (2) (128, 48), (128)
        #neg_target = getattr(batch, ModelOutput.NEG_TOK)
        sequence = getattr(batch, ModelInput.SEQUENCE)
        #teacher_forcing_input, teacher_forcing_lens = self._make_teacher_forcing(*target)
        #neg_tf_input, neg_tf_lens = self._make_teacher_forcing(*neg_target)
        # seq_input_emb = getattr(batch, ModelInput.SEQ_EMB)
        # [JG_support input]
        #return (
        #  sequence,
        ## flatten the seq input into the list of parameters
        #seq_input[0],  # (128, 5, 35)
        #seq_input[3],  # None
        #teacher_forcing_input,
        #seq_input[1],  # n seqs
        #seq_input[2],  # n words per seq
        #teacher_forcing_lens,   # n words per output seq

        #  *(
        #    getattr(batch, key)
        #    for key in self.features
        #    if key not in [ModelInput.SEQ]
        #  ),
        #)
        return sequence

    def _target_from_batch(self, batch):
        out_tok, out_len = getattr(batch, ModelOutput.TOK)
        # remove potential final </s> (could indicate incomplete sentence)
        return (out_tok[..., 1:].contiguous(), out_len - 1)

    def _test_input_from_batch(self, batch):
        #seq_input = getattr(batch, ModelInput.SEQ)
        sequence = getattr(batch, ModelInput.SEQUENCE)
        #return (
            # flatten the seq input into the list of parameters
            #seq_input[0],
            #seq_input[3],
            #None,  # teacher_forcing_input[0],
            #seq_input[1],  # n seqs
            #seq_input[2],  # n words per seq
            #None,  # teacher_forcing_input[1],   # n words per output seq

        #    *(
        #        getattr(batch, key)
        #        for key in self.features
        #        if key not in [ModelInput.SEQ]
        #    ),
        #)
        return sequence

    def _context_from_batch(self, batch):
        res = dict()
        if hasattr(batch, ModelOutput.TOK):
            # only happens in train/eval mode, not in predict
            target_token_lens = getattr(batch, ModelOutput.TOK)[1]
            res[DatasetFieldName.TARGET_SEQ_LENS] = target_token_lens
        res.update(**super()._context_from_batch(batch))
        return res

    def _field_to_fairseq_dict(self, field):
        fs_dict = FairseqDict()

        ll = list(range(field.meta.vocab_size))
        # stoi
        fs_dict.indices = {k: self.text_embedder.decode_id_as_token(k) for k in ll}
        # itos
        fs_dict.symbols = {v: k for k, v in fs_dict.indices.items()}
        fs_dict.bos_index = field.meta.bos_token_idx
        fs_dict.eos_index = field.meta.eos_token_idx
        fs_dict.pad_index = field.meta.pad_token_idx
        fs_dict.unk_index = field.meta.unk_token_idx

        # not in there by default
        if fs_dict.unk_index not in fs_dict.symbols:
            fs_dict.symbols[fs_dict.unk_index] = '<unk>'
        if fs_dict.pad_index not in fs_dict.symbols:
            fs_dict.symbols[fs_dict.pad_index] = '<pad>'
        if fs_dict.bos_index not in fs_dict.symbols:
            fs_dict.symbols[fs_dict.bos_index] = '<s>'
        if fs_dict.eos_index not in fs_dict.symbols:
            fs_dict.symbols[fs_dict.eos_index] = '</s>'

        fs_dict.unk_word = (fs_dict.symbols[fs_dict.unk_index],)
        fs_dict.pad_word = fs_dict.symbols[fs_dict.pad_index]
        fs_dict.bos_word = fs_dict.symbols[fs_dict.bos_index]
        fs_dict.eos_word = fs_dict.symbols[fs_dict.eos_index]

        # not provided (and we probably should not threshold on this, instead modify preproc)
        # fs_dict.count = [field.vocab.freqs[s] for s in fs_dict.symbols]
        return fs_dict


class MetaDataHandler(DialogueDataHandler):
    class Config(DialogueDataHandler.Config):
        # Support set size per task, i.e. base-learner minibatch size
        support_batch_size: int = 128 # 128
        meta_batch_size: int = 2 # 2

    @classmethod
    def from_config(cls, config: Config,
                    feature_config: ModelInputConfig,
                    target_config: ModelOutputConfig, **kwargs):

        self = super().from_config(config, feature_config, target_config, **kwargs)
        self.support_batch_size = config.support_batch_size
        self.meta_batch_size = config.meta_batch_size
        return self

    def get_train_iter_from_path(
            self, *args, **kwargs
    ) -> BatchIterator:
        kwargs["support_batch_size"] = self.support_batch_size
        kwargs["meta_batch_size"] = self.meta_batch_size
        return super().get_train_iter_from_path(*args, **kwargs)

    def get_test_iter_from_path(
            self, *args, **kwargs
    ) -> BatchIterator:
        kwargs["support_batch_size"] = self.support_batch_size
        kwargs["meta_batch_size"] = 1
        kwargs["is_predict"] = True
        kwargs["max_n_turns"] = 10  # prevent GPU memory errors
        return super().get_test_iter_from_path(*args, **kwargs)

    def get_predict_iter_from_path(
            self, *args, **kwargs
    ) -> BatchIterator:
        kwargs["support_batch_size"] = self.support_batch_size
        kwargs["meta_batch_size"] = self.meta_batch_size
        kwargs["is_predict"] = True
        kwargs["max_n_turns"] = 10  # prevent GPU memory errors
        return super().get_predict_iter_from_path(*args, **kwargs)

    def get_test_iter(self, *args, **kwargs):
        kwargs["support_batch_size"] = self.support_batch_size
        kwargs["meta_batch_size"] = 1
        kwargs["is_predict"] = True
        kwargs["max_n_turns"] = 10  # prevent GPU memory errors
        return super().get_test_iter(*args, **kwargs)

    def get_train_iter(self, *args, **kwargs):
        kwargs["support_batch_size"] = self.support_batch_size
        kwargs["meta_batch_size"] = self.meta_batch_size
        return super().get_train_iter(*args, **kwargs)

    def get_trainset(self, *args, **kwargs):
        return super().get_trainset(*args, **kwargs)

    def get_eval_iter(self, *args, **kwargs):
        kwargs["support_batch_size"] = self.support_batch_size
        kwargs["meta_batch_size"] = 1
        return super().get_eval_iter(*args, **kwargs)
