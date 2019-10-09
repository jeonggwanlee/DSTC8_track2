import logging
import numpy as np
import sys

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from itertools import chain, islice
import random

from mldc.data.schema import MetaDlgDataDialog
from pytext.config.component import Component, ComponentType, ConfigBase
from typing import Sequence
from tqdm import tqdm

from mldc.preprocessing.input_embedding import EmbedderInterface
from mldc.util import TqdmToLogger


LOG = logging.getLogger('mldc.preprocessing.featurizer')
INIT_DONE = False


def no_progress():
    """
    determines if we want to see progressbars in the output

    do not show progress bars if:
    - if we aren't on an interactive terminal or
    - the user wants verbose logging
    """
    return False
    return not sys.stdout.isatty()


class OutputRecord:
    turns: Sequence[str]
    token_ids: np.array
    domain_id: str
    task_id: str
    dlg_id: str
    neg_turns : Sequence[str]
    neg_token_ids : np.array
    # lm_labels : later
    # mc_token_ids : later
    #



class TokenIdFeaturizer(Component):
    __COMPONENT_TYPE__ = ComponentType.FEATURIZER
    __EXPANSIBLE__ = True
    """
    This "featurizes" the whole dataset. Because the data is quite big and needs to fit in RAM,
    the featurizer only tokenizes the text converts tokens to integer IDs, thereby compressing
    it from the original text format.
  
    The action can be performed in parallel by a pool of workers.
    """

    class Config(Component.Config):
        pass

    @classmethod
    def from_config(cls, config: Config, feature_config: ConfigBase, text_embedder_config: EmbedderInterface.Config):
        return cls(config, feature_config, text_embedder_config)

    def __init__(self, config: Config, feature_config, text_embedder_config):
        self.text_embedder = EmbedderInterface.from_config(text_embedder_config)

    def featurize_batch(self, input_record_list: Sequence[MetaDlgDataDialog]) -> Sequence[OutputRecord]:
        return [self.featurize(record) for record in input_record_list]

    def featurize(self, input_record: MetaDlgDataDialog):
        ret = OutputRecord()
        # ORIGINAL
        #ret.token_ids = [
        #  np.array([self.text_embedder.bos_idx] + self.text_embedder.encode_text_as_ids(turn).tolist() + [self.text_embedder.eos_idx])
        #  for turn in input_record.turns]
        # input_record.turns
        # <class 'list'>: ['Hello how may I help you?', 'I need to find out my late fees', 'What is you account number?', '<num>',
        # 'You have $ <num> i late fees', 'What?', 'You took out <num> books last year and never returned them',
        # 'I need to get that debt erased please. I will bring the books back tomorrow', "I'm sorry I cannot erase your late fees.",
        # "But I don't have <num> dollars", 'If you have bee working on blisdomains since <weekday>. You probably do',
        # 'That statement is true lol', 'Are you paying?', 'Yes you my card on file', 'Okay Thanks', 'Have a good day', 'Hav a nice day too']

        ret.task_id = input_record.task_id
        ret.domain_id = input_record.domain
        ret.dlg_id = input_record.id
        # [JG_add]
        #ret.turns = input_record.turns
        ret.turns = []
        ret.token_ids = []
        for t_idx, turn in enumerate(input_record.turns):
            if t_idx % 2 == 0:
                token_id = np.array([self.text_embedder.sys_idx] + self.text_embedder.encode_text_as_ids(turn).tolist() + [self.text_embedder.eos_idx])
                ret.turns.append(self.text_embedder.sys_token + turn + self.text_embedder.eos_token)
            else:
                token_id = np.array([self.text_embedder.usr_idx] + self.text_embedder.encode_text_as_ids(turn).tolist() + [self.text_embedder.eos_idx])
                ret.turns.append(self.text_embedder.usr_token + turn + self.text_embedder.eos_token)
            ret.token_ids.append(token_id)

        ret.neg_turns = []
        ret.neg_token_ids = []
        for t_idx, turn in enumerate(input_record.turns):
            if t_idx % 2 == 0: # <sys>
                neg_idx = random.choice(range(0, len(input_record.turns), 2))
            else:
                neg_idx = random.choice(range(1, len(input_record.turns), 2))
            while neg_idx == t_idx:
                if t_idx % 2 == 0:  # <sys>
                    neg_idx = random.choice(range(0, len(input_record.turns), 2))
                else:
                    neg_idx = random.choice(range(1, len(input_record.turns), 2))
            neg_turn = input_record.turns[neg_idx]
            if t_idx % 2 == 0:
                neg_token_id = np.array([self.text_embedder.sys_idx] + self.text_embedder.encode_text_as_ids(neg_turn).tolist() + [self.text_embedder.eos_idx])
                ret.neg_turns.append(self.text_embedder.sys_token + neg_turn + self.text_embedder.eos_token)
            else:
                neg_token_id = np.array([self.text_embedder.usr_idx] + self.text_embedder.encode_text_as_ids(neg_turn).tolist() + [self.text_embedder.eos_idx])
                ret.neg_turns.append(self.text_embedder.usr_token + neg_turn + self.text_embedder.eos_token)
            ret.neg_token_ids.append(neg_token_id)

        return ret

    @classmethod
    def _featurize_worker(
            cls,
            config,
            text_embedder_cfg,
            feature_config,
            batch: Sequence[MetaDlgDataDialog],
    ) -> Sequence[OutputRecord]:
        # init/initargs isn't supported in python 3.6 yet, so we emulate it here
        global FEATURIZER, INIT_DONE
        if not INIT_DONE:
            FEATURIZER = cls(config, text_embedder_config=text_embedder_cfg, feature_config=feature_config)
            INIT_DONE = True
        if not len(batch):
            return []
        return FEATURIZER.featurize_batch(batch)

    @classmethod
    def parallel_featurize_batch(
            cls,
            batch: Sequence[MetaDlgDataDialog],
            max_workers=4,
            chunksize: int = 1000,
            text_embedder_cfg: EmbedderInterface.Config = None,
            feature_config=None
    ) -> Sequence[OutputRecord]:
        # tokenizer models are relatively small so load separately in each process
        config = TokenIdFeaturizer.Config()

        # function to split the input iterator into smaller ones, one for each process
        def split_iterator(iterator, chunksize=chunksize):
            iterator = iter(iterator)
            while True:
                chunk = tuple(islice(iterator, chunksize))
                if not chunk:
                    return
                yield chunk

        worker = partial(cls._featurize_worker, config, text_embedder_cfg, feature_config)

        LOG.debug("featurizing data with %d workers", max_workers)
        tqdm_out = TqdmToLogger(LOG, level=logging.INFO)
        if max_workers > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                featurized_lol, futures = [], []
                with tqdm(unit=" chunks", desc="Loading dialogues in chunks of size %d" % chunksize, disable=no_progress(),
                          file=tqdm_out, mininterval=30) as bar:
                    chunks = iter(split_iterator(batch)) # chunks : generator
                    for chunk in chunks: # chunk (tuple 1000) chunk[0] MetaDlgDataDialog id='f3b7d8c7' turns=['Hello how may I help you?', 'Hi, I was wondering if you could answer a questi… domain='POLICY_BOT' task_id='5ff19ea8' user_id='af067699' bot_id='c05f0462'
                        bar.update(1)
                        futures += executor.submit(worker, chunk),
                for future in tqdm(futures, unit=" chunks", desc="Processing dialogues in chunks of size %d" % chunksize,
                                   disable=no_progress(), file=tqdm_out, mininterval=30):
                    featurized_lol.append(future.result())
                    # temp = future.result()
                    # temp
        else:
            featurized_lol = [worker(b) for b in split_iterator(batch)]
            # clear the INIT flag for next time this is called
            global INIT_DONE
            INIT_DONE = False

        # merge the results back into a single iterator
        return chain(*featurized_lol)
