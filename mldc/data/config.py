from typing import List
from pytext.config.field_config import WordFeatConfig
from pytext.config.field_config import ConfigBase
from pytext.config.module_config import ModuleConfig


class ModelInput:
  SEQ = "seq_word_feat"
  SEQ_EMB = "pretrained_model_embedding"
  TASK_ID = "task_id"
  DOMAIN_ID = "domain_id"
  DLG_ID = "dlg_id"
  DLG_LEN = "dlg_len"
  NEG_SEQ = "neg_seq_word_feat"
  ITTI = "input_token_type_ids"
  TTTI = "target_token_type_ids"
  NTTTI = "neg_tar_token_type_ids"
  HISTORY = "input_history"
  SEQUENCE = "sequence"

class ModelOutput:
  TOK = "out_tokens"
  NEG_TOK = "neg_out_tokens"


class ModelInputConfig(ModuleConfig):
  seq_word_feat: WordFeatConfig = WordFeatConfig(
    min_freq=1,
  )
  # pretrained_model_embedding: Optional[PretrainedModelEmbeddingConfig] = None


class ModelOutputConfig(ConfigBase):
  _name = ModelOutput.TOK
  _neg = ModelOutput.NEG_TOK
  export_output_names: List[str] = ['word_scores']
  min_freq: int = 1
