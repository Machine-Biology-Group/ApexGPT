import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import datasets
import evaluate
import torch
import numpy as np
import esm
from datasets import load_dataset,DatasetDict,concatenate_datasets
from torch.utils.data import DataLoader
import transformers
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    default_data_collator,
    is_torch_xla_available,
    set_seed,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from model import ProGenForTrainerlLM, ProGenConfig,ProGenModel
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
#check_min_version("4.41.0.dev0")
require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default='./checkpoints_base/',
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default='/Users/changge/Desktop/Transformer-Visualization/AMP_GPT/Filter_length_Remove_InHouse_Database_Antibacterial_unique_standard.fasta', metadata={"help": "The input training data file (a text file)."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."
        },
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    #we design the parallel
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )
    raw_datasets = DatasetDict({
            "train": raw_datasets["train"].map(lambda x: {"text": x["text"].replace("\n", "")}),
            "validation": raw_datasets["validation"].map(lambda x: {"text": x["text"].replace("\n", "")}),
            })
    raw_datasets_reverse = DatasetDict({
    "train": raw_datasets["train"].map(lambda x: {"text": x["text"][::-1]}),
    "validation": raw_datasets["validation"].map(lambda x: {"text": x["text"][::-1]}),
    })
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = ProGenConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer=PreTrainedTokenizerFast(tokenizer_file=model_args.tokenizer_name, pad_token="<|pad|>",bos_token='<|bos|>',eos_token='<|eos|>',**tokenizer_kwargs) 
        #tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = 0
    print(tokenizer.pad_token)
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = ProGenForTrainerlLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            ignore_mismatched_sizes=True,
        )
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    '''
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    '''
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]
    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    if data_args.line_by_line:
        padding = "max_length" if data_args.pad_to_max_length else False
        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line
                for line in examples[text_column_name]
                if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length-2,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=[text_column_name],
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset line_by_line",
                )
                tokenized_datasets_reverse = raw_datasets_reverse.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=[text_column_name],
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset line_by_line",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=[text_column_name],
                )
                tokenized_datasets_reverse = raw_datasets_reverse.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=[text_column_name],
                )
        # Define a function to add BOS and EOS tokens to a single example
        def add_special_tokens(example, bos, eos):
            # Add BOS and EOS tokens to input_ids
            example['input_ids'] = [bos] + example['input_ids'] + [eos]
            # Update attention_mask to match the new length of input_ids
            example['attention_mask'] = [1] + example['attention_mask'] + [1]
    
            # Update token_type_ids to match the new length of input_ids (if necessary)
            if 'token_type_ids' in example:
                example['token_type_ids'] = [0] + example['token_type_ids'] + [0]
            #example['labels'] = example['input_ids'].copy() 
            return example

        # Apply the function to each dataset split, passing the bos and eos token ids
        bos_token_id = 3
        eos_token_id = 4
        tokenized_datasets = tokenized_datasets.map(lambda x: add_special_tokens(x, bos_token_id, eos_token_id))
        tokenized_datasets_reverse = tokenized_datasets_reverse.map(lambda x: add_special_tokens(x, eos_token_id, bos_token_id))
        tokenized_datasets =DatasetDict({
            'train': concatenate_datasets([tokenized_datasets['train'], tokenized_datasets_reverse['train']]),
            'validation': concatenate_datasets([tokenized_datasets['validation'], tokenized_datasets_reverse['validation']])
            })
        #tokenized_datasets['train'] =tokenized_datasets['train'].shuffle(seed=42)
    else:
        def tokenize_function(examples):
            with CaptureLogger(tok_logger) as cl:
                output = tokenizer(examples[text_column_name])
            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                    " before being passed to the model."
                )
            return output

        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                )
        if hasattr(config, "max_position_embeddings"):
            max_pos_embeddings = config.max_position_embeddings
        else:
            # Define a default value if the attribute is missing in the config.
            max_pos_embeddings = 1024

        if data_args.block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > max_pos_embeddings:
                logger.warning(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                    f"Using block_size={min(1024, max_pos_embeddings)} instead. You can change that default value by passing --block_size xxx."
                )
                if max_pos_embeddings > 0:
                    block_size = min(1024, max_pos_embeddings)
                else:
                    block_size = 1024
        else:
            if data_args.block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model "
                    f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                )
            block_size = min(data_args.block_size, tokenizer.model_max_length)

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/process#map

        with training_args.main_process_first(desc="grouping texts together"):
            if not data_args.streaming:
                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )
            else:
                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                )
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
    
    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)
    def truncate(sample, terminals):
        pos = []
        for terminal in terminals:
            find_pos = sample.find(terminal, 1)
            if find_pos != -1:
                pos.append(find_pos)
        if len(pos) > 0:
            return sample[:(min(pos))]
        else:
           return sample
    def truncate_begin(sample, terminals):
        pos = []
        for terminal in terminals:
            find_pos = sample.find(terminal, 0)
            if find_pos != -1:
                if terminal == '<|eos|>' or terminal == '<|bos|>' or terminal == '<|pad|>':
                    pos.append(find_pos+7)
                else:
                    pos.append(find_pos+1)
        if len(pos) > 0:
            return sample[(max(pos)):]
        else:
            return sample
    def decode_process(decode_seq_r,decode_seq):
        terminals=['<|bos|>', '<|eos|>','1','2','<|pad|>']
        decode_seq_P = [truncate(completion, terminals) for completion in decode_seq]
        decode_seq_R = [truncate(completion, terminals) for completion in decode_seq_r]
        decode_seq_P_F = []
        decode_seq_R_F = []
        for i, seq in enumerate(decode_seq_R):
            if seq[0]=='1':
                decode_seq_R_F.append(seq[1:])
                decode_seq_P_C = truncate_begin(decode_seq_P[i],terminals)
                decode_seq_P_F.append(decode_seq_P_C)
            elif seq[0]=='2':
                decode_seq_R_F.append(seq[1:][::-1])
                decode_seq_P_C = truncate_begin(decode_seq_P[i],terminals)
                decode_seq_P_F.append(decode_seq_P_C[::-1])
        return decode_seq_R_F,decode_seq_P_F


    def representation_extract(seq_list,model_E,batch_converter):
        model_E.eval()
        input_ESM = [(f'peptide_{i}',seq) for i, seq in enumerate(seq_list)]
        batch_labels, batch_strs, batch_tokens = batch_converter(input_ESM)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(esm_device)
        with torch.no_grad():
            results = model_E(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        sequence_representations = token_representations[:,0,:]
        return sequence_representations

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8)
    train_dataset = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, collate_fn=data_collator)
    eval_dataset = DataLoader(eval_dataset, batch_size=training_args.per_device_train_batch_size, collate_fn=data_collator)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    esm_device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else device)
    model.deparallelize(device)
    model.to(device,non_blocking=True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=training_args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps =training_args.num_train_epochs*len(train_dataset)*0.05 ,
                num_training_steps =training_args.num_train_epochs*len(train_dataset))
    record_vector = {}
    model_E, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model_E = model_E.to(esm_device)
    record = 0
    beta = 0.9
    best_loss_eval = 10
    path_eval_mode = training_args.output_dir + '/best_mode_eval'
    for epoch in range(training_args.num_train_epochs):
        path_model_epoch = training_args.output_dir + f'/epoch{epoch+1}'
        model.train()
        total_loss = 0
        mse_total_loss = 0
        for i,batch in enumerate(train_dataset):
            batch = data_collator(batch)
            input_ids = batch['input_ids'].to(model.device)
            #attention_mask = batch['attention_mask'].to(model.device)
            optimizer.zero_grad()
            with torch.cuda.device(0):
                outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            labels = np.argmax(outputs.logits.detach().cpu().numpy(), axis=-1)
            decode_seq = tokenizer.batch_decode(labels)
            decode_seq_r = tokenizer.batch_decode(batch['input_ids'])
            decode_seq_R_F,decode_seq_P_F = decode_process(decode_seq_r,decode_seq)
            r=representation_extract(decode_seq_R_F,model_E,batch_converter)
            r1 = representation_extract(decode_seq_P_F,model_E,batch_converter)
            if i ==0:
                for j, seq in enumerate(decode_seq_R_F):
                    if seq not in record_vector:
                        record_vector[seq] = [r[j].tolist()]
                        record_vector[seq].append(r1[j].tolist())
                    else:
                        record_vector[seq].append(r1[j].tolist())
            mse_loss = torch.nn.functional.mse_loss(r, r1)
            mse_loss = mse_loss.detach().to(model.device)
            loss=outputs.loss*(mse_loss-record)*10 + outputs.loss
            record = beta*record + (1-beta)*mse_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            mse_total_loss += mse_loss*10
        train_loss = total_loss / len(train_dataset)
        train_mse_loss = mse_total_loss /len(train_dataset)
    
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_dataset)}, MSE Loss: {train_mse_loss}')
        if (epoch + 1) % 1 ==0:
            model.eval()
            eval_loss = 0
            for i, batch in enumerate(eval_dataset):
                batch = data_collator(batch)
                input_ids = batch['input_ids'].to(model.device)
                with torch.no_grad():
                    outputs = model(input_ids, labels=input_ids)
                eval_loss += outputs.loss.item()
            print(f'Epoch {epoch+1}, Eval Loss: {eval_loss / len(eval_dataset)}, Perplexity: {math.exp(eval_loss/len(eval_dataset))}')
            if (eval_loss / len(eval_dataset)) <best_loss_eval:
                best_loss_eval = eval_loss / len(eval_dataset)
                model.save_pretrained(path_eval_mode)
        model.save_pretrained(path_model_epoch)

    import pandas as pd
    
    with open('output_vector_MDH_10.csv','w') as w:
        df = pd.DataFrame(record_vector)
        df.T.to_csv(w)
    model.save_pretrained(training_args.output_dir)
            


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
