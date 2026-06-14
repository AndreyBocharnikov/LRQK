from mmengine.config import read_base
from opencompass.partitioners import NumWorkerPartitioner

with read_base():
    from opencompass.configs.datasets.text2json.text2json import (
        text2json_datasets,
    )
    from opencompass.configs.datasets.needlebench_v2.needlebench_v2_128k.needlebench_v2_multi_retrieval_128k import (
        needlebench_en_datasets,
    )
    from opencompass.configs.datasets.longbenchv2.longbenchv2_gen import (
        LongBenchv2_datasets
    )

from opencompass.models import HuggingFacewithChatTemplate
from opencompass.models import LlamaShadowKV, Qwen3ShadowKV


datasets = text2json_datasets # needlebench_en_datasets # LongBenchv2_datasets

models = [
    # dict(
    #    type="LRQKChatBot",
    #    path="Qwen/Qwen3-30B-A3B-Instruct-2507",
    #    batch_size=1,
    #    run_cfg=dict(num_gpus=1),
    #    lrqk_rank=32,
    #    lrqk_num_active_tokens=0.0009765625 + 0.0016,
    #    lrqk_lite_tokens=64,
    #    lrqk_init_aq_ak_method='topcol',
    #    abbr=f"qwen;rank=32;sparse_budget=0.0009765625",
    #    max_length=66 * 1024
    # ),
    # dict(
        # type="LRQKChatBot",
        # path="meta-llama/Llama-3.1-8B-Instruct",
        # batch_size=1,
        # run_cfg=dict(num_gpus=1),
        # lrqk_rank=16,
        # lrqk_num_active_tokens=0.0009765625 + 0.0016,
        # lrqk_lite_tokens=64,
        # lrqk_init_aq_ak_method='topcol',
        # abbr=f"llama;rank=16;sparse_budget=0.0009765625",
    # ),
     dict(
         type="LRQKChatBot",
         path="Qwen/Qwen3-4B-Instruct-2507",
         batch_size=1,
         run_cfg=dict(num_gpus=1),
         lrqk_rank=32,
         lrqk_num_active_tokens=0.0009765625 + 0.0016,
         lrqk_lite_tokens=64,
         lrqk_init_aq_ak_method='topcol',
         abbr=f"qwen;rank=32;sparse_budget=0.0009765625",
     ),
]

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner),
    runner=dict(
        type="WrapedLocalRunner",
        max_num_workers=8,
        retry=5,
        task=dict(type="WrapedOpenICLInferTask"),
    ),
)
