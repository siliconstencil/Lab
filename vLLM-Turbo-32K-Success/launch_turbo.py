import sys
import os
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

sys.path.insert(0, os.path.expanduser('~/ArGeLab/vLLM-Turbo/turboquant'))

def main():
    import vllm.v1.core.kv_cache_utils
    vllm.v1.core.kv_cache_utils._check_enough_kv_cache_memory = lambda *a, **k: None

    from vllm import LLM, SamplingParams
    llm = LLM(
        model='/home/hank/ArGeLab/models/llama31-8b-q4.gguf',
        gpu_memory_utilization=0.85,
        max_model_len=32768,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        max_num_seqs=1
    )

    engine = llm.llm_engine
    core = getattr(engine, "engine_core", engine)
    inner = getattr(core, "engine_core", core)
    executor = inner.model_executor

    def _install(worker):
        from turboquant.vllm_attn_backend import install_turboquant_hooks, MODE_ACTIVE
        return len(install_turboquant_hooks(worker.model_runner, key_bits=3, value_bits=2, buffer_size=128, mode=MODE_ACTIVE))
    
    executor.collective_rpc(_install)

    params = SamplingParams(temperature=0.7, max_tokens=100)
    output = llm.generate(['Explain KV cache compression in LLM instruction tuning in two sentences:'], params)
    print('\\n\\n*** TurboQuant + vLLM 32K Hazir! ***\\n')
    print(output[0].outputs[0].text)

if __name__ == '__main__':
    main()
