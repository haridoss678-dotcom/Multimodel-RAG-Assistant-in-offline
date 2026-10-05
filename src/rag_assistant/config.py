import logging
import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# pydantic-settings reads .env only to populate Settings *fields* below, all of
# which are RAG_-prefixed - it does not export anything else into the process
# environment. Third-party libraries read their own variables straight from
# os.environ, so without this they never see .env at all. Confirmed the hard
# way: HF_TOKEN sat correctly in .env while os.environ.get("HF_TOKEN") returned
# None, which would have made diarization (ingestion/diarization.py) skip
# silently - the user does everything right and still gets no speaker labels.
# Does not override variables already set in the real environment, so an
# explicitly exported value still wins.
load_dotenv()

# Must be set before any library that downloads HF models is imported. Those
# imports are all lazy (inside functions), so setting it here - at the first
# import of the package's own config - is early enough. CLAUDE.md §7 item 2.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
# Since symlinks are deliberately disabled above, silence the resulting
# per-download warning noise rather than seeing it on every model load.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


class _AnonymousHubWarningFilter(logging.Filter):
    """Drops huggingface_hub's "you are sending unauthenticated requests"
    advisory, and nothing else.

    It is advisory, not an error: without an HF_TOKEN, Hub API calls are
    anonymous and rate-limited per IP. Every model load re-checks its cache
    against the Hub, so this reprints on every server start and every test
    run, burying real errors in the log. Setting HF_TOKEN removes the cause
    outright (and is required anyway for the gated diarization models,
    CLAUDE.md §7 item 3) - this filter only suppresses the nag when no token
    is configured, and matches on the message text so genuine Hub failures
    (401/403/429/network) still surface.
    """

    _NEEDLE = "unauthenticated requests"

    def filter(self, record: logging.LogRecord) -> bool:
        return self._NEEDLE not in record.getMessage()


if not os.environ.get("HF_TOKEN"):
    logging.getLogger("huggingface_hub.utils._http").addFilter(_AnonymousHubWarningFilter())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    llm_backend: str = "local"
    llm_model: str = "qwen3:0.6b"
    # 4096, not 2048. Measured, not preference (CLAUDE.md §7 item 21): the real
    # generation prompt for an ordinary question over the 28-minute recording
    # was 1915-2042 tokens - at or over a 2048 window - and Ollama responded by
    # silently truncating the START of the prompt, taking the citation
    # instructions and evidence item [1] with it. Every such answer came back
    # with zero [n] markers and grounded=False.
    # Cost: llama-server RSS 778MB -> 1002MB, i.e. +224MB, measured via
    # `ollama ps` at each setting. Paid deliberately - at 2048 a correct
    # evidence budget is ~3,060 chars, but the pipeline's own output (5 reranked
    # chunks, parent-expanded) measured 4,960-5,758 chars, so 2048 would mean
    # routinely discarding the parent expansion that creates citation precision.
    llm_num_ctx: int = 4096
    llm_temperature: float = 0.2  # low, not 0 - see ollama_client.py for why
    ollama_host: str = "http://localhost:11434"

    # How long to wait for ONE Ollama call. 600s, not the 120s that shipped.
    #
    # Measured failure (CLAUDE.md §7 item 49): with num_ctx raised to 4096 and
    # the machine at 0.24GB free, `generate` exceeded 120s and httpx aborted the
    # turn - so a working pipeline reported a timeout. The UI already waits
    # 600s, but its patience is irrelevant when the layer beneath it quits
    # first. §23.1's own rule - "a timeout under ~5 minutes will fire during
    # normal use" - applies to this client too, not just to the UI.
    ollama_timeout_seconds: float = 600.0

    # Let the classifier think before answering. OFF by default: measured
    # 12.5s -> 4.3s with no loss of correctness on the fields that route
    # (§7 item 53). The one field thinking DID help with - time_reference - is
    # now extracted by regex from the question instead, which is the more
    # reliable signal anyway (§7 item 52). Set true to restore the old
    # behaviour if a future model needs it.
    classify_think: bool = False
    port: int = 8000

    data_dir: str = "data"
    qdrant_path: str = "storage/qdrant"
    sqlite_path: str = "storage/app.db"

    # Domain vocabulary biased during ASR decoding (§13). Comma-separated, and
    # configurable because different clients have different jargon - a hardcoded
    # constant would need a code change per client. Keep it SHORT: §7 item 20
    # measured a 10-term list at RTF 2.03 against 0.28 for four terms, because
    # every hotword token is prepended to every decoding window.
    asr_hotwords: str = "DC, GRN, E-way Bill, landed cost"

    # Batched ASR decoding. Measured on 45s of real audio: sequential RTF 1.75
    # vs batched 0.34 at size 8 - a ~5x speedup with slightly MORE words, so it
    # is not trading content for speed. The cost is coarser segmentation (11
    # segments -> 2 on that slice), which means timestamp citations land within
    # ~20s rather than ~4s. Set to 1 to fall back to sequential decoding when
    # precise timestamps matter more than wall-clock.
    asr_batch_size: int = 8

    # Maximum length of one ASR segment, in seconds.
    #
    # This is the knob that actually controls segmentation, and finding that out
    # cost a measurement (CLAUDE.md §7 item 34). asr.py passed
    # vad_parameters={"max_speech_duration_s": 30}, but faster-whisper's BATCHED
    # pipeline pops that key and overwrites it with chunk_length:
    #     if "max_speech_duration_s" in vad_parameters.keys():
    #         vad_parameters.pop("max_speech_duration_s")
    #     ... **vad_parameters, max_speech_duration_s=chunk_length
    # So once batching became the default, that setting became dead code -
    # sweeping it over 30/15/10/6 produced byte-identical output every time
    # (68 segments, 24.4s mean, 31.1s max, 3992 words). The original "free -
    # keep" measurement was taken under SEQUENTIAL decoding and silently stopped
    # applying when the decoder changed underneath it.
    asr_chunk_length: int = 30

    embedding_model_version: str = "bge-m3-v2"  # v2: dense+sparse schema, see vector_store.py
    langgraph_checkpoint_path: str = "storage/langgraph_checkpoints.db"
    # v3: markdown tables kept intact and tagged element_type="table" (which is
    # what finally makes §20.1's table_lookup/fact_lookup routing match anything
    # at all), plus chunk_index for document-order parent reassembly and
    # multi_speaker carried through from diarization.
    # v2: fenced code blocks kept intact and tagged element_type="code"
    # (CLAUDE.md §3/§20.1). Bumped per §19.4 so chunks written before each
    # change stay identifiable; the Qdrant collection name is unaffected, since
    # only embedding_model_version is part of collection identity.
    chunking_version: str = "structure-light-v3"
    parser_version: str = "pymupdf4llm-v1"
    # Written onto every source row so an answer's provenance survives a future
    # reranker change. Standard PyTorch weights, not the ONNX-int8 build (§13).
    reranker_version: str = "bge-reranker-v2-m3-torch"

    # Free the cross-encoder's 2221MB after each turn's rerank step (§29's
    # model-residency rule, which the code otherwise violates). Measured
    # tradeoff: reloading costs 12.6-18.8s against ~6s of rerank compute, so it
    # only pays if the freed RAM saves more paging than the reload costs.
    # Default OFF until an end-to-end comparison says otherwise (§7 item 48).
    release_reranker: bool = False


settings = Settings()
