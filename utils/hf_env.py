import os
from functools import lru_cache
from typing import Optional

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled


def configure_hf_endpoint() -> str:
    endpoint = os.environ.get('HF_ENDPOINT', '').strip()
    if endpoint:
        os.environ['HF_ENDPOINT'] = endpoint
    return endpoint


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, '').strip().lower()
    return value not in {'', '0', 'false', 'no', 'off'}


def use_local_hf_files(default: bool = False) -> bool:
    value = os.environ.get('HF_LOCAL_FILES_ONLY', '').strip().lower()
    if not value:
        return default
    return value not in {'0', 'false', 'no', 'off'}


def _offline_or_local_only() -> bool:
    return use_local_hf_files(default=False) or _env_truthy('HF_HUB_OFFLINE') or _env_truthy('TRANSFORMERS_OFFLINE')


@lru_cache(maxsize=None)
def _resolve_hf_repo_path_cached(repo_id: str, endpoint: Optional[str], allow_download: bool) -> str:
    local_kwargs = {
        'repo_id': repo_id,
        'local_files_only': True,
    }
    if endpoint:
        local_kwargs['endpoint'] = endpoint

    try:
        return snapshot_download(**local_kwargs)
    except (LocalEntryNotFoundError, OfflineModeIsEnabled) as local_exc:
        if not allow_download:
            raise RuntimeError(
                f'Local Hugging Face cache for {repo_id} is missing or incomplete while offline/local-only mode is enabled.'
            ) from local_exc

    download_kwargs = {
        'repo_id': repo_id,
        'local_files_only': False,
    }
    if endpoint:
        download_kwargs['endpoint'] = endpoint
    return snapshot_download(**download_kwargs)


def resolve_hf_repo_path(repo_id: str, *, endpoint: Optional[str] = None) -> str:
    endpoint = endpoint if endpoint is not None else configure_hf_endpoint() or None
    allow_download = not _offline_or_local_only()
    return _resolve_hf_repo_path_cached(repo_id, endpoint, allow_download)
