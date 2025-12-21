import time
import logging
import random
from typing import Callable, Iterable, Optional, Any, Type

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests

from quant_system.utils.logger import log

LOGGER = logging.getLogger("quant_system.api_retry")
LOGGER.setLevel(logging.INFO)

if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[RETRY] %(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(handler)


def linear_backoff(base: float, attempt: int) -> float:
    return base * attempt


def exponential_backoff(base: float, attempt: int) -> float:
    return base * (2 ** attempt)


def exponential_jitter(base: float, attempt: int) -> float:
    delay = base * (2 ** attempt)
    jitter = random.uniform(0, delay * 0.1)
    return delay + jitter


def retry(
    func: Callable,
    *,
    max_attempts: int = 8,
    exceptions: Iterable[Type[BaseException]] = (Exception,),
    backoff_strategy: Callable[[float, int], float] = exponential_backoff,
    base_delay: float = 1.0,
    retry_callback: Optional[Callable[[int, float, Exception], None]] = None,
) -> Callable:
    """
    Retry wrapper for callables with configurable backoff.
    """
    def wrapper(*args, **kwargs) -> Any:
        attempt = 0
        while True:
            attempt += 1
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if attempt > max_attempts:
                    log(f"Maximum attempts reached. Raising last exception: {e}")
                    raise
                sleep_sec = backoff_strategy(base_delay, attempt)
                if retry_callback:
                    retry_callback(attempt, sleep_sec, e)
                else:
                    log(f"Attempt {attempt}/{max_attempts} failed: {e}; sleeping {sleep_sec:.2f}s")
                time.sleep(sleep_sec)
    return wrapper


class RetrySession:
    """
    requests.Session with HTTPAdapter retry configuration.
    """
    def __init__(self, retries: int = 5, backoff_factor: float = 0.5, status_forcelist=None):
        status_forcelist = status_forcelist or [429, 500, 502, 503, 504]

        self.session = requests.Session()
        retry_cfg = Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=frozenset(["GET", "POST"])
        )
        adapter = HTTPAdapter(max_retries=retry_cfg)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get(self, url: str, params=None, **kwargs):
        return self.session.get(url, params=params, **kwargs)

    def post(self, url: str, data=None, json=None, **kwargs):
        return self.session.post(url, data=data, json=json, **kwargs)
