from concurrent.futures import ThreadPoolExecutor
from core.executors import HEAVY_EXECUTOR, LIGHT_EXECUTOR

def test_heavy_executor_is_thread_pool():
    assert isinstance(HEAVY_EXECUTOR, ThreadPoolExecutor)

def test_light_executor_is_thread_pool():
    assert isinstance(LIGHT_EXECUTOR, ThreadPoolExecutor)

def test_executors_are_different_objects():
    assert HEAVY_EXECUTOR is not LIGHT_EXECUTOR
