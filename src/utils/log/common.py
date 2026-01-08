import os


def is_prod() -> bool:
    return os.getenv("COZE_PROJECT_ENV") == "PROD"

def get_execute_mode() -> str:
    return "test_run" if not is_prod() else "run"
