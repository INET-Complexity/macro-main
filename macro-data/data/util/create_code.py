import random
import string


def create_code() -> str:
    return "".join(random.choice(string.digits) for _ in range(12))
