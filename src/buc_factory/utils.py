from typing import Any


def fmt(template: str, **kwargs: Any) -> str:
    for key, value in kwargs.items():
        template = template.replace(f"{{{key}}}", str(value))
    return template
