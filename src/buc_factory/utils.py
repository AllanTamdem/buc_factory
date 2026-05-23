def fmt(template: str, **kwargs) -> str:
    for key, value in kwargs.items():
        template = template.replace(f"{{{key}}}", str(value))
    return template
