def fmt(template: str, **kwargs) -> str:
    return template.format_map(kwargs)
