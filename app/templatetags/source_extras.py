from django import template

register = template.Library()


@register.filter
def source_url(value):
    if isinstance(value, dict):
        return (value.get("url") or "").strip()
    if not value:
        return ""
    text = str(value).strip()
    idx = text.lower().find("http")
    if idx != -1:
        return text[idx:].strip()
    return text


@register.filter
def source_title(value):
    if isinstance(value, dict):
        return (value.get("title") or value.get("url") or "").strip()
    if not value:
        return ""
    text = str(value).strip()
    idx = text.lower().find("http")
    if idx != -1:
        title = text[:idx].strip()
        return title.rstrip(" -–—:").strip() or text[idx:].strip()
    return text
