import os

from dotenv import load_dotenv


def resolve_project_root(file_path: str, *, parents: int) -> str:
    current = os.path.dirname(os.path.abspath(file_path))
    for _ in range(max(parents, 0)):
        current = os.path.dirname(current)
    return current


def load_project_dotenv(file_path: str, *, parents: int) -> str:
    project_root = resolve_project_root(file_path, parents=parents)
    dotenv_path = os.path.join(project_root, ".env")
    # In Docker Compose, container env vars are snapshotted when the container is
    # created. Overriding from the bind-mounted project .env keeps restarts in
    # sync with the latest on-disk config.
    load_dotenv(dotenv_path=dotenv_path, override=True, encoding="utf-8")
    return dotenv_path


def _quote_env_value(value: str, *, quote_mode: str = "always") -> str:
    if quote_mode not in {"always", "auto", "never"}:
        raise ValueError(f"Unknown quote_mode: {quote_mode}")

    string_value = str(value)
    should_quote = quote_mode == "always" or (
        quote_mode == "auto" and not string_value.isalnum()
    )

    if not should_quote:
        return string_value

    return "'{}'".format(string_value.replace("'", "\\'"))


def persist_env_updates(
    env_path: str,
    updates: dict[str, str],
    *,
    quote_mode: str = "always",
    encoding: str = "utf-8",
) -> None:
    os.makedirs(os.path.dirname(env_path), exist_ok=True)

    try:
        with open(env_path, "r", encoding=encoding) as source:
            content = source.read()
    except FileNotFoundError:
        content = ""

    newline = "\r\n" if "\r\n" in content else "\n"
    lines = content.splitlines(keepends=True)
    rendered_updates = {
        key: f"{key}={_quote_env_value(value, quote_mode=quote_mode)}{newline}"
        for key, value in updates.items()
    }

    result_lines: list[str] = []
    seen_keys: set[str] = set()

    for line in lines:
        replaced = False
        for key, rendered_line in rendered_updates.items():
            if line.startswith(f"{key}="):
                result_lines.append(rendered_line)
                seen_keys.add(key)
                replaced = True
                break
        if not replaced:
            result_lines.append(line)

    if result_lines and not result_lines[-1].endswith(("\n", "\r")):
        result_lines[-1] = result_lines[-1] + newline

    for key, rendered_line in rendered_updates.items():
        if key in seen_keys:
            continue
        if result_lines and not result_lines[-1].endswith(("\n", "\r")):
            result_lines[-1] = result_lines[-1] + newline
        result_lines.append(rendered_line)

    with open(env_path, "w", encoding=encoding) as target:
        target.write("".join(result_lines))
