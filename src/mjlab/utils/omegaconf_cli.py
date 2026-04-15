"""Tyro + OmegaConf CLI helpers for repo scripts."""

from __future__ import annotations

import copy
import dataclasses
import functools
import inspect
import sys
import types
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar, get_args, get_origin, get_type_hints

from omegaconf import OmegaConf
import tyro

from mjlab.utils.config_loader import (
    apply_config_overrides,
    load_config_source,
    load_dataclass_from_dict,
)

T = TypeVar("T")


def _normalize_dotlist_args(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token.startswith("--"):
            if token in {"--help", "-h"}:
                normalized.append(token)
                idx += 1
                continue
            key = token[2:]
            if "=" in key:
                name, raw_value = key.split("=", 1)
                normalized.append(f"{name.replace('-', '_')}={raw_value}")
                idx += 1
                continue
            if idx + 1 < len(argv) and not argv[idx + 1].startswith("--"):
                normalized.append(f"{key.replace('-', '_')}={argv[idx + 1]}")
                idx += 2
            else:
                normalized.append(f"{key.replace('-', '_')}=true")
                idx += 1
            continue
        normalized.append(token)
        idx += 1
    return normalized


def _flag_name(name: str) -> str:
    return ".".join(part.replace("_", "-") for part in name.split("."))


def _normalize_tyro_args(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token in {"-h", "--help"}:
            normalized.append(token)
            idx += 1
            continue
        if token.startswith("--"):
            key = token[2:]
            if "=" in key:
                name, raw_value = key.split("=", 1)
                normalized.append(f"--{_flag_name(name)}={raw_value}")
            else:
                normalized.append(f"--{_flag_name(key)}")
            idx += 1
            continue
        if "=" in token:
            name, raw_value = token.split("=", 1)
            normalized.append(f"--{_flag_name(name)}={raw_value}")
            idx += 1
            continue
        normalized.append(token)
        idx += 1
    return normalized


def _default_dataclass_data(config_type: type[Any]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for field in dataclasses.fields(config_type):
        if field.default is not dataclasses.MISSING:
            data[field.name] = copy.deepcopy(field.default)
        elif field.default_factory is not dataclasses.MISSING:
            data[field.name] = field.default_factory()
    return data


def _annotation_name(annotation: Any) -> str:
    if annotation is inspect._empty:
        return "Any"
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _print_choice_help(
    choices: Iterable[str],
    *,
    description: str | None = None,
    prog: str | None = None,
) -> None:
    if description:
        print(description.strip())
        print()
    usage = prog or Path(sys.argv[0]).name
    print(
        f"Usage: {usage} <choice> [config_path=<path>] [config_source=<source>] "
        "[field=value] [--field value]"
    )
    print("Choices:")
    for choice in choices:
        print(f"  {choice}")
    print(f"Run `{usage} <choice> --help` for per-task config help.")


def _extract_config_source(argv: list[str]) -> str | None:
    normalized = _normalize_dotlist_args(argv)
    dotlist = [token for token in normalized if "=" in token]
    if not dotlist:
        return None
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cli_data = OmegaConf.to_container(cli_cfg, resolve=True) or {}
    source = cli_data.get("config_source") or cli_data.get("config_path")
    return source if isinstance(source, str) and source else None


@functools.lru_cache(maxsize=None)
def _build_cli_type(config_type: type[Any]) -> type[Any]:
    resolved_hints = get_type_hints(config_type)
    fields: list[tuple[Any, ...]] = [
        ("config_path", str | None, dataclasses.field(default=None)),
        ("config_source", str | None, dataclasses.field(default=None)),
    ]
    for field in dataclasses.fields(config_type):
        annotation = resolved_hints.get(field.name, field.type)
        if field.default_factory is not dataclasses.MISSING:
            fields.append(
                (
                    field.name,
                    annotation,
                    dataclasses.field(default_factory=field.default_factory),
                )
            )
        elif field.default is not dataclasses.MISSING:
            fields.append(
                (
                    field.name,
                    annotation,
                    dataclasses.field(default=copy.deepcopy(field.default)),
                )
            )
        else:
            fields.append((field.name, annotation))
    return dataclasses.make_dataclass(
        f"{config_type.__name__}Cli",
        fields,
        kw_only=True,
    )


def _build_wrapper_default(config_type: type[T], default: T) -> Any:
    wrapper_type = _build_cli_type(config_type)
    values: dict[str, Any] = {
        "config_path": None,
        "config_source": None,
    }
    for field in dataclasses.fields(config_type):
        values[field.name] = copy.deepcopy(getattr(default, field.name))
    return wrapper_type(**values)


def parse_dataclass_cli(
    config_type: type[T],
    *,
    argv: list[str] | None = None,
    default: T | None = None,
    description: str | None = None,
    prog: str | None = None,
) -> T:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    config_source = _extract_config_source(raw_args)

    if default is None:
        result = load_dataclass_from_dict(config_type, _default_dataclass_data(config_type))
    else:
        result = copy.deepcopy(default)

    if config_source is not None:
        apply_config_overrides(result, load_config_source(config_source), recursive=True)

    wrapper_default = _build_wrapper_default(config_type, result)
    wrapper_type = _build_cli_type(config_type)
    parsed = tyro.cli(
        wrapper_type,
        args=_normalize_tyro_args(raw_args),
        default=wrapper_default,
    )
    parsed_data = dataclasses.asdict(parsed)
    parsed_data.pop("config_path", None)
    parsed_data.pop("config_source", None)
    return load_dataclass_from_dict(config_type, parsed_data)


def parse_choice_and_dataclass(
    choices: Iterable[str],
    config_type: type[T],
    *,
    argv: list[str] | None = None,
    default_factory: Callable[[str], T] | None = None,
    description: str | None = None,
    prog: str | None = None,
) -> tuple[str, T]:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    normalized = _normalize_dotlist_args(raw_args)
    available = list(choices)

    if not normalized or normalized[0] in {"-h", "--help"}:
        _print_choice_help(available, description=description, prog=prog)
        raise SystemExit(0)

    choice = normalized[0]
    if choice not in available:
        raise ValueError(f"Unknown choice '{choice}'. Expected one of: {available}")

    default = default_factory(choice) if default_factory is not None else None
    cfg = parse_dataclass_cli(
        config_type,
        argv=normalized[1:],
        default=default,
        description=description,
        prog=f"{prog or Path(sys.argv[0]).name} {choice}",
    )
    return choice, cfg


def _coerce_scalar(value: Any, annotation: Any) -> Any:
    if annotation is inspect._empty or annotation is Any:
        return value
    if annotation is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        raise ValueError(f"Cannot coerce {value!r} to bool")
    if annotation is Path:
        return Path(value)
    if annotation in {int, float, str}:
        return annotation(value)
    return value


def _coerce_value(value: Any, annotation: Any) -> Any:
    if value is None:
        return None

    origin = get_origin(annotation)
    if origin is None and isinstance(annotation, types.UnionType):
        origin = types.UnionType

    if origin is not None:
        if str(origin) == "typing.Union" or origin is types.UnionType:
            for arg in (arg for arg in get_args(annotation) if arg is not type(None)):
                try:
                    return _coerce_value(value, arg)
                except Exception:
                    continue
            return value
        if str(origin).endswith("Literal"):
            return value
        if origin is list:
            elem_type = get_args(annotation)[0] if get_args(annotation) else Any
            return [_coerce_value(item, elem_type) for item in value]
        if origin is tuple:
            args = get_args(annotation)
            if len(args) == 2 and args[1] is Ellipsis:
                return tuple(_coerce_value(item, args[0]) for item in value)
            return tuple(
                _coerce_value(item, arg)
                for item, arg in zip(value, args, strict=False)
            )

    return _coerce_scalar(value, annotation)


def _print_function_help(func: Callable[..., Any], description: str | None = None) -> None:
    if description:
        print(description.strip())
        print()
    sig = inspect.signature(func)
    usage = f"Usage: {Path(sys.argv[0]).name}"
    for param in sig.parameters.values():
        if param.default is inspect._empty:
            usage += f" <{param.name}>"
        else:
            usage += f" [{param.name}=...]"
    print(usage)
    print("Parameters:")
    for param in sig.parameters.values():
        default = ""
        if param.default is not inspect._empty:
            default = f" = {param.default!r}"
        print(f"  {param.name}: {_annotation_name(param.annotation)}{default}")


def run_cli_function(
    func: Callable[..., Any],
    *,
    argv: list[str] | None = None,
    description: str | None = None,
) -> Any:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    normalized = _normalize_dotlist_args(raw_args)
    if any(token in {"-h", "--help"} for token in normalized):
        _print_function_help(func, description=description)
        raise SystemExit(0)

    positional = [token for token in normalized if "=" not in token]
    named_tokens = [token for token in normalized if "=" in token]
    named_cfg = (
        OmegaConf.from_dotlist(named_tokens) if named_tokens else OmegaConf.create({})
    )
    named_values = OmegaConf.to_container(named_cfg, resolve=True) or {}

    sig = inspect.signature(func)
    kwargs: dict[str, Any] = {}
    positional_index = 0
    for param in sig.parameters.values():
        if param.name in named_values:
            kwargs[param.name] = _coerce_value(named_values[param.name], param.annotation)
            continue
        if positional_index < len(positional):
            kwargs[param.name] = _coerce_value(
                positional[positional_index], param.annotation
            )
            positional_index += 1
            continue
        if param.default is not inspect._empty:
            kwargs[param.name] = param.default
            continue
        raise ValueError(f"Missing required argument: {param.name}")

    if positional_index != len(positional):
        raise ValueError(
            f"Unexpected positional arguments: {positional[positional_index:]}"
        )
    return func(**kwargs)
