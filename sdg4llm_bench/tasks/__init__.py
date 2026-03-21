"""Extension point for custom benchmark tasks.

This module allows researchers to register custom tasks for non-official
benchmark experiments. Custom tasks:
  - Are NOT part of the canonical SDG4LLM-Bench leaderboard
  - Require --allow-custom-config in the training script
  - Are kept in a separate registry (_CUSTOM_REGISTRY) to prevent
    accidental contamination of the locked canonical TASK_REGISTRY

Usage
-----
    from sdg4llm_bench.tasks import register_task
    from sdg4llm_bench.config import TaskSpec

    register_task(
        "my_custom_task",
        TaskSpec(
            name="my_custom_task",
            eval_tool="lm_eval",
            lm_eval_task="my_lm_eval_task",
            metric="exact_match",
            higher_is_better=True,
            num_fewshot=0,
            seed_datasets=(("my_org/my_dataset", None),),
            description="My custom task for internal evaluation.",
        ),
    )
"""

from __future__ import annotations

from sdg4llm_bench.config import TASK_REGISTRY, Task, TaskSpec

# Custom registry — separate from canonical TASK_REGISTRY to prevent
# accidental overrides of locked benchmark tasks.
_CUSTOM_REGISTRY: dict[str, TaskSpec] = {}

# Set of canonical task names (computed once to avoid repeated enum iteration)
_CANONICAL_TASK_NAMES: frozenset[str] = frozenset(t.value for t in Task)


def register_task(name: str, spec: TaskSpec) -> None:
    """Register a custom task for non-official experiments.

    Parameters
    ----------
    name:
        Task identifier string. Must not conflict with any canonical task name
        (ifeval, gsm8k, humaneval_plus, bbh).
    spec:
        TaskSpec for the custom task.

    Raises
    ------
    ValueError:
        If name conflicts with a canonical benchmark task.
    TypeError:
        If spec is not a TaskSpec instance.
    """
    if not isinstance(spec, TaskSpec):
        raise TypeError(f"spec must be a TaskSpec instance, got {type(spec).__name__!r}")

    if name in _CANONICAL_TASK_NAMES:
        raise ValueError(
            f"Cannot override locked benchmark task {name!r}. "
            f"Canonical tasks are: {sorted(_CANONICAL_TASK_NAMES)}. "
            f"Choose a different name for your custom task."
        )

    if name in _CUSTOM_REGISTRY:
        import warnings  # noqa: PLC0415

        warnings.warn(
            f"Overwriting existing custom task {name!r}.",
            UserWarning,
            stacklevel=2,
        )

    _CUSTOM_REGISTRY[name] = spec


def get_task(name: str) -> TaskSpec:
    """Retrieve a task spec by name (canonical or custom).

    Parameters
    ----------
    name:
        Task name string.

    Returns
    -------
    TaskSpec for the named task.

    Raises
    ------
    KeyError:
        If the task is not registered.
    """
    # Check canonical registry first
    try:
        task_enum = Task(name)
        return TASK_REGISTRY[task_enum]
    except ValueError:
        pass

    # Check custom registry
    if name in _CUSTOM_REGISTRY:
        return _CUSTOM_REGISTRY[name]

    raise KeyError(
        f"Task {name!r} not found. "
        f"Canonical tasks: {sorted(_CANONICAL_TASK_NAMES)}. "
        f"Registered custom tasks: {sorted(_CUSTOM_REGISTRY.keys())}."
    )


def list_tasks() -> dict[str, TaskSpec]:
    """Return all registered tasks (canonical + custom).

    Returns
    -------
    dict mapping task name str -> TaskSpec.
    """
    result = {t.value: spec for t, spec in TASK_REGISTRY.items()}
    result.update(_CUSTOM_REGISTRY)
    return result


def clear_custom_tasks() -> None:
    """Remove all registered custom tasks. Useful for test isolation."""
    _CUSTOM_REGISTRY.clear()
