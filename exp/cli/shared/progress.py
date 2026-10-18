"""One in-place progress line shared by every long-running CLI command.

An interactive terminal shows a single rich ``Live`` line naming the active stage; a countable
stage renders a progress bar with its exact completed and total counts plus a rate-based
remaining-time estimate. Each finished stage is printed once above the line as a permanent
``[x]`` row, matching the picker screens, unless the display owns a single-line section that
keeps exactly one in-place line with a continuously animated spinner and elapsed time. A
non-interactive stream receives stable newline-delimited stage updates with no cursor control,
so piped and scripted sessions stay readable line by line.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from time import monotonic

from rich.console import Console, RenderableType
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from exp.common.progress import ProgressEvent, ProgressHook

_ROW_INDENT = "  "
_BAR_WIDTH = 20


class ProgressDisplay:
    """Renders progress events for one command run on one console."""

    def __init__(self, console: Console, *, single_line: bool = False) -> None:
        """Prepare the display without drawing anything yet.

        Args:
            console: Interactive terminal, or a stream that can only be appended to.
            single_line: Keep exactly one in-place line with no permanent finished-stage rows.
        """
        self._console = console
        self._single_line = single_line
        self._current: ProgressEvent | None = None
        self._stage_started = 0.0
        self._stage_baseline = 0
        self._started_at = 0.0
        self._spinner = Spinner("dots")
        self._live = (
            Live(
                console=console,
                auto_refresh=single_line,
                refresh_per_second=4,
                transient=True,
                get_renderable=self._render_current if single_line else None,
            )
            if console.is_interactive
            else None
        )

    def start(self) -> None:
        """Begin owning the in-place line on an interactive terminal."""
        if self._single_line:
            self._started_at = monotonic()
        if self._live is not None:
            self._live.start(refresh=False)

    def stop(self) -> None:
        """Print the final stage as finished and release the in-place line."""
        if self._current is not None:
            self._print_finished(self._current)
            self._current = None
        if self._live is not None:
            self._live.stop()

    def abort(self) -> None:
        """Print the interrupted stage as unfinished and release the in-place line."""
        if self._current is not None:
            if self._live is not None:
                self._live.console.print(
                    Text(f"{_ROW_INDENT}[ ] {_label(self._current)}", style="red")
                )
            self._current = None
        if self._live is not None:
            self._live.stop()

    def observe(self, event: ProgressEvent) -> None:
        """Render one truthful stage update.

        Args:
            event: Current stage name with optional exact completed and total counts.
        """
        if self._live is None:
            self._console.print(f"{_ROW_INDENT}. {_label(event)}", markup=False, highlight=False)
            return
        if self._current is None or _key(self._current) != _key(event):
            if self._current is not None:
                self._print_finished(self._current)
            self._stage_started = monotonic()
            self._stage_baseline = event.completed or 0
        self._current = event
        if self._single_line:
            self._live.refresh()
        else:
            self._live.update(self._render(event), refresh=True)

    def _render_current(self) -> RenderableType:
        """Refresh elapsed time and animation even while no new work has completed."""
        event = self._current
        return self._render(event) if event is not None else Text("")

    def _render(self, event: ProgressEvent) -> RenderableType:
        """Compose the in-place line, with a bar and remaining-time estimate when countable.

        Args:
            event: Update being rendered on the live line.

        Returns:
            The current stage, with continuous animation and elapsed time in single-line mode.
        """
        if event.completed is None or event.total is None or event.total == 0:
            if not self._single_line:
                return Text(f"{_ROW_INDENT}> {_label(event)}", style="cyan")
            line = Text(_label(event), style="cyan")
        else:
            filled = min(_BAR_WIDTH, _BAR_WIDTH * event.completed // event.total)
            prefix = "" if self._single_line else f"{_ROW_INDENT}> "
            line = Text(f"{prefix}{_name(event)} ", style="cyan")
            line.append("\u2501" * filled, style="cyan")
            line.append("\u2501" * (_BAR_WIDTH - filled), style="dim")
            line.append(f" {event.completed}/{event.total}", style="cyan")
            remaining = self._estimated_remaining(event)
            if remaining is not None:
                line.append(f" eta {remaining}", style="dim")
        if self._single_line:
            elapsed = max(0, int(monotonic() - self._started_at))
            duration = _duration(elapsed) if elapsed else "0s"
            line.append(f" · {duration} elapsed", style="dim")
            self._spinner.update(text=line)
            return self._spinner
        return line

    def _estimated_remaining(self, event: ProgressEvent) -> str | None:
        """Estimate the stage's remaining wall-clock time from its observed rate.

        Args:
            event: Countable update carrying the current exact counts.

        Returns:
            Compact remaining-time text, or ``None`` before any rate is observable.
        """
        assert event.completed is not None and event.total is not None
        done = event.completed - self._stage_baseline
        elapsed = monotonic() - self._stage_started
        remaining = event.total - event.completed
        if done <= 0 or elapsed <= 0.0 or remaining <= 0:
            return None
        return _duration(remaining * elapsed / done)

    def _print_finished(self, event: ProgressEvent) -> None:
        """Print one permanent finished-stage row above the in-place line.

        Args:
            event: Last update observed for the stage that just finished.
        """
        if self._live is None or self._single_line:
            return
        self._live.console.print(Text(f"{_ROW_INDENT}[x] {_label(event)}", style="dim"))


@contextmanager
def progress_display(console: Console, *, single_line: bool = False) -> Iterator[ProgressHook]:
    """Own one progress region for the duration of a long-running command section.

    Args:
        console: Terminal, or non-terminal stream, receiving the updates.
        single_line: Keep the whole section on one in-place line without finished-stage rows.

    Yields:
        The hook the command hands to its long-running services.
    """
    display = ProgressDisplay(console, single_line=single_line)
    display.start()
    try:
        yield display.observe
    except BaseException:
        display.abort()
        raise
    display.stop()


def qualified(hook: ProgressHook | None, detail: str) -> ProgressHook | None:
    """Re-emit a service's events under one distinguishing qualifier.

    Args:
        hook: Downstream observer, or ``None`` when nobody is watching.
        detail: Short qualifier separating repeated stages, such as an index name.

    Returns:
        A forwarding hook carrying the qualifier, or ``None`` when the observer is absent.
    """
    if hook is None:
        return None

    def forward(event: ProgressEvent) -> None:
        """Forward one event with the owning command's qualifier attached."""
        hook(
            ProgressEvent(
                stage=event.stage,
                completed=event.completed,
                total=event.total,
                detail=event.detail or detail,
            )
        )

    return forward


@dataclass(frozen=True)
class Phase:
    """One phase of a multi-phase command, as the single progress line shows it.

    Attributes:
        key: Identity the command uses to report into this phase.
        label: Short status prefix, such as ``Learning``.
        weight: Share of the overall bar this phase fills.
        driver: Stage whose exact counts measure the phase's completion; other stages only
            update the status text.
        unit: Plural noun for the driver counts, such as ``traces``.
    """

    key: str
    label: str
    weight: float
    driver: str | None = None
    unit: str = ""


class PipelineDisplay:
    """One in-place line for a whole multi-phase command: an overall bar, a status, elapsed.

    An interactive terminal sees a single line that never grows: the bar fills across every
    phase by weight, and the status names the phase, its exact counts and the latest stage of
    the work inside it. A non-interactive stream receives one line when each phase starts and
    nothing else, so logs stay short. Events may arrive from worker threads.
    """

    def __init__(self, console: Console, phases: Sequence[Phase]) -> None:
        """Prepare the display without drawing anything yet."""
        self._console = console
        self._phases = {phase.key: phase for phase in phases}
        self._fraction = {phase.key: 0.0 for phase in phases}
        self._counts: dict[str, tuple[int, int]] = {}
        self._active: str | None = None
        self._status = ""
        self._started = 0.0
        self._lock = threading.Lock()
        self._spinner = Spinner("dots")
        self._live = (
            Live(
                console=console,
                auto_refresh=True,
                refresh_per_second=4,
                transient=True,
                get_renderable=self._render,
            )
            if console.is_interactive
            else None
        )

    def start(self) -> None:
        """Begin owning the in-place line."""
        self._started = monotonic()
        if self._live is not None:
            self._live.start(refresh=True)

    def stop(self) -> None:
        """Release the in-place line; nothing of it stays on screen."""
        if self._live is not None:
            self._live.stop()

    @contextmanager
    def paused(self) -> Iterator[None]:
        """Release the line while the command asks a question, then take it back."""
        if self._live is not None:
            self._live.stop()
        try:
            yield
        finally:
            if self._live is not None:
                self._live.start(refresh=True)

    def phase(self, key: str) -> ProgressHook:
        """The hook a command hands to the services running one phase."""
        phase = self._phases[key]

        def observe(event: ProgressEvent) -> None:
            """Fold one service event into the single line."""
            with self._lock:
                self._enter(key)
                if event.stage == phase.driver and event.total:
                    self._fraction[key] = (event.completed or 0) / event.total
                    self._counts[key] = (event.completed or 0, event.total)
                else:
                    self._status = _label(event)

        return observe

    def complete(self, key: str) -> None:
        """Mark one phase finished, whether or not it reported any work."""
        with self._lock:
            self._fraction[key] = 1.0

    def _enter(self, key: str) -> None:
        """Make ``key`` the active phase, announcing it once on a non-interactive stream."""
        if self._active == key:
            return
        if self._active is not None:
            self._fraction[self._active] = 1.0
        self._active, self._status = key, ""
        if self._live is None:
            self._console.print(
                f"{_ROW_INDENT}{self._phases[key].label}...", markup=False, highlight=False
            )

    def _render(self) -> RenderableType:
        """The single line: spinner, overall bar, percent, status and elapsed time."""
        with self._lock:
            total = sum(phase.weight for phase in self._phases.values()) or 1.0
            done = sum(self._phases[key].weight * value for key, value in self._fraction.items())
            share = min(1.0, done / total)
            status = ""
            if self._active is not None:
                phase = self._phases[self._active]
                status = phase.label
                if self._active in self._counts:
                    completed, of = self._counts[self._active]
                    status += f" {completed}/{of} {phase.unit}".rstrip()
                if self._status:
                    status += f" · {self._status}"
        filled = round(_BAR_WIDTH * share)
        line = Text(no_wrap=True, overflow="ellipsis")
        line.append("\u2501" * filled, style="cyan")
        line.append("\u2501" * (_BAR_WIDTH - filled), style="dim")
        line.append(f" {share:4.0%}", style="cyan")
        if status:
            line.append(f" · {status}", style="cyan")
        elapsed = max(0, int(monotonic() - self._started))
        line.append(f" · {_duration(elapsed) if elapsed else '0s'} elapsed", style="dim")
        self._spinner.update(text=line)
        return self._spinner


@contextmanager
def pipeline_display(console: Console, phases: Sequence[Phase]) -> Iterator[PipelineDisplay]:
    """Own one single-line progress region for a whole multi-phase command."""
    display = PipelineDisplay(console, phases)
    display.start()
    try:
        yield display
    finally:
        display.stop()


def _key(event: ProgressEvent) -> tuple[str, str | None]:
    """Return the identity that separates one stage row from the next."""
    return (event.stage, event.detail)


def _name(event: ProgressEvent) -> str:
    """Format one stage name with its optional qualifier.

    Args:
        event: Progress update being rendered.

    Returns:
        The stage name with its optional parenthesized detail.
    """
    if event.detail:
        return f"{event.stage} ({event.detail})"
    return event.stage


def _label(event: ProgressEvent) -> str:
    """Format one stage with its qualifier and exact counts.

    Args:
        event: Progress update being rendered.

    Returns:
        The stage name, optional parenthesized detail, and ``completed/total`` counts.
    """
    parts = [_name(event)]
    if event.completed is not None and event.total is not None:
        parts.append(f"{event.completed}/{event.total}")
    return " ".join(parts)


def _duration(seconds: float) -> str:
    """Format a positive duration compactly for the in-place line.

    Args:
        seconds: Positive estimated remaining seconds.

    Returns:
        Whole seconds below one minute, then ``NmSSs``, then ``NhMMm``.
    """
    rounded = max(round(seconds), 1)
    if rounded < 60:
        return f"{rounded}s"
    minutes, secs = divmod(rounded, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"
