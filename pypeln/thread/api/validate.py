import threading
import time
import typing as tp
from dataclasses import dataclass
from queue import Empty, Full, Queue

from pypeln import utils as pypeln_utils
from pypeln.utils import A, B

from .. import utils
from ..queue import IterableQueue
from ..stage import Stage
from ..supervisor import Supervisor
from ..worker import ApplyProcess, Worker
from .from_iterable import FromIterable
from .to_stage import to_stage


class ValidationError(Exception):
    """
    Default "agreed" validation exception for `pl.thread.validate`: raise it from the
    validation function to send the offending record to the quarantined path. Custom
    exception types can be used instead through the `except_type` argument.
    """


class Quarantined(tp.NamedTuple):
    """
    Record emitted on the quarantined path of `pl.thread.validate`.

    Attributes:
        value: The original input record that failed validation.
        index: The index the record had at the source.
        error: The agreed validation exception raised for the record.
    """

    value: tp.Any
    index: tp.Tuple[int, ...]
    error: BaseException


class ValidateFn(pypeln_utils.Protocol):
    def __call__(self, A, **kwargs) -> B:
        ...


def _materialize_exception(
    pipeline_exception: pypeln_utils.PipelineException,
) -> BaseException:
    exception, trace = pipeline_exception

    try:
        exception = exception(f"\n\n{trace}")
    except:
        exception = Exception(f"\n\nOriginal: {exception}\n\n{trace}")

    return exception


class _PathState:
    """
    Output queue of one path plus its open/closed state.
    """

    def __init__(self, maxsize: int, total_sources: int):
        self.queue: IterableQueue = IterableQueue(
            maxsize=maxsize, total_sources=total_sources
        )
        self.namespace = utils.Namespace(closed=False)

    @property
    def closed(self) -> bool:
        return self.namespace.closed

    def close(self):
        with self.namespace:
            self.namespace.closed = True

        # abandon whatever was received but not consumed yet
        self.queue.stop()


class _DualPathExecution:
    """
    Shared, single-pass run of a `validate` pipeline.

    The validation stage and its dependencies are built and started at most once,
    the first time any of the two paths is consumed. Each element produced by the
    validation workers is routed to exactly one of the two path queues, so the
    source is never consumed twice and every record is validated exactly once.
    """

    def __init__(self, workers: int, qualified_maxsize: int, quarantine_maxsize: int):
        self.qualified = _PathState(maxsize=qualified_maxsize, total_sources=workers)
        self.quarantined = _PathState(maxsize=quarantine_maxsize, total_sources=workers)
        self.namespace = utils.Namespace(started=False, stopped=False, error=None)
        self.stage: tp.Optional[Stage] = None
        self.workers: tp.List[Worker] = []
        self.supervisor: tp.Optional[Supervisor] = None

    # ----------------------------------------------------------------
    # routing
    # ----------------------------------------------------------------

    def _put(self, state: _PathState, elem: pypeln_utils.Element):
        # Blocks while the path queue is full so a slow path applies backpressure,
        # but gives up as soon as the path is closed or the execution is stopped:
        # a closed/failed path must never block the other one.
        while not state.closed and not self.namespace.stopped:
            try:
                Queue.put(state.queue, elem, block=True, timeout=pypeln_utils.TIMEOUT)
                return
            except Full:
                continue

    def put_qualified(self, elem: pypeln_utils.Element):
        self._put(self.qualified, elem)

    def put_quarantined(self, elem: pypeln_utils.Element):
        self._put(self.quarantined, elem)

    # ----------------------------------------------------------------
    # error handling (acts as the main_queue of every worker)
    # ----------------------------------------------------------------

    def raise_exception(self, exception: BaseException):
        # A single shared record of the failure: both paths re-raise it no matter
        # which one is consumed first, and buffered results are discarded in favor
        # of the error, mirroring pypeln's regular error semantics.
        pipeline_exception = self.qualified.queue.get_pipeline_exception(exception)

        with self.namespace:
            if self.namespace.error is None:
                self.namespace.error = pipeline_exception

        self.shutdown()

    # ----------------------------------------------------------------
    # lifecycle
    # ----------------------------------------------------------------

    def start(self):
        with self.namespace:
            if self.namespace.started:
                return

            self.namespace.started = True
            stopped = self.namespace.stopped

        if stopped:
            return

        built: tp.Dict[Stage, tp.Any] = {}
        workers = list(self.stage.build(built, self.qualified.queue, self))
        # Register the quarantined queue as a second output of the validation
        # stage without rebuilding it (the same mechanism pypeln uses to share
        # dependencies): workers are created only once and their `worker_done`
        # signals are broadcasted to both path queues.
        workers.extend(self.stage.build(built, self.quarantined.queue, self))

        self.workers = workers
        self.supervisor = Supervisor(workers=workers, main_queue=self)
        self.supervisor.start()

        threading.Thread(target=self._watch_workers, daemon=True).start()

        with self.namespace:
            stopped = self.namespace.stopped

        if stopped:
            # a close/error raced with the start, make sure nothing keeps running
            self._stop_workers()

    def shutdown(self):
        with self.namespace:
            if self.namespace.stopped:
                return

            self.namespace.stopped = True

        # Stop from a separate thread: `shutdown` is also triggered by the workers
        # themselves (through `raise_exception`) and `Worker.stop` asynchronously
        # raises `StopThreadException` on the target thread.
        threading.Thread(target=self._stop_workers, daemon=True).start()

    def close_path(self, path: str):
        state = self.qualified if path == "qualified" else self.quarantined
        state.close()

        if self.qualified.closed and self.quarantined.closed:
            # both paths abandoned: stop this run of the source
            self.shutdown()

    def iter_path(self, path: str) -> tp.Iterator[pypeln_utils.Element]:
        state = self.qualified if path == "qualified" else self.quarantined
        queue = state.queue

        self.start()

        while True:
            error = self.namespace.error

            if error is not None:
                raise _materialize_exception(error)

            if queue.is_done():
                return

            try:
                yield queue.get(timeout=pypeln_utils.TIMEOUT)
            except Empty:
                continue

    # ----------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------

    def _stop_workers(self):
        if self.supervisor is not None:
            self.supervisor.done = True

        for worker in self.workers:
            worker.stop()

    def _watch_workers(self):
        # release the supervisor's timeout monitor once every worker finished
        while any(
            worker.process is not None and worker.process.is_alive()
            for worker in self.workers
        ):
            time.sleep(0.01)

        if self.supervisor is not None:
            self.supervisor.done = True


@dataclass
class _Validate(ApplyProcess):
    f: ValidateFn
    except_type: tp.Union[tp.Type[BaseException], tp.Tuple[tp.Type[BaseException], ...]]
    execution: _DualPathExecution

    def apply(self, worker: Worker, elem: tp.Any, **kwargs):
        if "element_index" in worker.f_args:
            kwargs["element_index"] = elem.index

        try:
            y = self.f(elem.value, **kwargs)
        except self.except_type as e:
            self.execution.put_quarantined(
                elem.set(Quarantined(value=elem.value, index=elem.index, error=e))
            )
        else:
            self.execution.put_qualified(elem.set(y))


class _PathIterable(tp.NamedTuple):
    execution: _DualPathExecution
    path: str

    def __iter__(self) -> tp.Iterator[pypeln_utils.Element]:
        return self.execution.iter_path(self.path)


@dataclass(eq=False)
class ValidationPath(Stage[B]):
    """
    One output path of `pl.thread.validate`. It is a regular `pl.thread.Stage`: it
    can be iterated directly or used as a dependency of any other stage, and it can
    be closed to abandon its unconsumed results without affecting the other path.

    Both paths share a single execution of the pipeline, which starts the first
    time any of them is consumed.
    """

    execution: _DualPathExecution = None
    path: str = ""

    def to_iterable(
        self, maxsize: int = 0, return_index: bool = False
    ) -> tp.Iterable[B]:
        # Iterate the shared path queue directly: no extra workers or queues are
        # needed. Buffering/backpressure is bounded by the path's own `maxsize`
        # configured on `validate`, so the `maxsize` argument is ignored here.
        for elem in self.execution.iter_path(self.path):
            if return_index:
                yield elem
            else:
                yield elem.value

    def close(self):
        """
        Abandons the results received by this path that were not yet consumed and
        stops routing new ones to it, without stopping the other path. If both
        paths are closed the shared execution of the source is stopped.
        """
        self.execution.close_path(self.path)


class ValidationResult(tp.NamedTuple):
    """
    The two output paths of `pl.thread.validate`. Supports tuple unpacking:
    `qualified, quarantined = pl.thread.validate(f, stage)`.
    """

    qualified: ValidationPath
    quarantined: ValidationPath

    def close(self):
        """
        Closes both paths, stopping the shared execution of the source.
        """
        self.qualified.close()
        self.quarantined.close()


@tp.overload
def validate(
    f: ValidateFn,
    stage: tp.Union[Stage[A], tp.Iterable[A]],
    workers: int = 1,
    maxsize: int = 0,
    timeout: float = 0,
    on_start: tp.Callable = None,
    on_done: tp.Callable = None,
    except_type: tp.Union[
        tp.Type[BaseException], tp.Tuple[tp.Type[BaseException], ...]
    ] = ValidationError,
    qualified_maxsize: int = 0,
    quarantine_maxsize: int = 0,
) -> ValidationResult:
    ...


@tp.overload
def validate(
    f: ValidateFn,
    workers: int = 1,
    maxsize: int = 0,
    timeout: float = 0,
    on_start: tp.Callable = None,
    on_done: tp.Callable = None,
    except_type: tp.Union[
        tp.Type[BaseException], tp.Tuple[tp.Type[BaseException], ...]
    ] = ValidationError,
    qualified_maxsize: int = 0,
    quarantine_maxsize: int = 0,
) -> pypeln_utils.Partial[ValidationResult]:
    ...


def validate(
    f: ValidateFn,
    stage: tp.Union[
        Stage[A], tp.Iterable[A], pypeln_utils.Undefined
    ] = pypeln_utils.UNDEFINED,
    workers: int = 1,
    maxsize: int = 0,
    timeout: float = 0,
    on_start: tp.Callable = None,
    on_done: tp.Callable = None,
    except_type: tp.Union[
        tp.Type[BaseException], tp.Tuple[tp.Type[BaseException], ...]
    ] = ValidationError,
    qualified_maxsize: int = 0,
    quarantine_maxsize: int = 0,
) -> tp.Union[ValidationResult, pypeln_utils.Partial[ValidationResult]]:
    """
    Creates a pair of stages that share a single execution of a validation/cleaning
    function `f` over the data, routing each record to exactly one of two paths:

    * If `f(x)` returns a value, the value (with its original index) is routed to
      the **qualified** path.
    * If `f(x)` raises the agreed validation exception (`except_type`, defaults to
      `pl.thread.ValidationError`), a `Quarantined(value, index, error)` record is
      routed to the **quarantined** path.
    * Any other exception, or a failure of an upstream stage, is made visible on
      **both** paths: iterating either of them re-raises the error.

    The source is consumed and `f` is applied exactly once per element, no matter
    how the two paths are consumed. Each path is a regular `pl.thread.Stage`: it
    can be iterated independently of the other or used as a dependency of any
    other stage.

    ```python
    import pypeln as pl

    def clean(x):
        if x < 0:
            raise pl.thread.ValidationError("negative record")
        return x * 2

    result = pl.thread.validate(clean, range(-5, 5), workers=2)

    for value in result.qualified:
        print(value)  # cleaned records, e.g. 0, 2, 4, 6, 8

    for record in result.quarantined:
        print(record.value, record.error)  # rejected records sent to review
    ```

    !!! note
        Because of concurrency order is not guaranteed.

    Each path buffers results in its own bounded queue when `qualified_maxsize` /
    `quarantine_maxsize` are set, so a slow path applies backpressure to the shared
    execution. A path that is not needed anymore should be closed: closing a path
    abandons its unconsumed results without stopping the other path, and closing
    both paths stops the execution of the source. When the source finishes
    normally, paths that are still open drain the results they already received
    before ending.

    Arguments:
        f: A function with the signature `f(x) -> y` that validates/cleans each
            record. `f` can accept special additional arguments by name as
            described in [Advanced Usage](https://cgarciae.github.io/pypeln/advanced/#dependency-injection).
        stage: A Stage or Iterable.
        workers: The number of workers the validation stage should contain.
        maxsize: The maximum number of objects the validation stage can hold
            simultaneously, if set to `0` (default) then the stage can grow unbounded.
        timeout: Seconds before stoping the worker if its current task is not yet
            completed. Defaults to `0` which means its unbounded.
        on_start: A function with signature `on_start(worker_info?) -> kwargs?`,
            where `kwargs` can be a `dict` of keyword arguments that can be consumed
            by `f` and `on_done`. `on_start` can accept additional arguments by name
            as described in [Advanced Usage](https://cgarciae.github.io/pypeln/advanced/#dependency-injection).
        on_done: A function with signature `on_done(stage_status?)`. This function
            is executed once per worker when the worker finishes. `on_done` can
            accept additional arguments by name as described in [Advanced Usage](https://cgarciae.github.io/pypeln/advanced/#dependency-injection).
        except_type: The agreed validation exception type (or tuple of types) that
            routes a record to the quarantined path. Defaults to
            `pl.thread.ValidationError`.
        qualified_maxsize: The maximum number of results the qualified path can
            buffer simultaneously, if set to `0` (default) it can grow unbounded.
        quarantine_maxsize: The maximum number of results the quarantined path can
            buffer simultaneously, if set to `0` (default) it can grow unbounded.

    Returns:
        Returns a `ValidationResult` with the `qualified` and `quarantined` path
        stages if the `stage` parameter is given, else it returns a `Partial`.
    """

    if isinstance(stage, pypeln_utils.Undefined):
        return pypeln_utils.Partial(
            lambda stage: validate(
                f,
                stage=stage,
                workers=workers,
                maxsize=maxsize,
                timeout=timeout,
                on_start=on_start,
                on_done=on_done,
                except_type=except_type,
                qualified_maxsize=qualified_maxsize,
                quarantine_maxsize=quarantine_maxsize,
            )
        )

    stage = to_stage(stage, maxsize=maxsize)

    execution = _DualPathExecution(
        workers=workers,
        qualified_maxsize=qualified_maxsize,
        quarantine_maxsize=quarantine_maxsize,
    )

    execution.stage = Stage(
        process_fn=_Validate(f=f, except_type=except_type, execution=execution),
        workers=workers,
        maxsize=maxsize,
        timeout=timeout,
        total_sources=stage.workers,
        dependencies=[stage],
        on_start=on_start,
        on_done=on_done,
        f_args=pypeln_utils.function_args(f),
    )

    return ValidationResult(
        qualified=ValidationPath(
            process_fn=FromIterable(_PathIterable(execution, "qualified"), maxsize=0),
            workers=1,
            maxsize=0,
            total_sources=1,
            timeout=0,
            dependencies=[],
            on_start=None,
            on_done=None,
            f_args=[],
            execution=execution,
            path="qualified",
        ),
        quarantined=ValidationPath(
            process_fn=FromIterable(_PathIterable(execution, "quarantined"), maxsize=0),
            workers=1,
            maxsize=0,
            total_sources=1,
            timeout=0,
            dependencies=[],
            on_start=None,
            on_done=None,
            f_args=[],
            execution=execution,
            path="quarantined",
        ),
    )
