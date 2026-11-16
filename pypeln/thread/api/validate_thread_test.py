import threading
import time
import typing as tp

import hypothesis as hp
from hypothesis import strategies as st

import pypeln as pl

MAX_EXAMPLES = 10
T = tp.TypeVar("T")


class MyError(Exception):
    pass


class MyValidationError(Exception):
    pass


def _split_f(x):
    if x % 2:
        raise pl.thread.ValidationError("odd")
    return x * 2


@hp.given(nums=st.lists(st.integers()))
@hp.settings(max_examples=MAX_EXAMPLES)
def test_validate_all_valid(nums: tp.List[int]):
    result = pl.thread.validate(lambda x: x**2, nums)
    result.quarantined.close()

    assert list(result.qualified) == [x**2 for x in nums]
    assert list(result.quarantined) == []


@hp.given(nums=st.lists(st.integers()))
@hp.settings(max_examples=MAX_EXAMPLES)
def test_validate_split(nums: tp.List[int]):
    result = pl.thread.validate(_split_f, nums)

    # the quarantined path is only consumed after the qualified one finished:
    # it must still drain every result it received
    assert list(result.qualified) == [x * 2 for x in nums if x % 2 == 0]

    quarantined = list(result.quarantined)

    assert [r.value for r in quarantined] == [x for x in nums if x % 2]
    assert [r.index for r in quarantined] == [(i,) for i, x in enumerate(nums) if x % 2]
    assert all(isinstance(r.error, pl.thread.ValidationError) for r in quarantined)


@hp.given(nums=st.lists(st.integers()))
@hp.settings(max_examples=MAX_EXAMPLES)
def test_validate_pipe(nums: tp.List[int]):
    result = nums | pl.thread.validate(lambda x: x)
    result.quarantined.close()

    assert list(result.qualified) == nums


@hp.given(nums=st.lists(st.integers()))
@hp.settings(max_examples=MAX_EXAMPLES)
def test_validate_workers(nums: tp.List[int]):
    result = pl.thread.validate(lambda x: x**2, nums, workers=3)
    result.quarantined.close()

    assert sorted(list(result.qualified)) == sorted([x**2 for x in nums])


def test_validate_return_index():
    result = pl.thread.validate(lambda x: x * 2, range(10))
    result.quarantined.close()

    elems = list(pl.thread.to_iterable(result.qualified, return_index=True))

    assert [elem.index for elem in elems] == [(i,) for i in range(10)]
    assert [elem.value for elem in elems] == [2 * i for i in range(10)]


def test_validate_custom_except_type():
    def f(x):
        if x % 2:
            raise MyValidationError("odd")
        return x

    result = pl.thread.validate(f, range(10), except_type=MyValidationError)

    assert list(result.qualified) == [0, 2, 4, 6, 8]

    quarantined = list(result.quarantined)

    assert [r.value for r in quarantined] == [1, 3, 5, 7, 9]
    assert all(isinstance(r.error, MyValidationError) for r in quarantined)


def test_validate_except_type_tuple():
    def f(x):
        if x % 2:
            raise MyValidationError("odd")
        if x == 4:
            raise ValueError("four")
        return x

    result = pl.thread.validate(
        f, range(10), except_type=(MyValidationError, ValueError)
    )

    assert list(result.qualified) == [0, 2, 6, 8]

    quarantined = list(result.quarantined)

    assert [r.value for r in quarantined] == [1, 3, 4, 5, 7, 9]
    assert all(
        isinstance(r.error, (MyValidationError, ValueError)) for r in quarantined
    )


def test_validate_unexpected_error_both_paths():
    def f(x):
        if x == 3:
            raise MyError()
        return x

    result = pl.thread.validate(f, range(10))

    qualified_error = None

    try:
        list(result.qualified)
    except MyError as e:
        qualified_error = e

    assert isinstance(qualified_error, MyError)

    quarantined_error = None

    try:
        list(result.quarantined)
    except MyError as e:
        quarantined_error = e

    assert isinstance(quarantined_error, MyError)


def test_validate_upstream_error_both_paths():
    def raise_error(x):
        raise MyError()

    stage = pl.thread.map(raise_error, range(10))
    result = pl.thread.validate(lambda x: x, stage)

    qualified_error = None

    try:
        list(result.qualified)
    except MyError as e:
        qualified_error = e

    assert isinstance(qualified_error, MyError)

    quarantined_error = None

    try:
        list(result.quarantined)
    except MyError as e:
        quarantined_error = e

    assert isinstance(quarantined_error, MyError)


def test_validate_kwargs():
    nums = range(100)
    letters = "abc"
    namespace = pl.thread.Namespace(on_done=None, done=None)

    def on_start():
        return dict(y=letters)

    def on_done(y, stage_status):
        namespace.on_done = y
        namespace.done = stage_status.done

    result = pl.thread.validate(
        lambda x, y: y, nums, on_start=on_start, on_done=on_done
    )
    result.quarantined.close()

    nums_pl = list(result.qualified)

    assert namespace.on_done == letters
    assert namespace.done == True
    assert nums_pl == [letters] * len(nums)


def test_validate_element_index():
    result = pl.thread.validate(lambda x, element_index: element_index[0], range(10))
    result.quarantined.close()

    assert list(result.qualified) == list(range(10))


def test_validate_shared_execution():
    namespace = pl.thread.Namespace(count=0)
    qualified_out = []
    quarantined_out = []

    def f(x):
        with namespace:
            namespace.count += 1

        if x % 2:
            raise pl.thread.ValidationError("odd")

        return x

    result = pl.thread.validate(f, range(20), workers=2)

    pl.thread.run(
        result.qualified | pl.thread.each(qualified_out.append),
        result.quarantined | pl.thread.each(quarantined_out.append),
    )

    # the source was consumed and validated exactly once for both paths
    assert namespace.count == 20
    assert sorted(qualified_out) == [x for x in range(20) if x % 2 == 0]
    assert sorted(r.value for r in quarantined_out) == [x for x in range(20) if x % 2]


def test_validate_native_stages():
    result = range(20) | pl.thread.validate(_split_f)

    doubled = result.qualified | pl.thread.map(lambda x: x * 2)
    reviewed = result.quarantined | pl.thread.map(lambda r: r.value)

    assert sorted(doubled) == sorted(x * 4 for x in range(20) if x % 2 == 0)
    assert sorted(reviewed) == sorted(x for x in range(20) if x % 2)


def test_close_quarantined_path():
    result = pl.thread.validate(_split_f, range(20), quarantine_maxsize=2)

    quarantined_iter = iter(result.quarantined)
    first = next(quarantined_iter)

    assert isinstance(first, pl.thread.Quarantined)
    assert first.value == 1

    # closing the quarantined path must not stop the qualified one, and its
    # bounded queue must not block the shared execution once abandoned
    result.quarantined.close()

    assert list(result.qualified) == [x * 2 for x in range(20) if x % 2 == 0]


def test_close_qualified_path():
    result = pl.thread.validate(_split_f, range(20), qualified_maxsize=2)

    qualified_iter = iter(result.qualified)
    first = next(qualified_iter)

    assert first == 0

    result.qualified.close()

    quarantined = list(result.quarantined)

    assert [r.value for r in quarantined] == [x for x in range(20) if x % 2]


def test_close_both_paths_stops_execution():
    namespace = pl.thread.Namespace(count=0)

    def gen():
        for i in range(10**6):
            with namespace:
                namespace.count += 1
            yield i

    result = pl.thread.validate(
        lambda x: x, gen(), maxsize=2, qualified_maxsize=2, quarantine_maxsize=2
    )

    iterator = iter(result.qualified)
    next(iterator)

    result.close()

    time.sleep(0.3)
    count = namespace.count
    time.sleep(0.3)

    # the source execution stopped: no more elements are consumed
    assert namespace.count == count
    assert count < 10**6

    # iterating a closed path does not hang nor yield abandoned results
    assert list(result.qualified) == []
    assert list(result.quarantined) == []


def test_validate_backpressure():
    namespace = pl.thread.Namespace(count=0)

    def f(x) -> tp.Any:
        with namespace:
            namespace.count += 1
        return x

    result = pl.thread.validate(f, range(20), qualified_maxsize=3)
    result.quarantined.close()

    iterator = iter(result.qualified)
    next(iterator)

    time.sleep(0.1)

    # + 1 element which was yielded on next(...)
    # + 3 elements which are on the path queue.
    # + 1 element which is pending to be routed.
    # -------------------------------------------
    # + 5 total
    assert namespace.count == 5
