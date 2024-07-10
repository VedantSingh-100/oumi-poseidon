import functools
import pathlib
from contextlib import contextmanager
from typing import Optional

import torch

from lema.core.types.params.profiler_params import ProfilerParams
from lema.logging import logger
from lema.utils.torch_utils import DeviceRankInfo, get_device_rank_info

_PROFILER_LOG_PREFIX = "PROF:"
_PROFILER_DEFAULT_SUB_DIR = "profiler"


def _configure_torch_profile_save_dir(
    params: ProfilerParams, training_output_dir: Optional[str]
) -> ProfilerParams:
    """Auto-generates ProfilerParams.saved_dir if not specified explicitly."""
    if not params.save_dir and training_output_dir:
        params.save_dir = str(
            pathlib.Path(training_output_dir) / _PROFILER_DEFAULT_SUB_DIR
        )
    return params


def _on_trace_ready(
    prof,
    *,
    out_prefix: str,
    enable_cpu_profiling: bool,
    enable_cuda_profiling: bool,
    params: ProfilerParams,
    save_dir_path: Optional[pathlib.Path],
) -> None:
    logger.info(f"{_PROFILER_LOG_PREFIX} on_trace_ready(out_prefix={out_prefix})")
    sort_by = []
    if enable_cpu_profiling:
        sort_by.extend(
            [
                "cpu_time_total",
                "self_cpu_time_total",
            ]
        )
        if params.profile_memory:
            sort_by.extend(
                [
                    "cpu_memory_usage",
                    "self_cpu_memory_usage",
                ]
            )

    if enable_cuda_profiling:
        sort_by.extend(
            [
                "cuda_time_total",
                "self_cuda_time_total",
            ]
        )
        if params.profile_memory:
            sort_by.extend(
                [
                    "cuda_memory_usage",
                    "self_cuda_memory_usage",
                ]
            )
    # if `params.record_shapes` is True, then also generate reports with breakdowns
    # by tensor shapes. Otherwise (the default), only produce profiling reports
    # without shape breakdowns (less verbose).
    for group_by_input_shape in [False] + ([True] if params.record_shapes else []):
        group_by_shape_tag = "_by_shape" if group_by_input_shape else ""
        prof_avgs = prof.key_averages(group_by_input_shape=group_by_input_shape)
        for sort_key in sort_by:
            prof_table = prof_avgs.table(sort_by=sort_key, row_limit=params.row_limit)
            logger.info(
                f"{_PROFILER_LOG_PREFIX} {sort_key} "
                f"[group_by_shape={group_by_input_shape}]"
                f"\n{prof_table}\n"
            )
            if save_dir_path:
                file_path: pathlib.Path = (
                    save_dir_path / f"{out_prefix}_{sort_key}{group_by_shape_tag}.txt"
                )
                with file_path.open("w") as f:
                    f.write(prof_table)

    if save_dir_path:
        file_name: pathlib.Path = save_dir_path / f"{out_prefix}_pt_trace.json"
        logger.info(f"Exporting profiler Chrome trace to {file_name} ...")
        prof.export_chrome_trace(str(file_name))


@contextmanager
def torch_profile(
    params: ProfilerParams,
    *,
    training_output_dir: Optional[str],
    record_function_name: str = "lema.train",
):
    """Initializes Profiler context."""
    params = _configure_torch_profile_save_dir(params, training_output_dir)

    device_rank_info: DeviceRankInfo = get_device_rank_info()
    out_prefix: str = f"prof_{device_rank_info.rank}"

    profile_activities = []
    enable_cpu_profiling = params.enable_cpu_profiling
    if enable_cpu_profiling:
        profile_activities.append(torch.profiler.ProfilerActivity.CPU)

    enable_cuda_profiling = False
    if params.enable_cuda_profiling:
        enable_cuda_profiling = torch.cuda.is_available()
        if enable_cuda_profiling:
            profile_activities.append(torch.profiler.ProfilerActivity.CUDA)
        else:
            logger.warning(
                f"{_PROFILER_LOG_PREFIX} CUDA profiling is requested "
                "while CUDA is not available!"
            )

    if not profile_activities:
        # Nothing to profile. Return noop/null context.
        logger.info(f"{_PROFILER_LOG_PREFIX} Torch Profiler disabled!")
        yield
        return

    logger.info(f"{_PROFILER_LOG_PREFIX} Starting profiling...")
    logger.info(f"{_PROFILER_LOG_PREFIX} Save dir: {params.save_dir}")
    save_dir_path: Optional[pathlib.Path] = (
        pathlib.Path(params.save_dir) if params.save_dir else None
    )
    if save_dir_path:
        save_dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"{_PROFILER_LOG_PREFIX} Save dir: {save_dir_path}")
    logger.info(f"{_PROFILER_LOG_PREFIX} Output prefix: {out_prefix}")
    logger.info(f"{_PROFILER_LOG_PREFIX} Function: {record_function_name}")
    logger.info(f"{_PROFILER_LOG_PREFIX} Params: {params}")

    # See also torch.profiler.tensorboard_trace_handler
    trace_handler = functools.partial(
        _on_trace_ready,
        out_prefix=out_prefix,
        enable_cpu_profiling=enable_cpu_profiling,
        enable_cuda_profiling=enable_cuda_profiling,
        params=params,
        save_dir_path=save_dir_path,
    )

    with torch.profiler.profile(
        activities=profile_activities,
        on_trace_ready=trace_handler,
        record_shapes=params.record_shapes,
        profile_memory=params.profile_memory,
        with_stack=params.with_stack,
        with_flops=params.with_flops,
        with_modules=params.with_modules,
    ):
        try:
            with torch.profiler.record_function(record_function_name):
                yield
        except Exception as e:
            # The inner function raised an error
            import traceback

            logger.error(
                f"{_PROFILER_LOG_PREFIX}"
                + "".join(traceback.format_exception(None, e, e.__traceback__))
            )
            raise

    logger.info(f"{_PROFILER_LOG_PREFIX} Finished post-processing!")
    return