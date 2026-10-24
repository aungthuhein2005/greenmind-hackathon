"""SLO-aware admission order for vLLM's priority scheduler (loaded in-process via --scheduler-cls).

Newest-first (newest_first.py) tripled on-time requests under the Final trace's overload, but the
requests it starves wait so long that p95 TTFT hit 42 s and the latency score collapsed (sL 0.12).

Before every scheduling step this re-ranks the waiting queue by each request's TTFT deadline
(arrival + 0.5 s + 0.25 s per 1K prompt tokens, the grader's SLO):
  1. still feasible  -> earliest deadline first (serve the one about to expire while it can still make it)
  2. already missed  -> oldest first, after all feasible ones (bounds how late the backlog gets)
Nothing is dropped and no request parameter is changed; only the order of admission.

Used with: --scheduling-policy priority --scheduler-cls slo_scheduler.SLOScheduler
Organizers confirmed on Discord (Oct 24) that server-side request prioritization is allowed.
"""
import heapq
import os
import time

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import PriorityRequestQueue
from vllm.v1.core.sched.scheduler import Scheduler

PREFILL_TPS = float(os.environ.get("SLO_PREFILL_TPS", 6000))  # prompt tokens/s while decoding
MARGIN_S = float(os.environ.get("SLO_MARGIN_S", 0.2))
MISSED = 10**13  # priority offset that puts missed requests behind every feasible one


class _SLOOrder:
    def schedule(self, *args, **kwargs):
        q = self.waiting
        if isinstance(q, PriorityRequestQueue) and len(q) > 1:
            now = time.time()
            for r in q._heap:
                deadline = r.arrival_time + 0.5 + 0.25e-3 * r.num_prompt_tokens
                if deadline - now > r.num_prompt_tokens / PREFILL_TPS + MARGIN_S:
                    r.priority = int(deadline * 1000)
                else:
                    r.priority = MISSED + int(r.arrival_time * 1000)
            heapq.heapify(q._heap)
        return super().schedule(*args, **kwargs)


class _SLOScheduler(_SLOOrder, Scheduler):
    pass


class _SLOAsyncScheduler(_SLOOrder, AsyncScheduler):
    pass


class SLOScheduler:
    """Builds the SLO-ordered variant of whichever scheduler vLLM would have used."""

    def __new__(cls, *args, **kwargs):
        vllm_config = kwargs["vllm_config"] if "vllm_config" in kwargs else args[0]
        base = _SLOAsyncScheduler if vllm_config.scheduler_config.async_scheduling else _SLOScheduler
        return base(*args, **kwargs)
