from __future__ import annotations

import queue
import threading
import time
from typing import Any

import zmq

from sglang.srt.speculative.decoupled_spec_io import (
    DraftClose,
    DraftControlBatch,
    DraftMeshIpcConfig,
    DraftMeshMessage,
    DraftMeshMessageType,
    DraftSync,
    VerifyCommit,
    iter_control_batch_messages,
)
from sglang.srt.speculative.draft_tail_buffer import DraftTailBuffer
from sglang.srt.utils import get_zmq_socket


class DraftProxyThread:
    """
    Verifier-side proxy thread for decoupled speculation.

    Control batches from the verifier are first applied to the local
    DraftTailBuffer, then forwarded to the drafter. Draft tail stream batches
    from the drafter are appended to the same buffer.
    """

    def __init__(
        self,
        *,
        context: zmq.Context,
        ipc_config: DraftMeshIpcConfig,
        verifier_rank: int,
        draft_tail_buffer: DraftTailBuffer,
        tracer: Any = None,
    ) -> None:
        self.verifier_rank = int(verifier_rank)
        self.draft_tail_buffer = draft_tail_buffer
        self.tracer = tracer
        # verifier -> drafter send control messages
        self.control_send_sockets: dict[int, zmq.Socket] = {
            drafter_rank: get_zmq_socket(
                context,
                zmq.PUSH,
                endpoint,
                False,
            )
            for drafter_rank, endpoint in sorted(ipc_config.control_endpoints.items())
        }
        self.result_recv_socket = get_zmq_socket(
            context,
            zmq.PULL,
            ipc_config.get_result_endpoint(self.verifier_rank),
            True,
        )
        self._send_queue: queue.SimpleQueue[DraftMeshMessage] = queue.SimpleQueue()
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="sglang-draft-proxy",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._closed.set()
        self.draft_tail_buffer.close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        for socket in self.control_send_sockets.values():
            socket.close(linger=0)
        self.result_recv_socket.close(linger=0)

    def submit_sync(self, message: DraftSync) -> None:
        self.submit_control_batch(
            DraftControlBatch(
                dst_drafter_rank=int(message.dst_drafter_rank),
                sync_messages=[message],
            )
        )

    def submit_verify_commit(self, message: VerifyCommit) -> None:
        self.submit_control_batch(
            DraftControlBatch(
                dst_drafter_rank=int(message.dst_drafter_rank),
                verify_commit_messages=[message],
            )
        )

    def submit_close(self, message: DraftClose) -> None:
        self.submit_control_batch(
            DraftControlBatch(
                dst_drafter_rank=int(message.dst_drafter_rank),
                close_messages=[message],
            )
        )

    def submit_control_batch(self, batch: DraftControlBatch) -> None:
        # Apply to local buffer first (thread-safe)
        self.draft_tail_buffer.apply_control_batch(batch)
        # Enqueue for async send to drafter
        self._send_queue.put(DraftMeshMessage.from_control_batch(batch))

    def _run(self) -> None:
        while not self._closed.is_set():
            did_work = False
            try:
                did_work = self._drain_send_queue() or did_work
                did_work = self._drain_result_socket() or did_work
            except zmq.error.ContextTerminated:
                break

            if not did_work:
                time.sleep(0.0005)  # 0.5ms

    def _drain_send_queue(self) -> bool:
        did_work = False
        while True:
            try:
                message = self._send_queue.get_nowait()
            except queue.Empty:
                break
            did_work = True
            self._send_control_batch(message)
        return did_work

    def _send_control_batch(self, message: DraftMeshMessage) -> None:
        if message.control_batch is None:
            return
        batch = message.control_batch
        dst_drafter_rank = int(batch.dst_drafter_rank)
        socket = self.control_send_sockets.get(dst_drafter_rank)
        if socket is None:
            raise RuntimeError(
                f"Missing control socket for dst_drafter_rank={dst_drafter_rank}"
            )

        trace_enabled = getattr(getattr(self, "tracer", None), "enabled", False)
        start_ns = time.perf_counter_ns() if trace_enabled else 0
        socket.send_pyobj(message)
        if trace_enabled:
            messages = iter_control_batch_messages(batch)
            self.tracer.record(
                "draft_proxy",
                "send_control_batch",
                duration_ms=(time.perf_counter_ns() - start_ns) / 1_000_000,
                verifier_rank=self.verifier_rank,
                dst_drafter_rank=dst_drafter_rank,
                batch_size=len(messages),
                num_sync=sum(isinstance(m, DraftSync) for m in messages),
                num_commit=sum(isinstance(m, VerifyCommit) for m in messages),
                num_close=sum(isinstance(m, DraftClose) for m in messages),
                request_ids=[m.request_id for m in messages],
            )

    def _drain_result_socket(self) -> bool:
        did_work = False
        while True:
            try:
                trace_enabled = getattr(getattr(self, "tracer", None), "enabled", False)
                start_ns = time.perf_counter_ns() if trace_enabled else 0
                message = self.result_recv_socket.recv_pyobj(zmq.NOBLOCK)
                recv_duration_ms = (
                    (time.perf_counter_ns() - start_ns) / 1_000_000
                    if trace_enabled
                    else 0
                )
            except zmq.error.ContextTerminated:
                raise
            except zmq.ZMQError:
                break
            did_work = True
            if not isinstance(message, DraftMeshMessage):
                continue
            if message.message_type == DraftMeshMessageType.TAIL_STREAM_OUTPUT_BATCH:
                batch = message.tail_stream_output_batch
                if batch is not None and batch.stream_outputs:
                    self.draft_tail_buffer.append_draft_stream(batch.stream_outputs)
                    if trace_enabled:
                        counts_by_request: dict[str, int] = {}
                        for output in batch.stream_outputs:
                            counts_by_request[output.request_id] = (
                                counts_by_request.get(output.request_id, 0) + 1
                            )
                        request_ids = list(counts_by_request.keys())
                        self.tracer.record(
                            "draft_proxy",
                            "recv_result_batch",
                            duration_ms=recv_duration_ms,
                            verifier_rank=self.verifier_rank,
                            batch_size=len(request_ids),
                            num_stream_outputs=len(batch.stream_outputs),
                            request_ids=request_ids,
                            draft_token_lens_by_req=[
                                counts_by_request[rid] for rid in request_ids
                            ],
                        )
            elif message.message_type == DraftMeshMessageType.TAIL_STREAM_OUTPUT:
                output = message.tail_stream_output
                if output is not None:
                    self.draft_tail_buffer.append_draft_stream([output])
        return did_work
