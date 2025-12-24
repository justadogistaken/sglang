import logging
import random
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class SpeculativeController:
    def __init__(self, initial_steps, tune_interval=10):
        self.max_steps = initial_steps
        self.step_candidates = list(range(1, initial_steps + 1))

        # Stats: (batch_size_bucket, num_steps) -> list of (latency, num_accepted_tokens)
        self.history = defaultdict(lambda: deque(maxlen=20))

        # Configuration
        self.warmup_samples = (
            5  # Require at least N samples to consider a config "known"
        )
        self.exploration_rate = 0.10  # 10% chance to explore random steps
        self.bs_bucket_size = 4

        # Current decision cache
        self.last_decision = {}  # bs_bucket -> num_steps

    def get_num_steps(self, batch_size):
        bs_bucket = (
            (batch_size + self.bs_bucket_size - 1)
            // self.bs_bucket_size
            * self.bs_bucket_size
        )

        # 1. Warmup / Exploration of unknown territory
        # If any candidate has insufficient samples, prioritize it (Round Robin style ideally, here just first found)
        for step in self.step_candidates:
            key = (bs_bucket, step)
            if len(self.history[key]) < self.warmup_samples:
                return step

        # 2. Epsilon-Greedy Exploration
        if random.random() < self.exploration_rate:
            return random.choice(self.step_candidates)

        # 3. Exploitation: Find the step with max TPS (Tokens Per Second)
        best_step = 1
        max_tps = -1.0

        for step in self.step_candidates:
            key = (bs_bucket, step)
            samples = self.history[key]
            if not samples:
                continue

            total_latency = sum(s[0] for s in samples)
            total_tokens = sum(s[1] for s in samples)

            if total_latency > 0:
                tps = total_tokens / total_latency
                if tps > max_tps:
                    max_tps = tps
                    best_step = step

        if max_tps > 0:
            return best_step

        return self.max_steps  # Default fallback

    def update(self, batch_size, accept_lens, latency):
        """
        batch_size: int
        accept_lens: list of int (number of tokens accepted for each req)
        latency: float (execution time in seconds for this batch step)
        """
        if latency <= 0:
            return

        bs_bucket = (
            (batch_size + self.bs_bucket_size - 1)
            // self.bs_bucket_size
            * self.bs_bucket_size
        )

        # Recover which step was likely used?
        # Actually, since we update immediately after execution, we can infer it
        # However, purely based on accepted length is hard if we don't know the `num_steps` used.
        # But wait, the Scheduler passes `speculative_num_steps` TO the batch.
        # We need to know what `num_steps` was actually used for THIS batch data.
        # The update method in SchedulerOutputProcessorMixin needs to pass the `num_steps` used.
        pass

    def update_with_step(self, batch_size, num_steps, accept_lens, latency):
        bs_bucket = (
            (batch_size + self.bs_bucket_size - 1)
            // self.bs_bucket_size
            * self.bs_bucket_size
        )
        key = (bs_bucket, num_steps)

        # accept_lens contains the raw accepted length (including the one verified token?)
        # Usually result.num_accepted_tokens is passed which is sum(accept_lens)
        # But we need sum of accepted tokens in the batch.
        total_accepted = sum(accept_lens)

        self.history[key].append((latency, total_accepted))

        # Optional: Log periodically
        if random.random() < 0.05:
            logger.info(
                f"[SpecController] BS={bs_bucket}, Step={num_steps}, Latency={latency*1000:.2f}ms, Tokens={total_accepted}, TPS={total_accepted/latency:.1f}"
            )
