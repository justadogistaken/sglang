from collections import deque


class SpeculativeTuner:
    def __init__(
        self, max_step: int, window_size: int = 10, target_acceptance_rate: float = 0.6
    ):
        self.max_step = max_step
        self.window_size = window_size
        self.target_acceptance_rate = target_acceptance_rate

        # Maps batch_size -> current_step suggestion
        self.bs_to_step = {}

        # Maps batch_size -> deque of (step, acceptance_rate)
        self.history = {}

    def get_step(self, batch_size: int) -> int:
        """
        Get the recommended speculative steps for a given batch size.
        """
        # Bucketize batch size to avoid too many sparse entries (e.g., nearest multiple of 4 or 8)
        # For now, let's just use exact batch size or maybe a simple bucket.
        # batch_size buckets: 1, 2-4, 5-8, 9-16, 17-32, ...
        # Let's simple use the batch size itself for now as it doesn't change too wildly in some workloads,
        # but for serving it might. Let's stick to exact batch size.

        if batch_size not in self.bs_to_step:
            # Initialize with max_step or a heuristic (e.g., max_step // 2)
            # Starting with max_step is optimistic.
            self.bs_to_step[batch_size] = self.max_step

        return self.bs_to_step[batch_size]

    def update(
        self, batch_size: int, step: int, accepted_tokens: int, drafted_tokens: int
    ):
        """
        Update the tuner with the result of a step.
        """
        if drafted_tokens == 0:
            return

        rate = accepted_tokens / drafted_tokens

        if batch_size not in self.history:
            self.history[batch_size] = deque(maxlen=self.window_size)

        self.history[batch_size].append((step, rate))

        # Analyze history to update step
        self._adjust_step(batch_size)

    def _adjust_step(self, batch_size: int):
        # Calculate average acceptance rate for the current step
        current_step = self.bs_to_step.get(batch_size, self.max_step)

        recent_records = [r for r in self.history[batch_size] if r[0] == current_step]
        if not recent_records:
            return

        # Weighted average or simple average? Simple for now.
        avg_rate = sum(r[1] for r in recent_records) / len(recent_records)

        # Heuristic adjustment
        # If acceptance rate is high, it implies the model is confident and we might be "wasting" potential by not drafting more.
        # If acceptance rate is low, we are wasting compute on rejected tokens.

        # Thresholds can be tuned.
        # If rate > 0.8, try increasing step.
        # If rate < 0.5, try decreasing step.

        new_step = current_step
        if avg_rate > self.target_acceptance_rate + 0.1:
            new_step = min(self.max_step, current_step + 1)
        elif avg_rate < self.target_acceptance_rate - 0.1:
            new_step = max(1, current_step - 1)

        self.bs_to_step[batch_size] = new_step
