import math

class LossScheduler:
    def __init__(self, start, end, max_steps, type="cosine"):
        self.start = start
        self.end = end
        self.max_steps = max_steps
        self.type = type

    def step(self, current_step):
        # decay value from start to end with cosine annealing
        if self.type == "cosine":
            return int(
                self.end
                + (self.start - self.end)
                * (1 + math.cos(math.pi * current_step / self.max_steps))
                / 2
            )

        if self.type == 'linear':
            return int(self.start - (self.start - self.end) * current_step / self.max_steps)


class LossScheduler2:
    def __init__(self, start, end, max_steps, increase_fraction=0.3, type="cosine"):
        self.min_value = end
        self.max_value = start
        self.max_steps = max_steps
        self.increase_fraction = increase_fraction
        self.type = type

    def step(self, current_step):
        if self.type == "cosine":
            if current_step <= self.max_steps * self.increase_fraction:
                # Linear increase phase
                value = self.min_value + (self.max_value - self.min_value) * current_step / (self.max_steps * self.increase_fraction)
            else:
                # Cosine annealing phase
                # Adjust the phase and scale to match the cosine annealing pattern from the peak to the end
                adjusted_step = current_step - self.max_steps * self.increase_fraction
                adjusted_max_steps = self.max_steps * (1 - self.increase_fraction)
                value = self.min_value + (self.max_value - self.min_value) * (1 + math.cos(math.pi * adjusted_step / adjusted_max_steps)) / 2
            return int(value)