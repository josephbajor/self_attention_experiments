import numpy as np
from rich.progress import ProgressColumn, Task, filesize
from rich.text import Text


def generate_rand_emb(x, y):
    rng = np.random.default_rng(seed=42)

    # generate initial set
    e = {tuple(rng.binomial(1, 0.5, x)) for i in range(y)}

    while len(e) < y:
        e.add(rng.binomial(1, 0.5, x))

    return np.array(tuple(e))


class RateColumn(ProgressColumn):
    """Renders human readable processing rate."""

    def render(self, task: "Task") -> Text:
        """Render the speed in iterations per second."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("", style="progress.percentage")
        unit, suffix = filesize.pick_unit_and_suffix(
            int(speed),
            ["", "×10³", "×10⁶", "×10⁹", "×10¹²"],
            1000,
        )
        data_speed = speed / unit
        return Text(f"{data_speed:.1f}{suffix} it/s", style="progress.percentage")
