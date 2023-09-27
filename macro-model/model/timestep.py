class Timestep:
    def __init__(self, year: int, month: int):
        self.year = year
        self.month = month

    def __str__(self):
        return str(self.year) + "-" + str(self.month)

    def step(self) -> None:
        self.month += 1
        if self.month == 13:
            self.year += 1
            self.month = 1
