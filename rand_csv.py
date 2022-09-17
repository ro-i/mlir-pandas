#!/usr/bin/env python3

import csv
import random

iterations: int = int(1e6)
max_rand: int = 2**63 - 1

def rand() -> int:
    return random.randint(0, max_rand)

with open(f"rand_{iterations}.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile, dialect="unix")
    writer.writerows(
        (rand(), rand(), rand())
        for _ in range(iterations)
    )
