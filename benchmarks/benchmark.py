import argparse
import logging
from typing import List


from common import TPCHResult
import tpc_h_1
import tpc_h_5
import tpc_h_6


def run() -> None:
    results: List[TPCHResult] = [
        tpc_h_1.run(),
        tpc_h_5.run(),
        tpc_h_6.run(),
    ]
    for result in results:
        print(result)


if __name__ == "__main__":
    argument_parser: argparse.ArgumentParser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "-d", "--debug", action="store_true", help="debugging mode switch"
    )
    args: argparse.Namespace = argument_parser.parse_args()
    if args.debug:
        logging.basicConfig(filename="benchmarks.log", filemode="w", level=logging.DEBUG)
    run()
