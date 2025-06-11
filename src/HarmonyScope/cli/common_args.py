import argparse


def add_common_args(ap: argparse.ArgumentParser) -> None:
    group = ap.add_argument_group("Common analysis parameters")

    group.add_argument(
        "--window",
        type=float,
        default=0.75,
        metavar="SEC",
        help="Analysis window size (sec)",
    )
    group.add_argument(
        "--interval",
        type=float,
        default=0.05,
        metavar="SEC",
        help="Hop length between analyses (sec)",
    )
    group.add_argument(
        "--min-frame-ratio",
        type=float,
        default=0.3,
        metavar="R",
        help="Min voiced-frame ratio for a pitch class to become active",
    )
    group.add_argument(
        "--min-prominence-db",
        type=float,
        default=8.0,
        metavar="dB",
        help="Min spectral-peak prominence (dB)",
    )
    group.add_argument(
        "--max-level-diff-db",
        type=float,
        default=15.0,
        metavar="dB",
        help="Max dB diff from loudest peak in frame",
    )
    group.add_argument(
        "--frame-energy-thresh-db",
        type=float,
        default=-40.0,
        metavar="dB",
        help="RMS threshold to mark frame voiced (dB)",
    )
    group.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="-v / -vv to raise log level",
    )
