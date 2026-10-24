"""Select measured calibration configuration; --max-accuracy-drop uses points."""

import sys

from ..metrics.__main__ import main as _main


def main(argv=None):
    return _main(["calibrate"] + list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    main()
