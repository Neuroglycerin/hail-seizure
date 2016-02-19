#!/usr/bin/env python3

import python.utils as utils
from python.predict import main


if __name__ == '__main__':

    parser = utils.get_parser()
    args = parser.parse_args()

    main(settings_file=args.settings, verbose=args.verbose)
