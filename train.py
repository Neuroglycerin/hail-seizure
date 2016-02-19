#!/usr/bin/env python3

import python.utils as utils
from python.train import main


if __name__ == '__main__':

    # get and parse CLI options
    parser = utils.get_parser()
    args = parser.parse_args()

    main(args.settings,
         verbose=args.verbose,
         save_training_detailed=args.pickle_detailed,
         parallel=int(args.parallel))
