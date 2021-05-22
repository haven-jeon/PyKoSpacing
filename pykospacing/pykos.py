# -*- coding: utf-8 -*-
import sys
import argparse
from pykospacing import Spacing


def get_parser():
    parser = argparse.ArgumentParser(description='Python script for automatic Korean word spacing')

    parser.add_argument('infile', type=argparse.FileType('r'),
                        default=sys.stdin)
    parser.add_argument('outfile', type=argparse.FileType('w'), nargs='?',
                        default=sys.stdout)
    parser.add_argument('-o', dest='overwrite', action='store_true', default=False,
                        help='Overwrite the result itself')

    return parser


def main(args=sys.argv[1:]):
    args = get_parser().parse_args(args)

    source = args.infile.read()
    
    result = '\n'
    spacing = Spacing()
    for line in source.splitlines():
        result += spacing(line)
        result += '\n'

    if args.overwrite:
        args.infile.close()
        with open(args.infile.name, 'w', encoding=args.infile.encoding) as f:
            f.write(result)
    else:
        args.outfile.write(result)

    return 0 if (source == result) else 1


if __name__ == '__main__':
    sys.exit(main())
