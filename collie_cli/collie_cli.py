from argparse import ArgumentParser

from .config import config_command_parser
from .gen_run import gen_run_command_parser


def main():
    # TODO: add help information
    parser = ArgumentParser(
        "CoLLiE CLI",
        usage="collie <command> [<args>]",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(help="CoLLiE command helpers")

    config_command_parser(subparsers=subparsers)
    gen_run_command_parser(subparsers=subparsers)

    args = parser.parse_args()

    if not hasattr(args, "entrypoint"):
        parser.print_help()
        exit(1)

    args.entrypoint(args)


if __name__ == "__main__":
    main()
