"""Start TensorBoard with its built-in plugins for Kanake runs."""

from pathlib import Path
import sys

from tensorboard import default, program


def main():
    run_dir = Path(__file__).resolve().parent / "runs"
    board = program.TensorBoard(plugins=default.get_static_plugins())
    board.configure(argv=[sys.argv[0], "--logdir", str(run_dir), *sys.argv[1:]])
    board.main()


if __name__ == "__main__":
    main()
