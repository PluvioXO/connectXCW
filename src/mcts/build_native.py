"""Build the optional native tree kernel: python -m src.mcts.build_native."""
import os
from pathlib import Path
import subprocess


def main():
    folder = Path(__file__).parent
    output = folder / '_native_search.so'
    temporary = output.with_suffix(f'.{os.getpid()}.tmp')
    try:
        subprocess.run(['g++', '-O3', '-std=c++17', '-shared', '-fPIC',
                        '-fno-fast-math', '-ffp-contract=off', str(folder / 'native_search.cpp'),
                        '-o', str(temporary)], check=True)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(output)


if __name__ == '__main__':
    main()
