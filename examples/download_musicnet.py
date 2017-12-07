import zounds
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--local-path',
        required=True,
        type=str,
        help='local path where music net csv and npz files should be stored')
    args = parser.parse_args()
    mn = zounds.MusicNet(path=args.local_path)
    for meta in mn:
        print meta