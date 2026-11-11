"""Utilidad de consola para probar la segmentación morfológica."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Segmenta una frase con MorphTokenizer."
    )
    parser.add_argument(
        "frase",
        nargs="?",
        help="frase que se va a segmentar; si se omite, se solicitará por consola",
    )
    args = parser.parse_args()

    phrase = args.frase if args.frase is not None else input("Frase: ")

    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_files = ("database.json", "vocab.json", "invariables.json")
    missing_files = [
        name
        for name in data_files
        if not (data_dir / name).is_file()
    ]
    if missing_files:
        parser.error(
            f"faltan archivos de datos en {data_dir}: "
            + ", ".join(missing_files)
            + ". Consulta README.md."
        )

    from tokenizer import MorphTokenizer

    tokenizer = MorphTokenizer()
    print(" ".join(tokenizer.tokenize(phrase)))


if __name__ == "__main__":
    main()
