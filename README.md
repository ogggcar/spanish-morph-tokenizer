# Spanish Morph Tokenizer

Tokenizador morfológico para español, implementado en `tokenizer.py`.

## Requisitos

- Python 3.10 o posterior
- [uv](https://docs.astral.sh/uv/)

## Configuración

```bash
uv sync
```

`uv sync` crea el entorno virtual del proyecto e instala sus dependencias. Para
usar el módulo dentro del entorno:

```bash
uv run python -c "from tokenizer import MorphTokenizer"
```

La clase `MorphTokenizer` necesita `database.json`, `vocab.json` e
`invariables.json` junto a `tokenizer.py`; esos archivos de datos aún no están
incluidos en este repositorio.