# Segmentador morfológico del español

Este repositorio acompaña el artículo
[«Un Segmentador Morfológico para la Generación de los Vocabularios de Grandes
Modelos de Lenguaje en Español»](articulo_construccion_segmentador.pdf), de
Óscar García-Sierra, Ana Fernández-Pampillón Cesteros y Miguel Ortega-Martín.
Incluye el segmentador morfológico final descrito en el artículo, preparado
para su uso en inferencia y en la generación de vocabularios para modelos de
lenguaje.

Las 49 condiciones morfológicas del segmentador se documentan en
[reglas_segmentador.yaml](reglas_segmentador.yaml). El fallback final que
segmenta por caracteres se describe aparte y no forma parte de ese recuento.

## Objetivo del estudio

El segmentador busca cada palabra en un índice palabra-morfemas y aplica
reglas morfológicas cuando no la encuentra, como describe el artículo. Este
repositorio incluye el índice ya preparado (`data/index.json`) y los demás
recursos necesarios para ejecutar el segmentador, pero **no incluye el proceso
de construcción ni de segmentación de ese índice**. Tampoco incluye los
conjuntos de datos ni los resultados de evaluación del artículo.

## Requisitos

- Python 3.10 o posterior
- [uv](https://docs.astral.sh/uv/)

## Procedimiento de uso

1. Instalar las dependencias con `uv sync`.
2. Cargar el índice morfológico, el vocabulario desde `data/`.
3. Pasar una frase en minúsculas al segmentador. El procesamiento conserva la
   distinción entre mayúsculas y minúsculas.

Los tres recursos de ejecución son:

- [Índice morfológico (`data/index.json`)](data/index.json)
- [Vocabulario (`data/vocab.json`)](data/vocab.json)

Para segmentar una frase desde la consola:

```bash
uv run python src/test_tokenizer.py "los gatos corren rápidamente."
```

También se puede usar la clase `MorphTokenizer` desde Python:

```python
from tokenizer import MorphTokenizer

tokenizer = MorphTokenizer()
tokens = tokenizer.tokenize("los gatos corren rápidamente.")
print(tokens)
```

## Estructura del repositorio

```text
data/
  index.json        Índice morfológico usado por el segmentador
  vocab.json        Vocabulario
reglas_segmentador.yaml
                  Condiciones morfológicas de MorphTokenizer.tokenize
src/
  tokenizer.py      Implementación del segmentador
  test_tokenizer.py Interfaz de prueba por consola
```

## Evaluación y errores

Los conjuntos de datos de evaluación, los resultados desglosados y agregados,
y los archivos de errores empleados para el análisis **no forman parte de esta
versión del repositorio**.
