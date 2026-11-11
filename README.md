# Segmentador morfológico del español

Este repositorio corresponde al **capítulo 10 de la tesis doctoral de Óscar
García Sierra** y contiene el segmentador morfológico final para español,
preparado para su uso en inferencia. Se relaciona con el artículo
[publicado en Dialnet](https://dialnet.unirioja.es/servlet/articulo?codigo=10413927).

## Objetivo del estudio

Proporcionar el segmentador morfológico final descrito en el capítulo 10 y los
archivos que necesita para segmentar texto en español. En esta versión se
publica el segmentador y sus recursos de ejecución; no se incluyen los
conjuntos de datos ni los resultados de evaluación del capítulo.

## Procedimiento de uso

1. Instalar las dependencias con `uv sync`.
2. Cargar el índice morfológico, el vocabulario y las palabras invariables
   desde `data/`.
3. Pasar una frase en minúsculas al segmentador. El procesamiento conserva la
   distinción entre mayúsculas y minúsculas.

Los tres recursos de ejecución son:

- [Índice morfológico (`data/index.json`)](data/index.json)
- [Vocabulario (`data/vocab.json`)](data/vocab.json)
- [Palabras invariables (`data/invariables.json`)](data/invariables.json)

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
  vocab.json        Vocabulario y correspondencia con los identificadores
  invariables.json  Palabras que se incorporan sin segmentación
src/
  tokenizer.py      Implementación del segmentador
  test_tokenizer.py Interfaz de prueba por consola
```

## Evaluación y errores

Los conjuntos de datos de evaluación, los resultados desglosados y agregados,
y los archivos de errores empleados para el análisis **no forman parte de esta
versión del repositorio**. Se incorporarán y documentarán aquí cuando se
publique el material de evaluación del capítulo 10.
