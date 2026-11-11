import math
from transformers import Trainer, TrainingArguments, AutoTokenizer, DataCollatorForLanguageModeling, BertConfig, BertForMaskedLM
from datasets import load_dataset
from collections import Counter
import torch
import re
import json
from pathlib import Path


import os

def load_json(path):
    with open(path, 'r', encoding="utf-8") as file:
        data = json.load(file)
    return data

def dump_json(data, path):
    with open(path, 'w', encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)

class MorphTokenizer:
    
    def __init__(self, vocab_size=31000):  # Tamaño de vocabulario configurable

        self.BASE_DIR = Path(__file__).resolve().parent.parent / "data"

        self.pre_vocab = load_json(self.BASE_DIR / "database.json")

        try:
            self.vocab = load_json(self.BASE_DIR / "vocab.json")
        except FileNotFoundError:
            raise ValueError(
                f"El archivo vocab.json no se encontró en: "
                f"{self.BASE_DIR / 'vocab.json'}"
            )

        self.vocab_size = len(self.vocab)

        self.inv_vocab = {
            v: k for k, v in self.vocab.items()
        }

        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"

        self.invariables = load_json(self.BASE_DIR / "invariables.json")

        self.prefixes = sorted(['a', 'an', 'anti', 'ante', 'archi', 'auto', 'bi', 'bis', 'biz', 'bio', 'ciber', 'circum', 'circun', 'contra', 'cuasi', 'cuatri', 'com', 'con',
    'des', 'de', 'dis', 'dia', 'ex', 'extra', 'em', 'en', 'endo', 'endó', 'entre', 'extra', 'exo', 'geo', 'hiper', 'hipo', 'inter', 'infra', 'intra',
    'in', 'im', 'i', "ir", 'mal', 'medio', 'mini', 'mili', 'mono', 'multi', 'neo', 'peri', 'perí', 'pos', 'post', 'pro', 'pre', 're', 'requete','retro', "para",
    'sub', 'so', 'super', 'sobre', 'supra', 'trans', 'tras', 'tele', 'tri', 'tetra', 'pluri', 'poli', 'semi', 'uni', 'ultra', 'vice', "un", "duo", "penta", "hexa",
    "hepta", "octo", "nona"])
        
    def add_invariables_to_database(self):
        """
        Añade las palabras invariables al pre-vocabulario
        y guarda el database.json en el directorio del tokenizer.
        """
        for word in self.invariables:
            if word not in self.pre_vocab:
                self.pre_vocab[word] = [word]

        dump_json(
            self.pre_vocab,
            self.BASE_DIR / "database.json"
        )

    def is_valid_token(self, token):
        # Excluir tokens con caracteres asiáticos
        return not re.search(r'[\u4E00-\u9FFF]', token)


    def build_vocab(self, dataset, balance=1000):

        token_frequencies_pre_vocab = Counter()
        token_frequencies_dataset = Counter()

        # --------------------------------------------------------
        # 1. Frecuencias del pre-vocabulario
        # --------------------------------------------------------
        for tokens in self.pre_vocab.values():
            valid_tokens = filter(self.is_valid_token, tokens)
            token_frequencies_pre_vocab.update(valid_tokens)

        # --------------------------------------------------------
        # 2. Frecuencias del corpus
        # --------------------------------------------------------
        for sentence in dataset:
            tokens = self.tokenize(sentence)
            valid_tokens = filter(self.is_valid_token, tokens)
            token_frequencies_dataset.update(valid_tokens)

        # --------------------------------------------------------
        # 3. Crear vocabulario
        # --------------------------------------------------------
        vocab = {}
        vocab_index = 0

        special_tokens = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]"
        ]

        for token in special_tokens:
            vocab[token] = vocab_index
            vocab_index += 1

        # --------------------------------------------------------
        # 4. Tokens morfológicos más frecuentes del pre-vocabulario
        # --------------------------------------------------------
        most_common_pre_vocab = token_frequencies_pre_vocab.most_common(balance)

        for token, _ in most_common_pre_vocab:
            if token not in vocab and len(vocab) < self.vocab_size:
                vocab[token] = vocab_index
                vocab_index += 1

        # --------------------------------------------------------
        # 5. AÑADIR INVARIABLES DIRECTAMENTE AL VOCABULARIO
        # --------------------------------------------------------
        for word in self.invariables:
            if word not in vocab and len(vocab) < self.vocab_size:
                vocab[word] = vocab_index
                vocab_index += 1

        # --------------------------------------------------------
        # 6. Puntuación
        # --------------------------------------------------------
        punctuation = list(".,-;:!?¿¡\"'()")

        for item in punctuation:
            if item not in vocab and len(vocab) < self.vocab_size:
                vocab[item] = vocab_index
                vocab_index += 1

            if f"##{item}" not in vocab and len(vocab) < self.vocab_size:
                vocab[f"##{item}"] = vocab_index
                vocab_index += 1

        # --------------------------------------------------------
        # 7. Letras y números
        # --------------------------------------------------------
        letras = (
            "a b c d e f g h i j k l m n ñ o p q r s t u v w x y z "
            "á é í ó ú ü"
        ).split()

        digitos = "0 1 2 3 4 5 6 7 8 9".split()

        letras_y_numeros = letras + digitos

        for item in letras_y_numeros:

            if item not in vocab and len(vocab) < self.vocab_size:
                vocab[item] = vocab_index
                vocab_index += 1

            if f"##{item}" not in vocab and len(vocab) < self.vocab_size:
                vocab[f"##{item}"] = vocab_index
                vocab_index += 1

        # --------------------------------------------------------
        # 8. Completar con los tokens más frecuentes del corpus
        # --------------------------------------------------------
        remaining_tokens = (
            token
            for token, _ in token_frequencies_dataset.most_common()
            if token not in vocab
        )

        for token in remaining_tokens:

            if len(vocab) >= self.vocab_size:
                break

            vocab[token] = vocab_index
            vocab_index += 1

        # --------------------------------------------------------
        # 9. Guardar vocabulario
        # --------------------------------------------------------
        dump_json(
            vocab,
            self.BASE_DIR / "vocab.json"
        )

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.inv_vocab = {v: k for k, v in vocab.items()}

        return vocab


    def remove_tildes_plural(self, word):
        tildes = 'áéíóú'
        vowels = 'aeiou'
        result = []

        for i, letter in enumerate(word):
            if letter in tildes:
                new_word = word[:i] + vowels[tildes.index(letter)] + word[i+1:]
                if new_word in self.pre_vocab:
                    tokens = list(self.pre_vocab[new_word])
                    for token in tokens[:-1]:
                        result.append(token)
                    last_without_accents = tokens[-1][:i] + letter + tokens[-1][i+1:]
                    result.append(last_without_accents)
                    break

        return result

    def remove_tildes(self, word):
        tildes = 'áéíóú'
        vowels = 'aeiou'
        result = []

        for i, letter in enumerate(word):
            if letter in tildes:
                new_word = word[:i] + vowels[tildes.index(letter)] + word[i+1:]
                if new_word in self.pre_vocab:
                    tokens = list(self.pre_vocab[new_word])
                    # Añadimos todos los tokens menos el último
                    result.extend(tokens[:-1])
                    
                    # Calculamos cuántos caracteres hemos acumulado hasta ahora (sin contar los '##')
                    previous = len(re.sub('##', '', ''.join(result)))
                    
                    # Calculamos la posición correcta para insertar la tilde en el último token
                    # Restamos el índice original de la letra con tilde en la palabra original
                    # de la longitud total ya procesada, y añadimos el desplazamiento por los '##'
                    # que tienen 2 caracteres
                    position_in_last_token = i - previous
                    if tokens[-1].startswith('##'):
                        position_in_last_token += 2  # Ajustamos por el prefijo '##'
                    
                    # Nos aseguramos de que la posición sea válida
                    position_in_last_token = max(0, min(position_in_last_token, len(tokens[-1]) - 1))
                    
                    # Reemplazamos la letra en la posición correcta
                    last_token_chars = list(tokens[-1])
                    if position_in_last_token < len(last_token_chars):
                        last_token_chars[position_in_last_token] = letter
                        result.append(''.join(last_token_chars))
                    else:
                        # Si la posición calculada está fuera de rango, simplemente añadimos el token original
                        result.append(tokens[-1])
                    break

        return result
    
    def remove_tildes(self, word):
        tildes = 'áéíóú'
        vowels = 'aeiou'
        result = []

        for i, letter in enumerate(word):
            if letter in tildes:
                new_word = word[:i] + vowels[tildes.index(letter)] + word[i+1:]
                if new_word in self.pre_vocab:
                    tokens = list(self.pre_vocab[new_word])
                    
                    # Unir todos los tokens (eliminando los ##) para tener la palabra completa
                    full_word_without_format = re.sub('##', '', ''.join(tokens))
                    
                    # Encontrar la posición de la vocal sin tilde en la palabra completa
                    target_vowel = vowels[tildes.index(letter)]
                    
                    # Buscar en la palabra completa la posición correspondiente a la vocal con tilde
                    vowel_positions = [j for j, char in enumerate(full_word_without_format) if char == target_vowel]
                    
                    # Si tenemos varias ocurrencias de la misma vocal, necesitamos saber cuál reemplazar
                    # Calculamos cuántas vocales hay antes de la posición i en la palabra original
                    vowels_before_i = sum(1 for c in word[:i] if c.lower() in 'aeiouáéíóú')
                    
                    # Elegimos la posición correcta de la vocal a reemplazar
                    if vowels_before_i < len(vowel_positions):
                        target_position = vowel_positions[vowels_before_i]
                    else:
                        # Fallback si algo sale mal
                        target_position = vowel_positions[-1] if vowel_positions else 0
                    
                    # Ahora reconstruimos los tokens con la tilde en la posición correcta
                    current_pos = 0
                    modified_tokens = []
                    
                    for token in tokens:
                        token_text = token.replace('##', '')
                        token_len = len(token_text)
                        
                        # Si la posición objetivo está en este token
                        if current_pos <= target_position < current_pos + token_len:
                            relative_pos = target_position - current_pos
                            
                            # Ajustar la posición si el token tiene prefijo ##
                            if token.startswith('##'):
                                char_list = list(token)
                                char_list[relative_pos + 2] = letter  # +2 por el prefijo '##'
                                modified_token = ''.join(char_list)
                            else:
                                char_list = list(token)
                                char_list[relative_pos] = letter
                                modified_token = ''.join(char_list)
                            
                            modified_tokens.append(modified_token)
                        else:
                            modified_tokens.append(token)
                        
                        current_pos += token_len
                    
                    return modified_tokens
                    
        return result
    
    def remove_tildes(self, word):
        """
        Convierte una palabra con tilde en sus tokens, colocando la tilde en el token adecuado.
        """
        tildes = 'áéíóú'
        vowels = 'aeiou'
        
        modified_tokens = []
        
        for i, letter in enumerate(word):
            if letter in tildes:
                # Encontrar la palabra sin tilde
                vowel_without_accent = vowels[tildes.index(letter)]
                new_word = word[:i] + vowel_without_accent + word[i+1:]
                
                if new_word in self.pre_vocab:
                    # Obtener los tokens de la palabra sin tilde
                    original_tokens = list(self.pre_vocab[new_word])
                    
                    # Reconstruir la palabra para encontrar dónde va la tilde
                    full_word = ""
                    token_boundaries = []  # Para rastrear dónde comienza cada token
                    
                    for token in original_tokens:
                        clean_token = token.replace('##', '')
                        token_boundaries.append(len(full_word))
                        full_word += clean_token
                    
                    # Encontrar qué token contiene la letra con tilde
                    accented_pos = i
                    containing_token_idx = None
                    position_in_token = None
                    
                    for j in range(len(token_boundaries)):
                        start_pos = token_boundaries[j]
                        end_pos = len(full_word) if j == len(token_boundaries) - 1 else token_boundaries[j + 1]
                        
                        if start_pos <= accented_pos < end_pos:
                            containing_token_idx = j
                            position_in_token = accented_pos - start_pos
                            if original_tokens[j].startswith('##'):
                                position_in_token += 2  # Ajuste por el prefijo '##'
                            break
                    
                    # Crear nuevos tokens con la tilde en la posición correcta
                    modified_tokens = []
                    
                    for j, token in enumerate(original_tokens):
                        if j == containing_token_idx:
                            chars = list(token)
                            if position_in_token < len(chars):
                                chars[position_in_token] = letter
                                modified_token = ''.join(chars)
                                modified_tokens.append(modified_token)
                            else:
                                # La posición está fuera de rango, mantener el token original
                                modified_tokens.append(token)
                        else:
                            modified_tokens.append(token)
                    
                    return modified_tokens
        
        if len(modified_tokens) == 0:
            return []

    def add_tildes(self, word):
            tildes = 'áéíóú'
            vowels = 'aeiou'
            result = []

            for i, letter in enumerate(word):
                if letter in vowels:
                    new_word = word[:i] + tildes[vowels.index(letter)] + word[i+1:]
                    if new_word in self.pre_vocab:
                        tokens = list(self.pre_vocab[new_word])
                        for token in tokens:
                            accented_to_regular = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u'}
                            word_without_accents = ''.join(accented_to_regular.get(char, char) for char in token)
                            result.append(word_without_accents)
                        break

            return result



    def tokenize(self, text):

        final_tokens = []

        pattern = r"\w+|[^\w\s]"
        doc = re.findall(pattern, text)
        for token in doc:
            tokens = []
            
            try:
                tokens = list(self.pre_vocab[token])
                
                # if "".join(tokens).replace("##", "") != token:
                #     resto = token - "".join(tokens).replace("##", "")
                #     tokens.insert(0, resto)
                #     print(token, tokens)
                
            except:
                
                try:
                    if token.endswith('s') and token[:-1] in self.pre_vocab:
                        tokens = list(self.pre_vocab[token[:-1]])
                        tokens.append('##s')

                    elif token.endswith('as') and (token[:-2] in self.pre_vocab or token[:-2]+'o' in self.pre_vocab):
                        if token[:-2] in self.pre_vocab and token[-3] != 'd':
                            tokens = list(self.pre_vocab[token[:-2]])
                            tokens.append('##a')
                            tokens.append('##s')
                        elif token[:-2]+'o' in self.pre_vocab:
                            tokens = list(self.pre_vocab[token[:-2]+'o'])[:-1]
                            tokens.append('##a')  
                            tokens.append('##s')   
                    elif token.endswith('onas')  and token[:-4]+'ón' in self.pre_vocab:
                        tokens = list(self.pre_vocab[token[:-4]+'ón'])[:-1]
                        tokens.append('##on')
                        tokens.append('##a')
                        tokens.append('##s')
                        

                    elif token.endswith('sas'):
                        tokens = self.add_tildes(token[:-2])
                        if len(tokens) > 0:
                            tokens.append('##a')
                            tokens.append('##s')
                        elif len(tokens) == 0:
                            tokens = list(self.pre_vocab[token[:-3]])
                            tokens.append('##sa')
                            tokens.append('##s')
                    

                    elif token.endswith('es'):
                        if token[:-2] in self.pre_vocab:
                            tokens = list(self.pre_vocab[token[:-2]])
                            tokens.append('##es')
                        else:
                            if token[:-2].endswith('c') and token[:-3]+'z' in self.pre_vocab:
                                tokens = list(self.pre_vocab[token[:-3]+'z'])
                                tokens[-1] = tokens[-1][:-1]+'c'
                                tokens.append('##es')
                            else:
                                tokens = self.add_tildes(token[:-2])
                                if len(tokens) > 0:
                                    tokens.append('##es')
                                else:
                                    tokens = self.remove_tildes(token[:-2])
                                    if len(tokens) > 0:
                                        tokens.append('##es')
                            
                    elif token.endswith('a') and (token[:-1] in self.pre_vocab or token[:-1]+'o' in self.pre_vocab or token[:-1]+'e' in self.pre_vocab):
                        if token[:-1]+'o' in self.pre_vocab:
                            tokens = list(self.pre_vocab[token[:-1]+'o'])[:-1]
                            tokens.append('##a')
                        elif token[:-1] in self.pre_vocab:
                            tokens = list(self.pre_vocab[token[:-1]])
                            tokens.append('##a')
                        if token[:-1]+'e' in self.pre_vocab:
                            tokens = list(self.pre_vocab[token[:-1]+'e'])[:-1]
                            tokens.append('##a')
                    
                    elif token.endswith('ona')  and token[:-3]+'ón' in self.pre_vocab:
                        tokens = list(self.pre_vocab[token[:-3]+'ón'])[:-1]
                        tokens.append('##on')
                        tokens.append('##a')

                    elif token.endswith('sa'):
                        tokens = self.add_tildes(token[:-1])
                        if len(tokens) > 0:
                            tokens.append('##a')
                        elif len(tokens) == 0:
                            tokens = list(self.pre_vocab[token[:-2]])
                            tokens.append('##sa')
                    
                    elif token.endswith('mente') and token[:-5] in self.pre_vocab and token[-6] != 'a':
                        
                        tokens = list(self.pre_vocab[token[:-5]])
                        tokens.append('##mente')

                    elif token.endswith('mente') and token[-6] == 'a' and token[:-6]+'o' in self.pre_vocab:
                        tokens = list(self.pre_vocab[token[:-6]+'o'])[:-1]
                        tokens.append('##a')
                        tokens.append('##mente')
                    
                    elif token.endswith('mente') and token[-6] == 'a' and token[:-6] in self.pre_vocab:
                        
                        tokens = list(self.pre_vocab[token[:-6]])
                        tokens.append('##mente')

                    
                    elif ((token.endswith('la') or token.endswith('lo')) and token[-3] != 'l') or ((token.endswith('las') or token.endswith('los')) and token[-4] != 'l'):
                        
                        if not token.endswith('s'):
                            if not token[:-2].endswith('se'):

                                try:
                                    tokens = list(self.pre_vocab[token[:-2]])
                                    tokens.append('##'+token[-2:])
                                except:
                                
                                    tokens = self.remove_tildes(token[:-2])
                                    if len(tokens) > 0:
                                        tokens.append('##'+token[-2:])

                                
                            else:
                                if token[:-2].endswith('se'):
                                    tokens = self.remove_tildes(token[:-4])
                                    if len(tokens) > 0:
                                        tokens.append('##se')
                                        tokens.append('##'+token[-2:])

                        else: 
                            if not token[:-3].endswith('se'):

                                try:
                                    tokens = list(self.pre_vocab[token[:-3]])
                                    tokens.append('##'+token[-3:])
                                except:
                                    tokens = self.remove_tildes(token[:-3])
                                    if len(tokens) > 0:
                                        tokens.append('##'+token[-3:])

                            else:
                                if token[:-3].endswith('se'):
                                    tokens = self.remove_tildes(token[:-5])
                                    if len(tokens) > 0:
                                        tokens.append('##se')
                                        tokens.append('##'+token[-3:])

                    elif token.endswith('les') or token.endswith('nos'): # cuidado plurales
                        try:
                            tokens = list(self.pre_vocab[token[:-3]])
                            tokens.append('##'+token[-3:])   
                        except:
                            tokens = self.remove_tildes(token[:-3])
                            if len(tokens) > 0:
                                tokens.append('##'+token[-3:])      


                    elif token.endswith('le') or token.endswith('os') or token.endswith('me') or token.endswith('te') or token.endswith('se'):
                        try:
                            tokens = list(self.pre_vocab[token[:-2]])
                            tokens.append('##'+token[-2:])   
                        except:
                            tokens = self.remove_tildes(token[:-2])
                            if len(tokens) > 0:
                                tokens.append('##'+token[-2:])

                                    
                        
                    
                    
                    else:
                        for suffix in ['it', 'ill', 'ic', 'in', 'ín', 'ísim', 'ot', 'et']:
                            if suffix in token[-5:]:
                                tokens = token.split(suffix)
                                tokens = tokens[:-1]+[suffix]+[tokens[-1]]
                                tokens = [t for t in tokens if len(t) > 0]
                                tokens = ["##" + t if i > 0 else t for i, t in enumerate(tokens)]
                                if tokens[-1] == '##as':
                                    tokens = tokens[:-1]
                                    tokens.append('##a')
                                    tokens.append('##s')
                                break
                    
                    
                    if type(tokens) == type(None) or len(tokens) == 0:
                        # Primera pasada: buscar prefijos en la palabra original
                        for prefix in self.prefixes:
                            if token.startswith(prefix):
                                remaining = token[len(prefix):]
                                if remaining in self.pre_vocab:
                                    # Si encontramos el prefijo y la palabra restante está en el vocabulario
                                    remaining_tokens = list(self.pre_vocab[remaining])
                                    # Añadir '##' a todos los tokens después del prefijo
                                    remaining_tokens = ["##" + t if t[0] != '#' else t for t in remaining_tokens]
                                    tokens = [prefix] + remaining_tokens
                                    break
                        
                        # Segunda pasada: si no se encontró nada y la palabra termina en 's', intentar sin la 's'
                        if (type(tokens) == type(None) or len(tokens) == 0) and token.endswith('s'):
                            token_without_s = token[:-1]
                            for prefix in self.prefixes:
                                if token_without_s.startswith(prefix):
                                    remaining = token_without_s[len(prefix):]
                                    if remaining in self.pre_vocab:
                                        # Si encontramos el prefijo y la palabra restante está en el vocabulario
                                        remaining_tokens = list(self.pre_vocab[remaining])
                                        # Añadir '##' a todos los tokens después del prefijo
                                        remaining_tokens = ["##" + t if t[0] != '#' else t for t in remaining_tokens]
                                        tokens = [prefix] + remaining_tokens + ["##s"]
                                        break
                        
                        # Tercera pasada: si no se encontró nada y la palabra termina en 'a', cambiar por 'o' y buscar
                        if (type(tokens) == type(None) or len(tokens) == 0) and token.endswith('a'):
                            token_with_o = token[:-1] + 'o'
                            for prefix in self.prefixes:
                                if token_with_o.startswith(prefix):
                                    remaining = token_with_o[len(prefix):]
                                    if remaining in self.pre_vocab:
                                        # Si encontramos el prefijo y la palabra restante está en el vocabulario
                                        remaining_tokens = list(self.pre_vocab[remaining])
                                        # Añadir '##' a todos los tokens después del prefijo
                                        remaining_tokens = ["##" + t if t[0] != '#' else t for t in remaining_tokens]
                                        tokens = [prefix] + remaining_tokens[:-1] + ["##a"]
                                        break

                    if type(tokens) == type(None) or len(tokens) == 0:
                        tokens = list(token)
                        tokens = ["##" + t if i > 0 else t for i, t in enumerate(tokens)]
                except:
                    tokens = list(token)
                    tokens = ["##" + t if i > 0 else t for i, t in enumerate(tokens)]

            final_tokens = final_tokens + tokens     

        return final_tokens



    def convert_tokens_to_ids(self, tokens):
        token_ids = []
        for token in tokens:
            # Maneja subtokens si es necesario
            token = token.lstrip("##")  # Elimina los prefijos '##' si usas WordPiece
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab.get('[UNK]', 0))  # ID para tokens desconocidos
        return token_ids

    # Método para obtener el vocabulario
    def get_vocab(self):
        return self.vocab



    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab["[UNK]"])

    def _convert_id_to_token(self, index):
        return self.inv_vocab.get(index, "[UNK]")

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)

    def build_inputs_with_special_tokens(self, token_ids):
        return [self.vocab["[CLS]"]] + token_ids + [self.vocab["[SEP]"]]
    
    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):  # Si es un único token (cadena)
            return self.vocab.get(tokens, self.vocab.get('[UNK]', 0))  # Devuelve el ID del token
        elif isinstance(tokens, list):  # Si es una lista de tokens
            return [self.vocab.get(token, self.vocab.get('[UNK]', 0)) for token in tokens]
        else:
            raise ValueError("El argumento `tokens` debe ser una cadena o una lista de cadenas.")
    
    def __len__(self):
        return self.vocab_size





    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        vocab = load_json(f'{pretrained_model_name_or_path}/vocab.json')
        return cls(vocab=vocab)

    @property
    def pad_token_id(self):
        return self.vocab[self.pad_token]

    @property
    def unk_token_id(self):
        return self.vocab[self.unk_token]

    @property
    def cls_token_id(self):
        return self.vocab[self.cls_token]

    @property
    def sep_token_id(self):
        return self.vocab[self.sep_token]

    @property
    def mask_token_id(self):
        return self.vocab[self.mask_token]

    # Método para obtener el vocabulario
    def get_vocab(self):
        return self.vocab

    def pad(self, encoded_inputs, padding=True, max_length=None, return_tensors=None, pad_to_multiple_of=None):
        # Verificar si `encoded_inputs` es una lista de secuencias o un diccionario
        if isinstance(encoded_inputs, list):
            # Convertir la lista en un diccionario con la clave `input_ids`
            encoded_inputs = {"input_ids": encoded_inputs}

        # Determinar `max_length` en función de las secuencias si no se especifica
        if padding and max_length is None:
            max_length = max(len(seq['input_ids']) for seq in encoded_inputs['input_ids'])

        # Ajustar `max_length` para que sea un múltiplo de `pad_to_multiple_of`, si está especificado
        if pad_to_multiple_of is not None and max_length % pad_to_multiple_of != 0:
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        # Aplicar padding a cada secuencia en `encoded_inputs`
        padded_inputs = {
            "input_ids": [],
            "attention_mask": []
        }

        for item in encoded_inputs['input_ids']:
            # Acceder a la secuencia de tokens dentro de cada diccionario
            seq = item['input_ids']
            seq_padded = seq + [self.pad_token_id] * (max_length - len(seq))
            attention_mask = [1] * len(seq) + [0] * (max_length - len(seq))
            padded_inputs["input_ids"].append(seq_padded)
            padded_inputs["attention_mask"].append(attention_mask)

        # Convertir a tensores si `return_tensors` está especificado
        if return_tensors == "pt":
            padded_inputs = {key: torch.tensor(val) for key, val in padded_inputs.items()}
        elif return_tensors == "tf":
            import tensorflow as tf
            padded_inputs = {key: tf.constant(val) for key, val in padded_inputs.items()}
        elif return_tensors == "np":
            import numpy as np
            padded_inputs = {key: np.array(val) for key, val in padded_inputs.items()}

        return padded_inputs

    
    def get_special_tokens_mask(self, token_ids, already_has_special_tokens=False):
        """
        Return a mask indicating which tokens are special.
        Args:
            token_ids (list[int]): Token IDs in the sequence.
            already_has_special_tokens (bool): Whether the input contains special tokens.
        Returns:
            List[int]: Mask with 1 for special tokens and 0 otherwise.
        """
        if not already_has_special_tokens:
            raise ValueError("The tokenizer assumes input sequences already contain special tokens.")
        
        # Obtener los IDs de los tokens especiales
        special_token_ids = {
            self.pad_token_id,
            self.unk_token_id,
            self.cls_token_id,
            self.sep_token_id,
            self.mask_token_id
        }

        # Crear la máscara
        return [1 if token_id in special_token_ids else 0 for token_id in token_ids]

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        vocab_path = os.path.join(save_directory, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f)
        print(f"Vocabulario guardado en {vocab_path}")

    @classmethod
    def from_pretrained(cls, save_directory, **kwargs):
        vocab_path = os.path.join(save_directory, "vocab.json")
        return cls(vocab_path, **kwargs)


    

    


    

    def build_inputs_with_special_tokens1(self, token_ids1):
        return [self.vocab["[CLS]"]] + token_ids1 + [self.vocab["[SEP]"]]

    def build_inputs_with_special_tokens2(self, token_ids1, token_ids2):
        return [self.vocab["[CLS]"]] + token_ids1 + [self.vocab["[SEP]"]] + token_ids2 + [self.vocab["[SEP]"]]

    def add_tokens(self, new_tokens):
        """
        Añade nuevos tokens al vocabulario.

        Args:
            new_tokens (list[str]): Lista de tokens a añadir al vocabulario.

        Returns:
            int: Número de tokens añadidos al vocabulario.
        """
        added_count = 0
        for token in new_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)  # Asigna un nuevo índice
                added_count += 1

        # Actualiza el tamaño del vocabulario
        self.vocab_size = len(self.vocab)
        return added_count
