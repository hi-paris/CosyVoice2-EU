# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import regex
chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]+')


# whether contain chinese character
def contains_chinese(text):
    return bool(chinese_char_pattern.search(text))


# whether contain french character patterns and common words
def contains_french(text):
    # French-specific characters
    french_chars = re.compile(r'[àâäéèêëïîôùûüÿç]')
    # Common French words/patterns (more comprehensive)
    french_words = re.compile(r'\b(le|la|les|un|une|des|du|de|et|est|avec|dans|pour|sur|par|ce|cette|qui|que|dont|où|si|mais|ou|donc|car|ni|or|je|tu|il|elle|nous|vous|ils|elles|mon|ma|mes|ton|ta|tes|son|sa|ses|notre|votre|leur|leurs|bonjour|bonsoir|merci|salut|français|habite|appelle|travaille)\b', re.IGNORECASE)
    
    # Check for French characters or multiple French words
    has_french_chars = bool(french_chars.search(text))
    french_word_matches = french_words.findall(text.lower())
    has_multiple_french_words = len(french_word_matches) >= 2
    
    return has_french_chars or has_multiple_french_words


# replace special symbol
def replace_corner_mark(text):
    text = text.replace('²', '平方')
    text = text.replace('³', '立方')
    return text


# remove meaningless symbol
def remove_bracket(text):
    text = text.replace('（', '').replace('）', '')
    text = text.replace('【', '').replace('】', '')
    text = text.replace('`', '').replace('`', '')
    text = text.replace("——", " ")
    return text


# spell Arabic numerals
def spell_out_number(text: str, inflect_parser):
    new_text = []
    st = None
    for i, c in enumerate(text):
        if not c.isdigit():
            if st is not None:
                num_str = inflect_parser.number_to_words(text[st: i])
                new_text.append(num_str)
                st = None
            new_text.append(c)
        else:
            if st is None:
                st = i
    if st is not None and st < len(text):
        num_str = inflect_parser.number_to_words(text[st:])
        new_text.append(num_str)
    return ''.join(new_text)


# spell Arabic numerals for French
def spell_out_number_french(text: str):
    try:
        import num2words
        
        def convert_number(match):
            number = match.group()
            return num2words.num2words(int(number), lang='fr')
        
        # Convert standalone numbers
        text = re.sub(r'\b\d+\b', convert_number, text)
        return text
    except ImportError:
        # Fallback to basic replacement if num2words not available
        return text


# French-specific symbol replacements
def replace_symbols_french(text: str):
    text = text.replace('&', ' et ')
    text = text.replace('@', ' arobase ')
    text = text.replace('%', ' pour cent ')
    text = text.replace('#', ' dièse ')
    text = text.replace('$', ' dollar ')
    text = text.replace('€', ' euros ')
    text = text.replace('£', ' livres ')
    text = text.replace('°', ' degrés ')
    text = text.replace('+', ' plus ')
    text = text.replace('=', ' égal ')
    return text


# French abbreviation expansion
def expand_abbreviations_french(text: str):
    abbreviations = {
        r'\bM\.': 'monsieur',
        r'\bMme\.?': 'madame',
        r'\bMlle\.?': 'mademoiselle',
        r'\bDr\.': 'docteur',
        r'\bPr\.': 'professeur', 
        r'\bSt\.': 'saint',
        r'\bCie\.?': 'compagnie',
        r'\betc\.': 'et cetera',
        r'\bc-à-d\.?': 'c\'est-à-dire',
        r'\bp\.ex\.': 'par exemple',
        r'\bav\.': 'avenue',
        r'\bbd\.?': 'boulevard',
        r'\bpl\.': 'place',
        r'\brue\.': 'rue',
    }
    
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text


# split paragrah logic：
# 1. per sentence max len token_max_n, min len token_min_n, merge if last sentence len less than merge_len
# 2. cal sentence len according to lang
# 3. split sentence according to puncatation
def split_paragraph(text: str, tokenize, lang="zh", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False):
    def calc_utt_length(_text: str):
        if lang == "zh":
            return len(_text)
        else:
            return len(tokenize(_text))

    def should_merge(_text: str):
        if lang == "zh":
            return len(_text) < merge_len
        else:
            return len(tokenize(_text)) < merge_len

    if lang == "zh":
        pounc = ['。', '？', '！', '；', '：', '、', '.', '?', '!', ';']
    else:
        pounc = ['.', '?', '!', ';', ':']
    if comma_split:
        pounc.extend(['，', ','])

    if text[-1] not in pounc:
        if lang == "zh":
            text += "。"
        else:
            text += "."

    st = 0
    utts = []
    for i, c in enumerate(text):
        if c in pounc:
            if len(text[st: i]) > 0:
                utts.append(text[st: i] + c)
            if i + 1 < len(text) and text[i + 1] in ['"', '”']:
                tmp = utts.pop(-1)
                utts.append(tmp + text[i + 1])
                st = i + 2
            else:
                st = i + 1

    final_utts = []
    cur_utt = ""
    for utt in utts:
        if calc_utt_length(cur_utt + utt) > token_max_n and calc_utt_length(cur_utt) > token_min_n:
            final_utts.append(cur_utt)
            cur_utt = ""
        cur_utt = cur_utt + utt
    if len(cur_utt) > 0:
        if should_merge(cur_utt) and len(final_utts) != 0:
            final_utts[-1] = final_utts[-1] + cur_utt
        else:
            final_utts.append(cur_utt)

    return final_utts


# remove blank between chinese character
def replace_blank(text: str):
    out_str = []
    for i, c in enumerate(text):
        if c == " ":
            if ((text[i + 1].isascii() and text[i + 1] != " ") and
                    (text[i - 1].isascii() and text[i - 1] != " ")):
                out_str.append(c)
        else:
            out_str.append(c)
    return "".join(out_str)


def is_only_punctuation(text):
    # Regular expression: Match strings that consist only of punctuation marks or are empty.
    punctuation_pattern = r'^[\p{P}\p{S}]*$'
    return bool(regex.fullmatch(punctuation_pattern, text))
