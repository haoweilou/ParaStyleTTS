from nltk.corpus import cmudict
# Example usage
import re
english_phoneme_to_ipa = {
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 
    'AY': 'aɪ', 'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð',
    'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'F': 'f', 'G': 'g',
    'HH': 'h', 'IH': 'ɪ', 'IY': 'i', 'JH': 'dʒ', 'K': 'k',
    'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ',
    'OY': 'ɔɪ', 'P': 'p', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ',
    'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v',
    'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'
}
# ipa_pho_dict = {'EMPTY': 0, 'a': 1, 'an': 2, 'aɪ': 3, 'aʊ': 4, 'b': 5, 'd': 6, 'dʒ': 7, 'eɪ': 8, 'f': 9, 'g': 10, 'h': 11, 'i': 12, 'in': 13, 'iŋ': 14, 'j': 15, 'ja': 16, 'jan': 17, 'jaʊ': 18, 'je': 19, 'joʊ': 20, 'jɑŋ': 21, 'k': 22, 'kʰ': 23, 'l': 24, 'm': 25, 'n': 26, 'o': 27, 'oʊ': 28, 'p': 29, 'pʰ': 30, 's': 31, 't': 32, 'ts': 33, 'tsʰ': 34, 'tɕ': 35, 'tɕʰ': 36, 'tʃ': 37, 'tʰ': 38, 'u': 39, 'v': 40, 'w': 41, 'wa': 42, 'wan': 43, 'waɪ': 44, 'weɪ': 45, 'wo': 46, 'wɑŋ': 47, 'wən': 48, 'x': 49, 'y': 50, 'yan': 51, 'yn': 52, 'yɛ': 53, 'z': 54, 'æ': 55, 'ð': 56, 'ŋ': 57, 'ɑ': 58, 'ɑŋ': 59, 'ɔ': 60, 'ɔɪ': 61, 'ɕ': 62, 'ən': 63, 'əŋ': 64, 'ɚ': 65, 'ɛ': 66, 'ɝ': 67, 'ɤ': 68, 'ɪ': 69, 'ɹ': 70, 'ɻ': 71, 'ʂ': 72, 'ʃ': 73, 'ʈʂ': 74, 'ʈʂʰ': 75, 'ʊ': 76, 'ʊŋ': 77, 'ʌ': 78, 'ʒ': 79, 'θ': 80, '|': 81, '&': 82, 'START': 83, 'END': 84}


pronouncing_dict = cmudict.dict()
ipa_to_word = {}

def word_to_phoneme(word):
    word_lower = word.lower()  # CMU Dict uses lowercase words
    if word_lower in pronouncing_dict:
        # Return the first pronunciation for simplicity
        return pronouncing_dict[word_lower][0]
    else:
        return None  # Word not found in the dictionary

def phoneme_to_ipa(phonemes):
    return [english_phoneme_to_ipa.get(p) for p in phonemes]

def pho_stress_split(phonemes):
    stress = []
    phoneme = []
    for pho in phonemes:
        last = pho[-1]
        if str.isnumeric(last): 
            stress.append(int(last))
            phoneme.append(pho[:-1])
        else: 
            stress.append(0)
            phoneme.append(pho)
    return phoneme,stress


def word_to_ipa(word):
    #vowel contain 0,1,2 to symbolize the stress
    #0, no stress, 1 primary stress, 2 secondary stress
    global ipa_to_word
    if word == "|": return ["|"],[0]
    elif word == '’':return ["’"],[0]

    phonemes = word_to_phoneme(word)
    phoneme, stress = pho_stress_split(phonemes)
    ipa_phoneme = phoneme_to_ipa(phoneme)
    if "".join(ipa_phoneme) not in ipa_to_word:
        ipa_to_word["".join(ipa_phoneme)] = word
    return ipa_phoneme, stress

def normalize_sentence(sentence):
    words = sentence.split(" ")
    output = []
    for word in words: 
        if word in pronouncing_dict:
            output.append([word])
        else: 
            output.append(split_into_words(word,pronouncing_dict))
    return output
    
def normalize_sentence_with_seg(sentence):
    words = sentence.split(" ")
    output = []
    for word in words: 
        if word in pronouncing_dict or word == "|":
            output.append([word])
        else: 
            output.append(split_into_words(word,pronouncing_dict))
    return output

def split_into_words(word, word_dictionary):
    word = word.lower()
    result = []
    i = 0
    while i < len(word):
        found = False
        for j in range(len(word), i, -1): 
            segment = word[i:j]
            if segment in word_dictionary:
                result.append(segment)
                i = j - 1
                found = True
                break
        if not found:  
            result.append(word[i])
        i += 1
    return result



import unicodedata
#phoneme, number of ipa phoneme
#stress, 3, 0: no stress, 1 for primary stress, 2 for non-stress, mostly applied to vowel
#tone
def english_to_ipa(sentence:str):
    normalized = unicodedata.normalize('NFKD', sentence)
    sentence = ''.join([c for c in normalized if not unicodedata.combining(c)])
    sentence = normalize_sentence(sentence)
    ipa_phonemes = ["START"]
    stresses = [0]
    for word in sentence:
        stress = []
        word_phoneme = []
        for subword in word:
            if subword in symbols:
                ipa = ["&"] #symbol indicate segmentation
                s = [0]
            else: 
                ipa, s = word_to_ipa(subword)
                ipa.append("|")
                s.append(0)
            word_phoneme += ipa
            stress += s

        ipa_phonemes += word_phoneme
        stresses += stress
    ipa_phonemes[-1] = "END"
    return ipa_phonemes, stresses




pinyin_to_ipa = {
    "b": "p", "p": "pʰ", "m": "m", "f": "f", 
    "d": "t", "t": "tʰ", "n": "n", "l": "l",
    "g": "k", "k": "kʰ", "h": "x",
    "j": "tɕ", "q": "tɕʰ", "x": "ɕ",
    "zh": "ʈʂ", "ch": "ʈʂʰ", "sh": "ʂ", "r": "ɻ",
    "z": "ts", "c": "tsʰ", "s": "s",
    "w": "w", "y": "j",
    "a": "a", "o": "o", "e": "ɤ", "i": "i", "u": "u", "v": "y",
    "ai": "aɪ", "ei": "eɪ", "ao": "aʊ", "ou": "oʊ",
    "an": "an", "en": "ən", "ang": "ɑŋ", "eng": "əŋ", "ong": "ʊŋ",
    "er": "ɚ",
    "ia": "ja", "iao": "jaʊ", "ie": "je", "iu": "joʊ",
    "ian": "jan", "in": "in", "iang": "jɑŋ", "ing": "iŋ",
    "ua": "wa", "uo": "wo", "uai": "waɪ", "ui": "weɪ",
    "uan": "wan", "un": "wən", "uang": "wɑŋ", "ue": "yɛ", "van": "yan", "vn": "yn",
    "ve":"yɛ"
}

from pypinyin import pinyin, Style
def hanzi_to_pinyin(hanzi):
    return [syllable[0] for syllable in pinyin(hanzi, style=Style.TONE3)]
                                               
initials = [
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s", "w", "y"
]

finals = [
    "a", "o", "e", "i", "u", "v",
    "ai", "ei", "ao", "ou",
    "an", "en", "ang", "eng", "ong",
    "ia", "ie", "iao", "iu", "ian", "in", "iang", "ing",
    "ua", "uo", "uai", "ui", "uan", "un", "uang",
    "ve", "van", "vn",
    "er"
]

def pinyin_to_phoneme(syllable):
    initial = None
    if syllable[:2] in initials: 
        initial = syllable[:2]
        final = syllable[2:]
    elif syllable[:1] in initials:
        initial = syllable[:1]
        final = syllable[1:]
    else: 
        final = syllable
    tone = final[-1] if final[-1].isdigit() else "0"
    if final[-1].isdigit(): final = final[:-1]
    if final == "iong": 
        final = ["i","ong"]
    else: 
        final = [final]
    if "" in final: final.remove("")
    if len(final) == 0: 
        phoneme = [initial]
        tones = [int(tone)]
    else: 
        phoneme = [initial]+final if initial is not None else final
        tones = [0]*(len(phoneme)-1)+[int(tone)]
    return phoneme, tones

def pinyin_to_ipa_phoneme(syllable):
    phoneme, tones = pinyin_to_phoneme(syllable)
    ipa_phoneme = [pinyin_to_ipa.get(p) for p in phoneme]
    return ipa_phoneme, tones


def ipa_to_idx(ipa_phonemes):
    return  [ipa_pho_dict[i] for i in ipa_phonemes]   

all_ipa_phoneme = list(pinyin_to_ipa.values())+list(english_phoneme_to_ipa.values())
all_ipa_phoneme = sorted(list(set(all_ipa_phoneme)))
all_ipa_phoneme = ["EMPTY"] + all_ipa_phoneme+["|","&","START","END"]
ipa_pho_dict = {k:i for i,k  in enumerate(all_ipa_phoneme)}
idx_to_ipa = {i:k for i,k  in enumerate(all_ipa_phoneme)}

symbols = ['!', '"', "’", '(', ')', ',', '-', '.', ':', ';', '?', '[', ']', '“', '”', '…', '、', "――","”",'“']
#make update, in between different word, there should be have a seperation token to indicate word seperation
#and between symbol, there should be another token for indicate phrase/or symbol segmentation
#& in here indicate phrase-level segmentation, whih is used for symbol
def chinese_to_ipa(sentence):
    global symbols
    pinyin_sentence = hanzi_to_pinyin(sentence)
    ipa_phonemes = ["START"]
    tones = [0]
    for pinyin in pinyin_sentence:
        if pinyin == " " or pinyin in symbols:
            ipa_phonemes.append("&")
            tones.append(0)
        else: 
            ipa_phoneme, tone = pinyin_to_ipa_phoneme(pinyin)
            if None not in ipa_phoneme: 
                ipa_phonemes += ipa_phoneme
                tones += tone
            ipa_phonemes.append("|")
            tones.append(0)
    if ipa_phonemes[-1] == "|":
        ipa_phonemes[-1] = "END"
    else:
        ipa_phonemes.append("END")
        tones.append(0)
    return ipa_phonemes, tones


def check_language(text):
    if re.search(r'[a-z]', text):
        return 'en'
    elif re.search(r'[\u4e00-\u9fff]', text):
        return 'zh'
    else:
        return 'other'

def mix_to_ipa(text):
    if check_language(text) == 'en':
        ipa,style = english_to_ipa(text)
        tone = [i+5 for i in style]
        return ipa,tone
    else:
        return chinese_to_ipa(text)
    
def segment_by_language(text):
    segments = []
    if not text:
        return segments

    current_lang = check_language(text[0])
    current_seg = text[0]

    for char in text[1:]:
        lang = check_language(char)
        if lang == current_lang or char == ' ':
            current_seg += char
        else:
            segments.append((current_lang, current_seg))
            current_seg = char
            current_lang = lang

    segments.append((current_lang, current_seg))
    return segments

def mixed_sentence_to_ipa(text):
    if '£¬' in text: text = text.replace('£¬', ',')
    
    segments = segment_by_language(text)
    ipas = ['START']
    styles = [0]
    for lang, segment in segments:
        if lang == 'en':
            ipa,style = english_to_ipa(segment)
            style = [i+5 for i in style]
            ipa = ipa[1:-1]
            style = style[1:-1]
        elif lang == 'zh':
            ipa,style = chinese_to_ipa(segment)
            ipa = ipa[1:-1]
            style = style[1:-1]
        else:
            ipa = ["|"] * len(segment)
            style = [0] * len(segment)
        ipas += ipa
        styles += style
    ipas += ['END']
    styles += [0]
    return ipas,styles