# """ from https://github.com/keithito/tacotron """

# '''
# Cleaners are transformations that run over the input text at both training and eval time.

# Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
# hyperparameter. Some cleaners are English-specific. You'll typically want to use:
#   1. "english_cleaners" for English text
#   2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
#      the Unidecode library (https://pypi.python.org/pypi/Unidecode)
#   3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
#      the symbols in symbols.py to match your data).
# '''

# import re
# from unidecode import unidecode
from .numbers import normalize_numbers


# # Regular expression matching whitespace:
# _whitespace_re = re.compile(r'\s+')

# # List of (regular expression, replacement) pairs for abbreviations:
# _abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
#   ('mrs', 'misess'),
#   ('mr', 'mister'),
#   ('dr', 'doctor'),
#   ('st', 'saint'),
#   ('co', 'company'),
#   ('jr', 'junior'),
#   ('maj', 'major'),
#   ('gen', 'general'),
#   ('drs', 'doctors'),
#   ('rev', 'reverend'),
#   ('lt', 'lieutenant'),
#   ('hon', 'honorable'),
#   ('sgt', 'sergeant'),
#   ('capt', 'captain'),
#   ('esq', 'esquire'),
#   ('ltd', 'limited'),
#   ('col', 'colonel'),
#   ('ft', 'fort'),
# ]]


# def expand_abbreviations(text):
#   for regex, replacement in _abbreviations:
#     text = re.sub(regex, replacement, text)
#   return text


# def expand_numbers(text):
#   return normalize_numbers(text)


# def lowercase(text):
#   return text.lower()


# def collapse_whitespace(text):
#   return re.sub(_whitespace_re, ' ', text)


# def convert_to_ascii(text):
#   return unidecode(text)


# def basic_cleaners(text):
#   '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
#   text = lowercase(text)
#   text = collapse_whitespace(text)
#   return text


# def transliteration_cleaners(text):
#   '''Pipeline for non-English text that transliterates to ASCII.'''
#   text = convert_to_ascii(text)
#   text = lowercase(text)
#   text = collapse_whitespace(text)
#   return text


# def english_cleaners(text):
#   '''Pipeline for English text, including number and abbreviation expansion.'''
#   text = convert_to_ascii(text)
#   text = lowercase(text)
#   text = expand_numbers(text)
#   text = expand_abbreviations(text)
#   text = collapse_whitespace(text)
#   return text

# coding: utf-8

# Code based on https://github.com/keithito/tacotron/blob/master/text/cleaners.py
'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
    1. "english_cleaners" for English text
    2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
         the Unidecode library (https://pypi.python.org/pypi/Unidecode)
    3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
         the symbols in symbols.py to match your data).
'''

import re
from text.korean import tokenize as ko_tokenize

# Added to support LJ_speech
from unidecode import unidecode
from jamo import h2j, j2hcj
from g2pk import G2p

from hangul_romanize import Transliter
from hangul_romanize.rule import academic  # 개정 로마자 표기법

_trans = Transliter(academic)

ALLOW = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? ")

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# def korean_to_english_cleaners(text: str) -> str:
#     # 1) 로마자 변환 (예: "안녕하세요?" -> "annyeonghaseyo?")
#     roman = _trans.translit(text)

#     # 2) 공백/구두점 정리
#     roman = re.sub(r"\s+", " ", roman).strip()

#     # 3) 허용 문자만 유지 (en_symbols 범위)
#     roman = "".join(ch for ch in roman if ch in ALLOW)

#     # 4) 빈 문자열 방어
#     if not roman:
#         roman = "a"

#     return roman

# def korean_cleaners(text):
#     '''Pipeline for Korean text, including number and abbreviation expansion.'''
#     text = ko_tokenize(text) # '존경하는' --> ['ᄌ', 'ᅩ', 'ᆫ', 'ᄀ', 'ᅧ', 'ᆼ', 'ᄒ', 'ᅡ', 'ᄂ', 'ᅳ', 'ᆫ', '~']
#     return text
# 초성, 중성, 종성 자모 정의
_CHOSUNG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
_JUNGSUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
_JONGSUNG = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

# G2P 변환기 초기화
g2p = G2p()


def decompose_korean_to_jamo(text):
    """
    한글을 자모 단위로 분해
    예: "안녕" → "ㅇㅏㄴㄴㅕㅇ"
    """
    result = []
    for char in text:
        if '가' <= char <= '힣':  # 한글인 경우
            # jamo 라이브러리로 분해
            jamos = j2hcj(h2j(char))
            result.extend(list(jamos))
        else:
            result.append(char)
    return ''.join(result)


def korean_cleaners_jamo(text):
    """
    한국어 텍스트를 자모 단위로 분해하는 Cleaner
    """
    # 1. 소문자 변환
    text = text.lower()
    
    # 2. 한글을 자모로 분해
    text = decompose_korean_to_jamo(text)
    
    # 3. 공백 정리
    text = collapse_whitespace(text)
    
    return text


def korean_cleaners_romanized(text):
    """
    한국어를 발음대로 변환 후 로마자로 변환하여 영어 임베딩 활용
    
    파이프라인:
    1. 한글 텍스트 → G2P(Grapheme-to-Phoneme) → 발음 표기
    2. 발음 표기 → Romanization → 영어 알파벳
    3. 영어 알파벳 → ASCII 정규화
    """
    # 1. G2P: 한글 → 발음 표기
    # 예: "안녕하세요" → "안녕하세요" (발음 규칙 적용)
    text = g2p(text)
    
    # 2. Romanization: 한글 발음 → 로마자
    # 예: "안녕하세요" → "annyeonghaseyo"
    text = _trans.translit(text)
    
    # 3. ASCII 변환 (악센트 제거 등)
    text = convert_to_ascii(text)
    
    # 4. 소문자 변환
    text = lowercase(text)
    
    # 5. 공백 정리
    text = collapse_whitespace(text)
    
    return text
    
# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


# def expand_numbers(text):
#     return en_normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)

def convert_to_ascii(text):
    '''Converts to ascii, existed in keithito but deleted in carpedm20'''
    return unidecode(text)
    

def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    # text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text


