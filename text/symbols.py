# # """ from https://github.com/keithito/tacotron """

# # '''
# # Defines the set of symbols used in text input to the model.

# # The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
# # from text import cmudict

# # _pad        = '_'
# # _punctuation = '!\'(),.:;? '
# # _special = '-'
# # _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# # # Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
# # _arpabet = ['@' + s for s in cmudict.valid_symbols]

# # # Export all symbols:
# # symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet
# # coding: utf-8
# '''
# Defines the set of symbols used in text input to the model.

# The default is a set of ASCII characters that works well for English or text that has been run
# through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
# '''
# from jamo import h2j, j2h
# from jamo.jamo import _jamo_char_to_hcj

# from .korean import ALL_SYMBOLS, PAD, EOS

# # # For english
# # en_symbols = PAD+EOS+'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? '  #<-For deployment(Because korean ALL_SYMBOLS follow this convention)
# # en_symbols_list = list(en_symbols)

# # symbols = en_symbols_list # for korean

# # """
# # 초성과 종성은 같아보이지만, 다른 character이다.

# # '_~ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑ하ᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵᆨᆩᆪᆫᆬᆭᆮᆯᆰᆱᆲᆳᆴᆵᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂ!'(),-.:;? '

# # '_': 0, '~': 1, 'ᄀ': 2, 'ᄁ': 3, 'ᄂ': 4, 'ᄃ': 5, 'ᄄ': 6, 'ᄅ': 7, 'ᄆ': 8, 'ᄇ': 9, 'ᄈ': 10, 
# # 'ᄉ': 11, 'ᄊ': 12, 'ᄋ': 13, 'ᄌ': 14, 'ᄍ': 15, 'ᄎ': 16, 'ᄏ': 17, 'ᄐ': 18, 'ᄑ': 19, 'ᄒ': 20, 
# # 'ᅡ': 21, 'ᅢ': 22, 'ᅣ': 23, 'ᅤ': 24, 'ᅥ': 25, 'ᅦ': 26, 'ᅧ': 27, 'ᅨ': 28, 'ᅩ': 29, 'ᅪ': 30, 
# # 'ᅫ': 31, 'ᅬ': 32, 'ᅭ': 33, 'ᅮ': 34, 'ᅯ': 35, 'ᅰ': 36, 'ᅱ': 37, 'ᅲ': 38, 'ᅳ': 39, 'ᅴ': 40, 
# # 'ᅵ': 41, 'ᆨ': 42, 'ᆩ': 43, 'ᆪ': 44, 'ᆫ': 45, 'ᆬ': 46, 'ᆭ': 47, 'ᆮ': 48, 'ᆯ': 49, 'ᆰ': 50, 
# # 'ᆱ': 51, 'ᆲ': 52, 'ᆳ': 53, 'ᆴ': 54, 'ᆵ': 55, 'ᆶ': 56, 'ᆷ': 57, 'ᆸ': 58, 'ᆹ': 59, 'ᆺ': 60, 
# # 'ᆻ': 61, 'ᆼ': 62, 'ᆽ': 63, 'ᆾ': 64, 'ᆿ': 65, 'ᇀ': 66, 'ᇁ': 67, 'ᇂ': 68, '!': 69, "'": 70, 
# # '(': 71, ')': 72, ',': 73, '-': 74, '.': 75, ':': 76, ';': 77, '?': 78, ' ': 79
# # """
# from text import cmudict

# _pad = '_'
# _punctuation = '!\'(),.:;? '
# _special = '-'

# # 방법 1: 자모 기반 (korean_cleaners_jamo 사용 시)
# _chosung = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
# _jungsung = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
# _jongsung = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

# # 자모 기반 심볼 (korean_cleaners_jamo용)
# symbols_korean_jamo = [_pad] + list(_special) + list(_punctuation) + _chosung + _jungsung + _jongsung

# # 방법 2: 로마자 기반 (korean_cleaners_romanized 사용 시)
# # 기존 영어 심볼 재사용 가능
# _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
# _arpabet = ['@' + s for s in cmudict.valid_symbols]

# # 로마자 기반 심볼 (korean_cleaners_romanized용)
# symbols_korean_romanized = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet

""" from https://github.com/keithito/tacotron """
"""
Symbol definitions for different languages and encoding schemes.
"""

from text import cmudict

# ========== 공통 심볼 ==========
_pad = '_'
_eos = '~'
_punctuation = '!\'(),.:;? '
_special = '-'


# ========== 영어 심볼 ==========
_letters_english = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'(),-.;? "  # <-- 공백 포함!


# ARPAbet 발음 기호 (영어 전용)
_arpabet = ['@' + s for s in cmudict.valid_symbols]

# 영어 심볼 세트
symbols_english = [_pad] + list(_letters_english) + _arpabet + [_eos]
# ========== 한국어 자모 심볼 ==========
# 초성 (19개)
_chosung = [
    'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 
    'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
]

# 중성 (21개)
_jungsung = [
    'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 
    'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 
    'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'
]

# 종성 (28개 - 빈 종성 포함)
_jongsung = [
    '', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 
    'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 
    'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 
    'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
]

# 한국어 자모 심볼 세트
symbols_korean_jamo = (
    [_pad] + 
    list(_special) + 
    list(_punctuation) + 
    _chosung + 
    _jungsung + 
    _jongsung + 
    [_eos]
)

# ========== 한국어 로마자 심볼 (영어 재사용) ==========
# 로마자 변환 후 영어 알파벳 사용
symbols_korean_romanized = symbols_english  # 영어 심볼 세트 재사용

# ========== 기본 심볼 세트 ==========
# 기본값: 한국어 로마자 (영어 임베딩 재사용 가능)
symbols = symbols_korean_romanized

# ========== 심볼 세트 정보 ==========
SYMBOL_SETS = {
    'english': symbols_english,
    'korean_jamo': symbols_korean_jamo,
    'korean_romanized': symbols_korean_romanized
}

# EOS 토큰
EOS = _eos

# 심볼 세트 크기 정보
print(f"Symbol sets loaded:")
print(f"  - English: {len(symbols_english)} symbols")
print(f"  - Korean Jamo: {len(symbols_korean_jamo)} symbols")
print(f"  - Korean Romanized: {len(symbols_korean_romanized)} symbols")
