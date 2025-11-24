# """ from https://github.com/keithito/tacotron """
# import re
# from text import cleaners
# from text.symbols import symbols_korean_romanized, symbols_korean_jamo
# from .symbols import symbols, EOS
# from .cleaners import korean_to_english_cleaners

# _symbol_to_id = {s: i for i, s in enumerate(symbols)}

# def text_to_sequence(text, cleaners=None):
#     text = korean_to_english_cleaners(text)
#     seq = [_symbol_to_id[s] for s in text if s in _symbol_to_id]
#     seq.append(_symbol_to_id[EOS])  # 선택
#     return seq

# # Mappings from symbol to numeric ID and vice versa:
# # _symbol_to_id = {s: i for i, s in enumerate(symbols)}
# _id_to_symbol = {i: s for i, s in enumerate(symbols)}

# # Regular expression matching text enclosed in curly braces:
# _curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


# # def text_to_sequence(text, cleaner_names):
# #   '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

# #     The text can optionally have ARPAbet sequences enclosed in curly braces embedded
# #     in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

# #     Args:
# #       text: string to convert to a sequence
# #       cleaner_names: names of the cleaner functions to run the text through

# #     Returns:
# #       List of integers corresponding to the symbols in the text
# #   '''
# #   sequence = []

# #   # Check for curly braces and treat their contents as ARPAbet:
# #   while len(text):
# #     m = _curly_re.match(text)
# #     if not m:
# #       sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
# #       break
# #     sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
# #     sequence += _arpabet_to_sequence(m.group(2))
# #     text = m.group(3)

# #   return sequence


# def sequence_to_text(sequence):
#   '''Converts a sequence of IDs back to a string'''
#   result = ''
#   for symbol_id in sequence:
#     if symbol_id in _id_to_symbol:
#       s = _id_to_symbol[symbol_id]
#       # Enclose ARPAbet back in curly braces:
#       if len(s) > 1 and s[0] == '@':
#         s = '{%s}' % s[1:]
#       result += s
#   return result.replace('}{', ' ')


# def _clean_text(text, cleaner_names):
#   for name in cleaner_names:
#     cleaner = getattr(cleaners, name)
#     if not cleaner:
#       raise Exception('Unknown cleaner: %s' % name)
#     text = cleaner(text)
#   return text


# def _symbols_to_sequence(symbols):
#   return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


# def _arpabet_to_sequence(text):
#   return _symbols_to_sequence(['@' + s for s in text.split()])


# def _should_keep_symbol(s):
#     return (s in _symbol_to_id) and (s != '_') and (s != '~')

""" from https://github.com/keithito/tacotron """
"""
Text processing module supporting multiple languages and symbol sets.
Supports: English, Korean (Jamo), Korean (Romanized)
"""

import re
from text import cleaners
from text.symbols import (
    symbols_english,
    symbols_korean_jamo, 
    symbols_korean_romanized
)

# 사용할 심볼 세트 선택 (설정에 따라 변경)
# 옵션: 'english', 'korean_jamo', 'korean_romanized'
SYMBOL_SET = 'korean_jamo'  # 기본값: 로마자 방식

# 심볼 세트 매핑
_symbol_sets = {
    'english': symbols_english,
    'korean_jamo': symbols_korean_jamo,
    'korean_romanized': symbols_korean_romanized
}

# 현재 사용 중인 심볼 세트
symbols = _symbol_sets[SYMBOL_SET]

# EOS 토큰 (End Of Sequence)
EOS = '~'

# 심볼 <-> ID 매핑
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# 중괄호 내 ARPAbet 매칭용 정규식
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def set_symbol_set(symbol_set_name):
    """
    사용할 심볼 세트 변경
    
    Args:
        symbol_set_name: 'english', 'korean_jamo', 'korean_romanized' 중 하나
    """
    global symbols, _symbol_to_id, _id_to_symbol, SYMBOL_SET
    
    if symbol_set_name not in _symbol_sets:
        raise ValueError(f"Unknown symbol set: {symbol_set_name}. "
                        f"Available: {list(_symbol_sets.keys())}")
    
    SYMBOL_SET = symbol_set_name
    symbols = _symbol_sets[symbol_set_name]
    _symbol_to_id = {s: i for i, s in enumerate(symbols)}
    _id_to_symbol = {i: s for i, s in enumerate(symbols)}
    
    print(f"Symbol set changed to: {symbol_set_name} ({len(symbols)} symbols)")


def text_to_sequence(text, cleaner_names=None):
    """
    텍스트를 심볼 ID 시퀀스로 변환
    
    Args:
        text: 변환할 텍스트
        cleaner_names: 적용할 cleaner 함수 이름 리스트
                      None이면 자동으로 심볼 세트에 맞는 cleaner 선택
    
    Returns:
        심볼 ID 리스트
    """
    # cleaner_names가 None이면 자동 선택
    if cleaner_names is None:
        cleaner_names = _get_default_cleaners()
    
    # 텍스트 정제
    text = _clean_text(text, cleaner_names)
    
    # 심볼 시퀀스로 변환
    sequence = _text_to_sequence_internal(text)
    
    # EOS 토큰 추가 (선택적)
    if EOS in _symbol_to_id:
        sequence.append(_symbol_to_id[EOS])
    
    return sequence


def text_to_sequence_with_arpabet(text, cleaner_names=None):
    """
    ARPAbet 발음 기호를 지원하는 텍스트 변환
    예: "Turn left on {HH AW1 S S T AH0 N} Street."
    
    Args:
        text: 변환할 텍스트 (중괄호 내 ARPAbet 포함 가능)
        cleaner_names: 적용할 cleaner 함수 이름 리스트
    
    Returns:
        심볼 ID 리스트
    """
    if cleaner_names is None:
        cleaner_names = _get_default_cleaners()
    
    sequence = []
    
    # 중괄호 처리
    while len(text):
        m = _curly_re.match(text)
        if not m:
            # 중괄호 없음 - 일반 텍스트 처리
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        
        # 중괄호 앞 텍스트
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        
        # 중괄호 내 ARPAbet
        sequence += _arpabet_to_sequence(m.group(2))
        
        # 중괄호 뒤 텍스트
        text = m.group(3)
    
    # EOS 토큰 추가
    if EOS in _symbol_to_id:
        sequence.append(_symbol_to_id[EOS])
    
    return sequence


def sequence_to_text(sequence):
    """
    심볼 ID 시퀀스를 텍스트로 역변환
    
    Args:
        sequence: 심볼 ID 리스트
    
    Returns:
        텍스트 문자열
    """
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            
            # ARPAbet은 중괄호로 감싸기
            if len(s) > 1 and s[0] == '@':
                s = '{%s}' % s[1:]
            
            result += s
    
    return result.replace('}{', ' ')


# ========== 내부 헬퍼 함수 ==========

def _get_default_cleaners():
    """심볼 세트에 따라 기본 cleaner 선택"""
    cleaner_map = {
        'english': ['english_cleaners'],
        'korean_jamo': ['korean_cleaners_jamo'],
        'korean_romanized': ['korean_cleaners_romanized']
    }
    return cleaner_map.get(SYMBOL_SET, ['basic_cleaners'])


def _clean_text(text, cleaner_names):
    """cleaner 함수들을 순차적으로 적용"""
    for name in cleaner_names:
        cleaner = getattr(cleaners, name, None)
        if not cleaner:
            raise Exception(f'Unknown cleaner: {name}')
        text = cleaner(text)
    return text


def _text_to_sequence_internal(text):
    """정제된 텍스트를 심볼 ID 시퀀스로 변환"""
    sequence = []
    for char in text:
        if char in _symbol_to_id:
            sequence.append(_symbol_to_id[char])
    return sequence


def _symbols_to_sequence(symbols_text):
    """심볼 텍스트를 ID 시퀀스로 변환 (필터링 포함)"""
    return [_symbol_to_id[s] for s in symbols_text if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    """ARPAbet 텍스트를 ID 시퀀스로 변환"""
    return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
    """심볼을 유지할지 결정 (패딩, 특수 문자 제외)"""
    return (s in _symbol_to_id) and (s != '_') and (s != '~')


# ========== 유틸리티 함수 ==========

def get_symbol_count():
    """현재 심볼 세트의 크기 반환"""
    return len(symbols)


def get_current_symbol_set():
    """현재 사용 중인 심볼 세트 이름 반환"""
    return SYMBOL_SET


def print_symbol_info():
    """심볼 세트 정보 출력"""
    print(f"Current Symbol Set: {SYMBOL_SET}")
    print(f"Total Symbols: {len(symbols)}")
    print(f"Symbols: {symbols[:50]}...")  # 처음 50개만 출력
    print(f"EOS Token: {EOS}")


# 모듈 로드 시 정보 출력
print(f"Text module initialized with symbol set: {SYMBOL_SET} ({len(symbols)} symbols)")

