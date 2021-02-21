import re
import unicodedata
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ThreadTitleNormalizer:
    ignore_phrases: List[str]
    special_token_map: Dict[str, str]
    special_token_regex_map: Dict[re.Pattern, str]

    def normalize(self, title: str):
        """
        スレッドタイトルに対して以下の前処理を行う
        ・NFKC正規化
        ・Lower
        ・ignore_phrasesに含まれる文字列を削除
        ・special_token_map, special_token_regex_mapに含まれる文字列を変換する

        Args:
          title (str): 前処理を行うスレッドタイトル

        Returns:
          str: 前処理後のスレッドタイトル
        """
        processed = unicodedata.normalize("NFKC", title)
        processed = processed.lower()
        for phrase in self.ignore_phrases:
            processed = processed.replace(phrase, "")
        for phrase, token in self.special_token_map.items():
            processed = processed.replace(phrase, token)
        for regex, token in self.special_token_regex_map.items():
            processed = regex.sub(token, processed)
        return processed.strip()