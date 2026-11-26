"""로컬 의도 분류기: 예시 문장(config/intents.yaml) ↔ 입력 문장 글자 n-gram TF-IDF 코사인 유사도.

이미 설치된 scikit-learn 만 사용 (설치·API·비용 없음, 오프라인, CPU 수 ms).
가장 비슷한 예시의 작업을 고르고, 유사도로 확신도를 정한다:
  sim < ask_below        → 'ask' (되묻기)
  ask_below ≤ sim < sure → 해당 작업, 낮은 confidence (decision_node 가 "~할까요?" 로 확인)
  sim ≥ sure             → 해당 작업, 높은 confidence (바로 실행)
"""
import os

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer


_ENDINGS = ('해주세요', '해줘요', '해줄래', '해줘', '주세요', '줄래', '줘요', '줘', '좀', '요', '?', '!', '.')


def _norm(t):
    """공백 제거 + 흔한 요청 어미(해줘/주세요/좀…) 제거 → 내용 단어 위주로 비교"""
    t = t.replace(' ', '').strip()
    changed = True
    while changed and len(t) > 2:
        changed = False
        for e in _ENDINGS:
            if t.endswith(e) and len(t) > len(e) + 1:
                t, changed = t[:-len(e)], True
    return t.replace('좀', '')


class LocalIntentClassifier:
    def __init__(self, path, ask_below=0.45, sure=0.65):
        with open(os.path.expanduser(path)) as f:
            self.intents = (yaml.safe_load(f) or {}).get('intents', {})
        self.ask_below, self.sure = ask_below, sure
        self.texts, self.labels = [], []
        for skill, d in self.intents.items():
            for ex in d.get('examples', []):
                self.texts.append(_norm(str(ex)))
                self.labels.append(skill)
        # 한국어는 띄어쓰기가 들쭉날쭉 → 공백 제거 후 글자 1~3그램
        self.vec = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), sublinear_tf=True)
        self.X = self.vec.fit_transform(self.texts)

    def classify(self, text):
        """→ (skill, target, say, confidence, best_example, sim)"""
        q = self.vec.transform([_norm(text)])
        sims = (self.X @ q.T).toarray().ravel()
        i = int(np.argmax(sims)) if len(sims) else -1
        sim = float(sims[i]) if i >= 0 else 0.0
        if i < 0 or sim < self.ask_below:
            return 'ask', None, '잘 못 알아들었어요. 무엇을 도와드릴까요?', sim, None, sim
        skill = self.labels[i]
        d = self.intents.get(skill, {})
        if sim >= self.sure:
            conf = 0.6 + 0.4 * min(1.0, (sim - self.sure) / (1 - self.sure + 1e-6))
        else:
            conf = 0.3 + 0.3 * (sim - self.ask_below) / (self.sure - self.ask_below + 1e-6)
        return skill, d.get('target'), d.get('say', ''), float(conf), self.texts[i], sim
