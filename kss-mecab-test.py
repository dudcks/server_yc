import kss
from mecab import MeCab

m = MeCab()
print(m.morphs("테스트 중입니다."))

text = "안녕하세요. kss 테스트 중..... 오늘 날씨가 참 좋네요."
sentences = kss.split_sentences(text)
print(sentences)
##정상##[Kss]: Oh! You have mecab in your environment. Kss will take this as a backend! :D
##pecab을 사용한다거나 mecab 등을 설치하라는 경우 README 확인.