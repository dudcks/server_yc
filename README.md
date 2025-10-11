1. 필요한 라이브러리 설치 : pip install -r dependencies.txt

2. 모델 파일 이름은 best-model.pt 이며 이 모델 파일이 있어야 함

3. 서버 실행 : uvicorn app:app --reload

4. 테스트 실행 : pytest test.py
- test는 모듈별로 클래스를 만들어 진행할 것

5. kss 사용시 mecab이 없어 pecab을 사용한다는 문구가 출력되는 경우 mecab설치 필요.
- kss에서 mecab을 인식하지 못하고 pecab으로 동작하는 경우, libstdcc의 버전 문제일 수 있음.
## conda install -c conda-forge libstdcxx-ng

