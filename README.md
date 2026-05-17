# Fallen Angel Screener — GitHub Pages 배포 가이드

S&P 500 고품질 낙폭 과대 종목을 매일 07:00 KST에 자동 스크리닝하여 GitHub Pages에 게시합니다.

---

## 1단계 — GitHub 저장소 생성

1. [github.com/new](https://github.com/new) 접속
2. Repository name: `fallen-angel-screener` (원하는 이름 가능)
3. **Public** 선택
4. "Add a README file" 체크 **해제**
5. **Create repository** 클릭

---

## 2단계 — 로컬 파일 업로드

```bash
# 이 폴더(fallen_angel_web)로 이동
cd fallen_angel_web

# git 초기화 및 원격 저장소 연결
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

> `<your-username>` 과 `<repo-name>` 을 본인 GitHub 정보로 교체하세요.

---

## 3단계 — GitHub Pages 활성화

1. 저장소 페이지 → **Settings** 탭
2. 왼쪽 메뉴 **Pages** 클릭
3. Source: **Deploy from a branch**
4. Branch: `main` / 폴더: `/ docs`
5. **Save** 클릭

수 분 후 `https://<your-username>.github.io/<repo-name>/` 에서 접속 가능합니다.

---

## 4단계 — 첫 번째 수동 실행 (선택)

Actions 탭 → **Daily Fallen Angel Report** → **Run workflow** → **Run workflow** 클릭

첫 실행에 10~20분 소요됩니다. 완료되면 `docs/index.html` 이 실제 리포트로 교체됩니다.

---

## 5단계 — 이후 자동 업데이트

GitHub Actions 크론(`0 22 * * *` UTC = 07:00 KST)이 매일 자동 실행합니다.  
별도 조작 없이 페이지가 매일 아침 갱신됩니다.

---

## 디렉터리 구조

```
fallen_angel_web/
├── .github/
│   └── workflows/
│       └── daily.yml       # GitHub Actions 크론 워크플로우
├── docs/
│   └── index.html          # GitHub Pages 서빙 경로 (자동 갱신)
├── data/
│   └── analysis_YYYYMMDD.csv  # 날짜별 분석 결과 (자동 생성)
├── src/
│   └── pipeline.py         # 메인 파이프라인
├── requirements.txt
└── .gitignore
```
