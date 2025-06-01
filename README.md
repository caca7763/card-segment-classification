# 신용카드 고객 세그먼트 분류 프로젝트

이 프로젝트는 신용카드 고객 데이터를 기반으로, 고위험군(E 클래스) 식별 및 비E군의 세분화를 목적으로 합니다.  
총 2단계 분류 모델(RandomForest, CatBoost, LightGBM 등)을 적용하여 고객군별 특성을 예측합니다.

## 🔍 프로젝트 개요

- 데이터 출처: 가공된 신용카드 고객 세그먼트 CSV
- 주요 목표:
  1. Segment E 여부 분류 (1단계)
  2. Non-E 그룹 세부 분류 (AB, C, D)

## 📁 파일 구성

- `card_classification.ipynb`: 전체 분석, 모델 학습 및 시각화 코드
- `신용카드_세그먼트_분류_최종.txt`: 최종 리포트 요약 (텍스트 기반)
- `README.md`: 본 프로젝트 소개 파일

## ⚙️ 사용 모델 및 기법

- RandomForestClassifier
- CatBoostClassifier (튜닝 포함)
- LightGBMClassifier
- SMOTE (클래스 불균형 처리)

## 📈 결과 요약

- E 클래스 분류 정확도: 87.4%
- Non-E 세부 분류 정확도: 모델별로 65~80% 수준

## 📌 주요 특징

- 다단계 분류 접근
- 파생변수 활용 및 SMOTE 적용
- 모델별 Confusion Matrix 및 F1-score 비교
