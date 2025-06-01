import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from google.colab import drive
drive.mount('/content/drive')


# CSV 파일 불러오기
df = pd.read_csv('/content/drive/MyDrive/블루문/2505_천안시_초급/데이터/card_train.csv')  # 파일명 변경 필요


# 사용할 변수만 선택
cols = [
    '남녀구분코드', '연령', '거주시도명', '직장시도명',
    '입회경과개월수_신용', '탈회횟수_누적', '최종탈회후경과월',
    '소지여부_신용', '소지카드수_유효_신용',
    '회원여부_이용가능', '회원여부_이용가능_CA',
    '마케팅동의여부', 'Life_Stage', 'Segment'
]
df = df[cols].copy()


#타겟값 가공

df['target'] = df['Segment'].apply(lambda x: 1 if x == 'E' else 0)
df.drop(columns=['Segment'], inplace=True)


#전처리

# 범주형 변수 인코딩
cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))


#분할

X = df.drop(columns='target')
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


#모델 학습

rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)


#모델 평가

from sklearn.metrics import classification_report, confusion_matrix

# 예측
y_pred = rf.predict(X_test)

# 성능 평가
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


#Non-E 데이터만 추출 + 타겟 병합

# 원본 CSV 처음부터 다시 불러오기 (2단계 시작할 때!)
df_original = pd.read_csv('/content/drive/MyDrive/블루문/2505_천안시_초급/데이터/card_train.csv')  # ← 실제 파일명으로 바꿔줘

# 원래 Segment에서 E 제외
df_multi = df_original[df_original['Segment'] != 'E'].copy()  # df_original은 전체 CSV에서 불러온 원본

# 타겟 병합: A or B → AB
df_multi['Segment_ABCD'] = df_multi['Segment'].apply(lambda x: 'AB' if x in ['A', 'B'] else x)

# 사용 변수 선택
columns = [
    '남녀구분코드', '연령', '거주시도명', '직장시도명',
    '입회경과개월수_신용', '탈회횟수_누적', '최종탈회후경과월',
    '소지여부_신용', '소지카드수_유효_신용',
    '회원여부_이용가능', '회원여부_이용가능_CA',
    '마케팅동의여부', 'Life_Stage', 'Segment_ABCD'
]
df_multi = df_multi[columns].copy()


#범주형 인코딩

# Label Encoding
cat_cols = df_multi.select_dtypes(include='object').columns

for col in cat_cols:
    df_multi[col] = LabelEncoder().fit_transform(df_multi[col].astype(str))


#학습,검증

X = df_multi.drop(columns='Segment_ABCD')
y = df_multi['Segment_ABCD']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


#모델학습 3개

# 모델 1: RandomForest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# 모델 2: LightGBM
lgb = LGBMClassifier(random_state=42)
lgb.fit(X_train, y_train)
lgb_pred = lgb.predict(X_test)

# 모델 3: XGBoost
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)


def evaluate_model(name, y_true, y_pred):
    print(f"\n📊 {name} 결과")
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))

evaluate_model("RandomForest", y_test, rf_pred)
evaluate_model("LightGBM", y_test, lgb_pred)
evaluate_model("XGBoost", y_test, xgb_pred)


#SMOTE 적용 시작

!pip install imbalanced-learn catboost

from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 타겟은 Segment_ABCD (AB=1, C=2, D=0)
X = df_multi.drop(columns='Segment_ABCD')
y = df_multi['Segment_ABCD']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


#SMOTE 실제 적용

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)


#모델별 적용

##RF

rf_sm = RandomForestClassifier(random_state=42)
rf_sm.fit(X_train_sm, y_train_sm)
rf_pred = rf_sm.predict(X_test)


##XGBC

xgb_sm = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_sm.fit(X_train_sm, y_train_sm)
xgb_pred = xgb_sm.predict(X_test)


##CBC

cat_sm = CatBoostClassifier(verbose=0, random_seed=42)
cat_sm.fit(X_train_sm, y_train_sm)
cat_pred = cat_sm.predict(X_test)


#평가함수 공통화

from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(name, y_true, y_pred):
    print(f"\n📊 {name} 결과")
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))

evaluate_model("SMOTE + RF", y_test, rf_pred)
evaluate_model("SMOTE + XGBoost", y_test, xgb_pred)
evaluate_model("SMOTE + CatBoost", y_test, cat_pred)


#D 클래스 모든 모델에서 거의 미미한 정도 따라서 제거 후 ABvsC 이진 분류

# 0 = D 제거
df_binary = df_multi[df_multi['Segment_ABCD'] != 0].copy()

# 새 타겟: AB=1, C=0
df_binary['target'] = df_binary['Segment_ABCD'].apply(lambda x: 1 if x == 1 else 0)
df_binary.drop(columns='Segment_ABCD', inplace=True)


##학습/검증 분할 + SMOTE

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

X = df_binary.drop(columns='target')
y = df_binary['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# SMOTE로 클래스 균형 맞추기
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)


##CatBoostClassifier 학습 및 평가(가장 우수한 모델)

from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix

cat = CatBoostClassifier(verbose=0, random_seed=42)
cat.fit(X_train_sm, y_train_sm)
y_pred = cat.predict(X_test)

# 성능 확인
print("📊 CatBoost 결과 (AB vs C):")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# 1. 나눔폰트 설치
!apt-get update -qq
!apt-get install -y fonts-nanum

# 2. 런타임에 폰트 적용
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rc('font', family='NanumGothic')  # 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지


feature_importance = cat.get_feature_importance(prettified=True)
print(feature_importance.head())


import matplotlib.pyplot as plt
import seaborn as sns

# 중요 변수 상위 10개 시각화
feature_importance = cat.get_feature_importance(prettified=True)
top_features = feature_importance.head(10)

plt.figure(figsize=(8, 6))
sns.barplot(data=top_features, x='Importances', y='Feature Id')
plt.title('Top 10 Feature Importance (CatBoost)')
plt.tight_layout()
plt.show()


from catboost import CatBoostClassifier

# 튜닝된 CatBoostClassifier
cat = CatBoostClassifier(
    iterations=600,        # 더 오래 학습
    depth=6,               # 나무 깊이 (복잡도 조절)
    learning_rate=0.03,    # 학습률 낮춰서 안정적으로
    l2_leaf_reg=5,         # 과적합 방지용 L2 정규화
    random_seed=42,
    verbose=100            # 학습 로그 출력 (100회 단위)
)

# 학습
cat.fit(X_train_sm, y_train_sm)


from sklearn.metrics import classification_report, confusion_matrix

y_pred = cat.predict(X_test)

print("📊 튜닝된 CatBoost 결과:")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


cat = CatBoostClassifier(
    iterations=600,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=5,
    class_weights=[1, 1.5],  # [C 클래스, AB 클래스]
    random_seed=42,
    verbose=100
)
cat.fit(X_train_sm, y_train_sm)

y_pred = cat.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print("📊 CatBoost (가중치 적용)")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


#정확도 70이상으로 올리기 시도

from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix

cat = CatBoostClassifier(
    iterations=600,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=5,
    random_seed=42,
    verbose=100
)

cat.fit(X_train_sm, y_train_sm)

y_pred = cat.predict(X_test)

print("📊 CatBoost 결과 (정확도 중심 튜닝)")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


##정확도 상승 위해 파생변수 추가

df['입회단계'] = pd.cut(df['입회경과개월수_신용'], bins=[0,12,36,1000], labels=['신규','중기','장기'])
df['카드3초과'] = (df['소지카드수_유효_신용'] > 3).astype(int)
df['거주직장일치'] = (df['거주시도명'] == df['직장시도명']).astype(int)


# 📌 1. 원본 복사
df_fe = df_binary.copy()

# 📌 2. 파생변수 생성 (💥 여기에서 수정!!)
# 🔽 기존 코드에서 이 부분을 완전히 교체하세요 🔽

df_fe['입회단계'] = pd.cut(
    df_fe['입회경과개월수_신용'],
    bins=[-1, 12, 36, df_fe['입회경과개월수_신용'].max()],
    labels=[0, 1, 2]  # 숫자 레이블 사용
).astype(int)  # 정수형으로 강제 변환 ✅

# 📌 3. 다른 파생변수 그대로 사용 OK
df_fe['카드3초과'] = (df_fe['소지카드수_유효_신용'] > 3).astype(int)
df_fe['거주직장일치'] = (df_fe['거주시도명'] == df_fe['직장시도명']).astype(int)

# 📌 4. Label Encoding
from sklearn.preprocessing import LabelEncoder

for col in df_fe.columns:
    if df_fe[col].dtype == 'object':
        df_fe[col] = LabelEncoder().fit_transform(df_fe[col].astype(str))

# 📌 5. SMOTE → 학습
X = df_fe.drop(columns='target')
y = df_fe['target']

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)


X = df_fe.drop(columns='target')
y = df_fe['target']


##LightGBM 학습 및 평가 코드(파생변수 이후)

from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# 1. 데이터 분할
X = df_fe.drop(columns='target')
y = df_fe['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. SMOTE 적용
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# 3. LightGBM 모델 학습
lgb = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
lgb.fit(X_train_sm, y_train_sm)

# 4. 예측 및 평가
y_pred = lgb.predict(X_test)

print("📊 LightGBM 결과:")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


import pandas as pd
import numpy as np

# 파일 업로드했을 경우 해당 경로로 불러오기
test_df = pd.read_csv('/content/drive/MyDrive/블루문/2505_천안시_초급/데이터/card_test.csv')

# 예시: 날짜 컬럼 처리
if '가입일' in test_df.columns:
    test_df['가입일'] = pd.to_datetime(test_df['가입일'], errors='coerce')
    test_df['가입연'] = test_df['가입일'].dt.year
    test_df['가입월'] = test_df['가입일'].dt.month
    test_df['가입일자'] = test_df['가입일'].dt.day

# 예시: 파생 변수 생성
if '이용금액' in test_df.columns:
    test_df['1회이용금액'] = test_df['이용금액'] / (test_df['이용건수'] + 1)

# 예시: 필요 없는 컬럼 삭제
drop_cols = ['고객ID', '가입일']
test_df = test_df.drop(columns=[col for col in drop_cols if col in test_df.columns], errors='ignore')


# 범주형 변수 라벨 인코딩 (같은 방식으로 train과 통일되어야 함)
from sklearn.preprocessing import LabelEncoder

for col in test_df.select_dtypes(include='object').columns:
    test_df[col] = LabelEncoder().fit_transform(test_df[col].astype(str))


# 간단히 평균/최빈값으로 대체 (train에서 했던 방식 따라야 함)
for col in test_df.columns:
    if test_df[col].isnull().sum() > 0:
        if test_df[col].dtype == 'object':
            test_df[col] = test_df[col].fillna(test_df[col].mode()[0])
        else:
            test_df[col] = test_df[col].fillna(test_df[col].mean())


print("전처리 완료 ✅")
print(test_df.shape)
test_df.head()


!pip install lightgbm
