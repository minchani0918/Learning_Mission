import torch
import torch.nn as nn
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score


# 1. MySQL 데이터 불러오기 및 데이터프레임 생성
username = 'root'
password = '1234'
host = 'localhost'
port = '3306'
database = 'test01'

# MySQL 연결 및 데이터 읽어오기
engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}")
query = """SELECT *
FROM `전력소비2017~2020`
WHERE `1월` != 0
  AND `2월` != 0
  AND `3월` != 0
  AND `4월` != 0
  AND `5월` != 0
  AND `6월` != 0
  AND `7월` != 0
  AND `8월` != 0
  AND `9월` != 0
  AND `10월` != 0
  AND `11월` != 0
  AND `12월` != 0;
"""
df = pd.read_sql(query, engine)

# 2. 범주형 데이터를 수치형으로 변환
label_encoder = LabelEncoder()

# "계약종별", "시군구", "시도" 같은 범주형 데이터를 숫자로 변환
df["계약종별"] = label_encoder.fit_transform(df["계약종별"])
df["시군구"] = label_encoder.fit_transform(df["시군구"])
df["시도"] = label_encoder.fit_transform(df["시도"])

# 데이터 전처리: 특성 및 시계열 데이터 분리
x_features = df[["연도", "계약종별", "시군구", "시도"]]  # 특성 데이터
x_sequence = df.iloc[:, 5:].values  # 월별 데이터

# 3. 데이터 정규화
scaler = MinMaxScaler()
x_features_scaled = scaler.fit_transform(x_features)
x_features = torch.tensor(x_features_scaled, dtype=torch.float32)
x_sequence = torch.tensor(x_sequence, dtype=torch.float32)

# 4. LSTM 기반 전기 사용량 예측 모델 정의
class ElectricityUsagePredictor(nn.Module):
    def __init__(self, feature_size, hidden_size=128, num_layers=3, output_size=12):
        super(ElectricityUsagePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x_features):
        out_features, _ = self.lstm(x_features.unsqueeze(1))  # x_features를 3D 텐서로 변환
        out = self.fc(out_features[:, -1, :])  # 마지막 타임스텝의 출력을 사용
        return out


# 모델 초기화
feature_size = x_features.size(1)
model = ElectricityUsagePredictor(feature_size=feature_size)

# 5. 배치 학습을 위한 데이터셋 및 데이터로더 준비
batch_size = 32
dataset = TensorDataset(x_features, x_sequence)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 6. 옵티마이저 및 손실 함수 설정
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

# 7. 학습 루프
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}')

# 8. 모델 평가: 정확도 측정 (테스트 데이터 사용)
model.eval()
x_features_test = x_features  # 예제 특성 데이터
y_true = x_sequence  # 실제 전기 사용량을 y_true로 설정

with torch.no_grad():
    y_pred = model(x_features_test)

# 정확도 평가 (MSE와 MAE 계산)
y_true_np = y_true.numpy()
y_pred_np = y_pred.numpy()

r2 = r2_score(y_true_np, y_pred_np)
print(f"R² 스코어: {r2:.4f}")

# 10. 예측 결과 시각화 (그래프 출력)
plt.figure(figsize=(12, 6))
for i in range(12):  # 12개월 예측 그래프
    plt.plot(y_true_np[:, i], label=f"실제 값 - {i+1}월")
    plt.plot(y_pred_np[:, i], label=f"예측 값 - {i+1}월", linestyle="--")

plt.title("실제 값과 예측 값 비교")
plt.xlabel("샘플")
plt.ylabel("전기 사용량")
plt.legend(loc="upper right")
plt.show()

# 11. 새로운 데이터로 예측 수행
x_features_new = torch.tensor([[2021, 0, 0, 0]], dtype=torch.float32)  # 예시 특성 데이터

# 예측 결과 확인
with torch.no_grad():
    y_new_pred = model(x_features_new)

print("예측 결과 (다음 해의 월별 전기 사용량):")
print(y_new_pred)
