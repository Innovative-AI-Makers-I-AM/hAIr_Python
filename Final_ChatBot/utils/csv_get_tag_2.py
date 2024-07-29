import pandas as pd

# CSV 파일을 불러옵니다.
df = pd.read_csv('new_hairstyles_filtered.csv')

# 'sex', 'length', 'style' 열을 결합하여 새로운 열을 생성합니다.
df['combined'] = df['sex'] + ' ' + df['length'] + ' ' + df['style']

# 중복을 제외하고 고유한 값들을 추출합니다.
unique_combined = df['combined'].unique()

# 결과를 출력합니다.
print('Unique combined values:')
for value in unique_combined:
    print(value)
