import pandas as pd

# CSV 파일을 불러옵니다.
df = pd.read_csv('new_hairstyles_filtered.csv')

# 중복을 제외하고 'Sex', 'Length', 'Style' 열의 고유한 값들을 추출합니다.
unique_sex = df['sex'].unique()
unique_length = df['length'].unique()
unique_style = df['style'].unique()

# 결과를 출력합니다.
print('Unique values in Sex:', unique_sex)
print('Unique values in Length:', unique_length)
print('Unique values in Style:', unique_style)