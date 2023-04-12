import pymysql
import pandas as pd

df = pd.read_excel("Data_EDA\df.xlsx")

# MySQL 데이터베이스 연결
connection = pymysql.connect(
    host="localhost",
    user="root",
    password="kdh711093!A",
    database="KMA"
)
# 데이터베이스 생성 및 선택
with connection.cursor() as cursor:
    cursor.execute("CREATE DATABASE IF NOT EXISTS KMA;")
    cursor.execute("USE KMA;")
connection.commit()

# 테이블 생성
with connection.cursor() as cursor:
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS KMA (
            날짜 VARCHAR(10),
            요일 VARCHAR(10),
            날씨 VARCHAR(10),
            총계 INT,
            평균기온 FLOAT,
            강수량 FLOAT
        );
    """)
connection.commit()

# 데이터 삽입
for i in range(len(df)):
    with connection.cursor() as cursor:
        cursor.execute("""
            INSERT INTO KMA (날짜, 요일, 날씨, 총계, 평균기온, 강수량)
            VALUES (%s, %s, %s, %s, %s, %s);
        """, (df.iloc[i, 0], df.iloc[i, 1],
            df.iloc[i, 2], df.iloc[i, 3],
            df.iloc[i, 4], df.iloc[i, 5]
            ))
connection.commit()

# 데이터 검색
with connection.cursor() as cursor:
    cursor.execute("SELECT 날짜, 요일, 날씨, 총계, 평균기온, 강수량 FROM KMA;")
    results = cursor.fetchall()

for row in results:
    print(row)

"""
# 데이터 삽입
with connection.cursor() as cursor:
    cursor.execute(\"""
        INSERT INTO KMA (날짜, 요일, 날씨, 총계, 평균기온, 강수량)
        VALUES (%s, %s, %s, %s, %s, %s);
    \""", (df.iloc[0, 0], df.iloc[0, 1],
          df.iloc[0, 2], df.iloc[0, 3],
          df.iloc[0, 4], df.iloc[0, 5]
          )) # SQL 인젝션을 방지하는 데 도움이 된다?
connection.commit()

# 데이터 삭제
with connection.cursor() as cursor:
    cursor.execute(\"""
        DELETE FROM kma;
    \""")
connection.commit()
"""

# 연결 종료
connection.close()