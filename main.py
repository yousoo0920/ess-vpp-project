from preprocessing.filter_jeju import filter_jeju_data

input_file = "data/한국전력거래소_지역별 시간별 태양광 및 풍력 발전량_2024.csv"
output_file = "data/제주도_풍력태양광_시간별.csv"

filter_jeju_data(input_file, output_file, use_solar=True)
