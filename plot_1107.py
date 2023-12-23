import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 읽기
df = pd.read_csv('ghost_positions_04.csv', header=None, names=['x', 'y', 'z'])


df_subset = df.iloc[1000:4000]

# 플롯 설정
plt.figure(figsize=(8, 8))

# 데이터 시각화
plt.scatter(df_subset['x'], df_subset['y'], alpha=0.5, color='red')

# 플롯 타이틀 및 레이블 설정
plt.title(f'Ghost Position Distribution (0.5 GAIL + 0.5 rule)')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.xlim(-80, 40) # X축 범위
plt.ylim(-50, 50) # Y축 범위
plt.grid(True) # 그리드 활성화

# 플롯 보여주기
plt.show()



# import pandas as pd
# import matplotlib.pyplot as plt
# r = 2540
# # 첫 번째 CSV 파일 읽기
# df1 = pd.read_csv('ghost_positions_rule_5.csv', header=None, names=['x', 'y', 'z'], nrows=r)

# # 두 번째 CSV 파일 읽기
# df2 = pd.read_csv('ghost_positions_04.csv', header=None, names=['x', 'y', 'z'] , nrows=r)

# # 서브플롯 설정
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8)) # 1행 2열의 서브플롯

# # 첫 번째 서브플롯 설정
# ax1.set_xlim(-80, 40)
# ax1.set_ylim(-50, 50)
# ax1.scatter(df1['x'], df1['y'], alpha=0.5, color='blue')
# ax1.set_title('Ghost Position Distribution (Rule)')
# ax1.set_xlabel('X Position')
# ax1.set_ylabel('Y Position')
# ax1.grid(True)

# # 두 번째 서브플롯 설정
# ax2.set_xlim(-80, 40)
# ax2.set_ylim(-50, 50)
# ax2.scatter(df2['x'], df2['y'], alpha=0.5, color='red')
# ax2.set_title('Ghost Position Distribution (0.9 GAIL + 0.1 rule)')
# ax2.set_xlabel('X Position')
# ax2.set_ylabel('Y Position')
# ax2.grid(True)

# # 서브플롯 사이의 간격 조정
# plt.tight_layout()

# # 플롯 보여주기
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# r = 4540
# # 첫 번째 CSV 파일 읽기
# df1 = pd.read_csv('ghost_positions_rule_5.csv', header=None, names=['x', 'y', 'z'], nrows=r)
# # 두 번째 CSV 파일 읽기
# df2 = pd.read_csv('ghost_positions_04.csv', header=None, names=['x', 'y', 'z'], nrows=r)

# # 각 포인트의 더미 번호 계산 (인덱스를 20으로 나눈 나머지를 사용)
# dummy_number_df1 = df1.index % 20
# dummy_number_df2 = df2.index % 20

# # 더미 번호에 따라 색상 맵 생성
# colors = plt.cm.jet(np.linspace(0, 1, 20))  # 20개의 더미, 각각에 대한 색상

# # 서브플롯 설정
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# # 첫 번째 서브플롯 설정 (df1)
# ax1.set_xlim(-80, 40)
# ax1.set_ylim(-50, 50)

# # df1의 각 더미 데이터를 해당하는 색상으로 플롯
# for i in range(20):  # 0부터 19까지 더미 번호에 대해 반복
#     ax1.scatter(df1.loc[dummy_number_df1 == i, 'x'], df1.loc[dummy_number_df1 == i, 'y'], alpha=0.5, color=colors[i])

# ax1.set_title('Ghost Position Distribution (Rule)')
# ax1.set_xlabel('X Position')
# ax1.set_ylabel('Y Position')
# ax1.grid(True)

# # 두 번째 서브플롯 설정 (df2)
# ax2.set_xlim(-80, 40)
# ax2.set_ylim(-50, 50)

# # df2의 각 더미 데이터를 해당하는 색상으로 플롯
# for i in range(20):  # 0부터 19까지 더미 번호에 대해 반복
#     ax2.scatter(df2.loc[dummy_number_df2 == i, 'x'], df2.loc[dummy_number_df2 == i, 'y'], alpha=0.5, color=colors[i])

# ax2.set_title('Ghost Position Distribution (GAIL + rule)')
# ax2.set_xlabel('X Position')
# ax2.set_ylabel('Y Position')
# ax2.grid(True)

# # 서브플롯 사이의 간격 조정
# plt.tight_layout()

# # 플롯 보여주기
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# r = 4500
# df1 = pd.read_csv('ghost_positions_rule_5.csv', header=None, names=['x', 'y', 'z'], nrows=r)
# # 두 번째 CSV 파일 읽기
# df2 = pd.read_csv('ghost_positions_04.csv', header=None, names=['x', 'y', 'z'], nrows=r)

# # 주석 처리된 부분에서 사용된 색상 맵 (plt.cm.jet)을 사용
# colors = plt.cm.jet(np.linspace(0, 1, 20))

# # 각 그룹에 대해 플롯을 생성하고 저장하는 함수
# def save_group_plots(df, filename):
#     fig, axs = plt.subplots(5, 4, figsize=(20, 25))  # 5x4 그리드

#     for i in range(20):
#         ax = axs[i // 4, i % 4]
#         # 해당 그룹의 데이터만 필터링하여 플롯
#         ax.scatter(df[df.index % 20 == i]['x'], df[df.index % 20 == i]['y'], color=colors[i], label=f'Group {i+1}')
#         ax.set_title(f'Group {i+1}')
#         ax.set_xlim(-80, 40)
#         ax.set_ylim(-50, 50)
#         ax.grid(True)
#         ax.legend()

#     # 불필요한 축을 숨김
#     for ax in axs.flat:
#         ax.label_outer()

#     plt.tight_layout()
#     plt.savefig(filename, format='jpg')
#     plt.close()

# # df1에 대한 플롯 저장
# save_group_plots(df1, 'group_plots_rule.jpg')

# # df2에 대한 플롯 저장
# save_group_plots(df2, 'group_plots_04.jpg')
