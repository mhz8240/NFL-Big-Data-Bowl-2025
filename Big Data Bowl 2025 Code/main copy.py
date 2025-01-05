import os
import zipfile
import numpy as np
import pandas as pd
import data_prep
import visualizations
# import seaborn as sns
import matplotlib.pyplot as plt
data_folder = os.path.join(os.path.dirname(__file__), "big-data-bowl-2025-datasets")
games = pd.read_csv(os.path.join(data_folder, "games.csv"))
player_plays = pd.read_csv(os.path.join(data_folder, 'player_play.csv'))
players = pd.read_csv(os.path.join(data_folder, 'players.csv'))
plays = pd.read_csv(os.path.join(data_folder, 'plays.csv'))
# tracking_week_1 = pd.read_csv(os.path.join(data_folder, 'tracking_week_1.csv'))
# tracking_week_2 = pd.read_csv(os.path.join(data_folder, 'tracking_week_2.csv'))
tracking_week_3 = pd.read_csv(os.path.join(data_folder, 'tracking_week_3.csv'))
tracking_week_4 = pd.read_csv(os.path.join(data_folder, 'tracking_week_4.csv'))
tracking_week_5 = pd.read_csv(os.path.join(data_folder, 'tracking_week_5.csv'))
tracking_week_6 = pd.read_csv(os.path.join(data_folder, 'tracking_week_6.csv'))
tracking_week_7 = pd.read_csv(os.path.join(data_folder, 'tracking_week_7.csv'))
tracking_week_8 = pd.read_csv(os.path.join(data_folder, 'tracking_week_8.csv'))
tracking_week_9 = pd.read_csv(os.path.join(data_folder, 'tracking_week_9.csv'))
print('finished reading')
tracking_combined = pd.concat([tracking_week_3,tracking_week_4,tracking_week_5,tracking_week_6,
                               tracking_week_7,tracking_week_8,tracking_week_9])
# tracking_combined = tracking_week_3

#filter plays
plays_with_motion, player_plays_with_motion=data_prep.filter_plays(plays, player_plays)
#only zone
# plays_with_motion, player_plays_with_motion = data_prep.filter_zone(plays_with_motion, player_plays_with_motion)
#only man
# plays_with_motion, player_plays_with_motion = data_prep.filter_man(plays_with_motion, player_plays_with_motion)
#only dropbacks
plays_with_motion, player_plays_with_motion = data_prep.filter_pass(plays_with_motion, player_plays_with_motion)
#only designated runs
# plays_with_motion, player_plays_with_motion = data_prep.filter_rush(plays_with_motion, player_plays_with_motion)

#filter tracking data
tracking_combined_filtered, tracking_combined_displacement, plays_combined = data_prep.filter_tracking_and_plays(tracking_combined, plays_with_motion,player_plays_with_motion)
tracking_combined_defenders=data_prep.tracking_defenders(tracking_combined,tracking_combined_filtered,plays_combined)
# #epa_allowed data
temp_df, temp_df2=data_prep.temp_df_function(tracking_combined_displacement,tracking_combined_defenders)
# print(tracking_combined_defenders.iloc[75:130][['game_play_nflId','frameId','distance']])
# plt.figure(figsize=(10,6))
# distance_ratio=visualizations.plot_distance_ratio(tracking_combined_filtered,tracking_combined_defenders,plays_combined)
plt.figure(figsize=(10,6))
distance_projection_ratio=visualizations.plot_distance_projection_ratio(tracking_combined_filtered,temp_df2,plays_combined)
plt.figure(figsize=(10,6))
displacement_projection_ratio=visualizations.plot_disp_projection_ratio(tracking_combined_displacement,temp_df,plays_combined)
plt.figure(figsize=(10,6))
sum_ratios=visualizations.plot_sum_scaled(distance_projection_ratio,displacement_projection_ratio,plays_combined)


# plt.figure(figsize=(10,6))
# visualizations.plot_bargraph(sum_ratios,plays_combined)
# visualizations.trs_order(distance_projection_ratio,displacement_projection_ratio,sum_ratios)
plt.show()
