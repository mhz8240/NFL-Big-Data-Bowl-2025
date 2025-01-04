
def filter_plays(plays, player_plays):
  plays_filtered = plays[(plays['qbSpike']!=1) & (plays['qbKneel']!=True) & (plays['playNullifiedByPenalty'] != 'Y')]
  plays_filtered['game_and_playId'] = plays_filtered['gameId'].astype(str) + '_' + plays_filtered['playId'].astype(str)
  # #pass coverage data preparation
  # plays_by_coverage = plays_filtered[plays_filtered['pff_passCoverage'].notnull()]
  # #filter by plays with motion and with coverage data
  # plays_by_coverage = plays_by_coverage[plays_by_coverage.apply(drop_condition,axis=1)]
  # plays_by_coverage['coverage'] = plays_by_coverage['pff_passCoverage'].apply(update_coverage)
  plays_filtered = plays_filtered[plays_filtered['pff_manZone']!='Other']
  player_plays['game_and_playId'] = player_plays['gameId'].astype(str) + '_' + player_plays['playId'].astype(str)
  player_plays_filtered = player_plays[player_plays['game_and_playId'].isin(plays_filtered.game_and_playId)]
  player_plays_motion = player_plays_filtered[player_plays_filtered['motionSinceLineset']==True]
  plays_motion = plays_filtered[plays_filtered['game_and_playId'].isin(player_plays_motion.game_and_playId)]
  # plays_with_motion_by_coverage = plays_motion[plays_motion['pff_passCoverage'].notnull()]
  # plays_with_motion_by_coverage = plays_with_motion_by_coverage[plays_with_motion_by_coverage.apply(drop_condition,axis=1)]
  # plays_with_motion_by_coverage['coverage'] = plays_with_motion_by_coverage['pff_passCoverage'].apply(update_coverage)
  # player_plays_by_coverage = player_plays_filtered[player_plays_filtered['game_and_playId'].isin(plays_with_motion_by_coverage.game_and_playId)]
  # player_plays_with_motion_by_coverage = player_plays_by_coverage[player_plays_by_coverage['motionSinceLineset']==True]
  player_plays_motion['game_play_nflId'] = player_plays_motion['game_and_playId'] + '_' + player_plays_motion['nflId'].astype(str)
  return plays_motion, player_plays_motion

def filter_zone(plays, player_plays):
  plays_filtered = plays[plays['pff_manZone']=='Zone']
  player_plays_filtered = player_plays[player_plays['game_and_playId'].isin(plays_filtered.game_and_playId)]
  return plays_filtered, player_plays_filtered

def filter_man(plays, player_plays):
  plays_filtered = plays[plays['pff_manZone']!='Zone']
  player_plays_filtered = player_plays[player_plays['game_and_playId'].isin(plays_filtered.game_and_playId)]
  return plays_filtered, player_plays_filtered

def filter_pass(plays, player_plays):
  plays_filtered = plays[plays['isDropback']==True]
  player_plays_filtered = player_plays[player_plays['game_and_playId'].isin(plays_filtered.game_and_playId)]
  return plays_filtered, player_plays_filtered

def filter_rush(plays, player_plays):
  plays_filtered = plays[plays['isDropback']!=True]
  player_plays_filtered = player_plays[player_plays['game_and_playId'].isin(plays_filtered.game_and_playId)]
  return plays_filtered, player_plays_filtered

#tracking data filtering method
def filter_tracking_and_plays(tracking_combined, plays_with_motion_by_coverage,player_plays_with_motion_by_coverage):
  tracking_combined['game_and_playId'] = tracking_combined['gameId'].astype(str)+ '_' + tracking_combined['playId'].astype(str)
  # tracking_combined['game_play_nflId'] = tracking_combined['game_and_playId']+'_'+tracking_combined['nflId'].fillna(0).astype(str)

  tracking_combined_temp = tracking_combined[tracking_combined['game_and_playId'].isin(plays_with_motion_by_coverage.game_and_playId)]
  tracking_combined_temp['game_play_nflId'] = tracking_combined_temp['game_and_playId']+'_'+tracking_combined_temp['nflId'].fillna(0).astype(int).astype(str)
  tracking_combined_temp = tracking_combined_temp[tracking_combined_temp['game_play_nflId'].isin(player_plays_with_motion_by_coverage.game_play_nflId)]

  tracking_combined_temp['nflId'] = tracking_combined_temp['nflId'].fillna(0).astype(int)

  # tracking_combined_temp['game_play_frameId'] = tracking_combined_temp['nflId'].fillna(0).astype(int)

  tracking_combined_keyevents = tracking_combined_temp[tracking_combined_temp.event.notnull()]

  temp=tracking_combined_keyevents[(tracking_combined_keyevents['event'].str.contains('set'))|(tracking_combined_keyevents.event.str.contains('snap'))][['frameId','game_and_playId','game_play_nflId','event']]
  temp_counts=temp.game_play_nflId.value_counts()
  temp=temp[temp['game_play_nflId'].isin(temp_counts[temp_counts>1].index)]

  #update dfs
  # player_plays_combined=player_plays_with_motion_by_coverage[player_plays_with_motion_by_coverage['gameId'].isin(games_combined.gameId)]
  # plays_combined=plays_with_motion_by_coverage[plays_with_motion_by_coverage['gameId'].isin(games_combined.gameId)]
  tracking_combined_filtered=tracking_combined_temp[tracking_combined_temp['game_play_nflId'].isin(temp.game_play_nflId)]
  # player_plays_combined=player_plays_combined[player_plays_combined['game_play_nflId'].isin(temp.game_play_nflId)]

  line_set_frames = temp[temp['event'].str.contains('set')][['frameId', 'game_play_nflId']]
  ball_snap_frames = temp[temp['event'].str.contains('snap')][['frameId', 'game_play_nflId']]

  # Rename columns for clarity
  line_set_frames.rename(columns={'frameId': 'linesetframe'}, inplace=True)
  ball_snap_frames.rename(columns={'frameId': 'ballsnapframe'}, inplace=True)
  line_set_frames['linesetframe'] += 8

  # Merge line set and ball snap frames back into the tracking data
  tracking_combined_filtered = tracking_combined_filtered.merge(
      line_set_frames,
      on=['game_play_nflId'],
      how='left'
  ).merge(
      ball_snap_frames,
      on=['game_play_nflId'],
      how='left'
  )
  # Filter rows based on frameId
  tracking_combined_filtered = tracking_combined_filtered[
      (tracking_combined_filtered['frameId'] >= tracking_combined_filtered['linesetframe']) &
      (tracking_combined_filtered['frameId'] <= tracking_combined_filtered['ballsnapframe'])
  ]

  # #Filter out plays that are too slow to be considered motion
  tracking_combined_filtered['snap_group'] = tracking_combined_filtered['frameId'].eq(tracking_combined_filtered.linesetframe).cumsum()
  group_max_speed = tracking_combined_filtered.groupby('snap_group')['s'].transform('max')
  # # Create the 'max_speed' column by aligning the max speeds back to the original DataFrame
  tracking_combined_filtered['max_speed'] = group_max_speed
  tracking_combined_filtered = tracking_combined_filtered[tracking_combined_filtered['max_speed']>=0.75]

  # #filter out frames before "motion"
  tracking_combined_filtered['motion_flag'] = (tracking_combined_filtered['s'] >= 0.75)
  tracking_combined_filtered['motion_group'] = tracking_combined_filtered['frameId'].eq(tracking_combined_filtered.linesetframe).cumsum()
  first_occur_df = tracking_combined_filtered[tracking_combined_filtered['motion_flag']].groupby('motion_group')['frameId'].min()
  tracking_combined_filtered['first_occur'] = tracking_combined_filtered['motion_group'].map(first_occur_df)
  tracking_combined_filtered = tracking_combined_filtered[
      tracking_combined_filtered['frameId'] >= tracking_combined_filtered['first_occur']
  ]

  # Drop temporary columns if not needed
  tracking_combined_filtered.drop(columns=['motion_flag', 'motion_group','snap_group'], inplace=True)


  # tracking_week_1_filtered['delta_x']=
  tracking_combined_filtered['x_diff']=tracking_combined_filtered['x'].diff().fillna(0)
  tracking_combined_filtered['y_diff']=tracking_combined_filtered['y'].diff().fillna(0)
  # new_arr=[]
  # count=-1
  tracking_combined_filtered.loc[tracking_combined_filtered['frameId'] == tracking_combined_filtered['first_occur'], ['x_diff', 'y_diff']] = 0
  tracking_combined_filtered['distance'] = (tracking_combined_filtered['x_diff']**2 + tracking_combined_filtered['y_diff']**2)**0.5


  tracking_combined_filtered['total_distance_on_play']=tracking_combined_filtered.groupby('game_play_nflId')['distance'].transform(sum)
  tracking_combined_filtered=tracking_combined_filtered[tracking_combined_filtered['total_distance_on_play']>0.75]
  # tracking_combined_filtered = tracking_combined_filtered.drop(columns=['total_distance'])

  # total displacement
  tracking_combined_displacement = tracking_combined_filtered[(tracking_combined_filtered['frameId'] == tracking_combined_filtered['first_occur']) | (tracking_combined_filtered['frameId'] == tracking_combined_filtered['ballsnapframe'])]
  tracking_combined_displacement = tracking_combined_displacement[tracking_combined_displacement['game_play_nflId'].isin(tracking_combined_displacement['game_play_nflId'].value_counts()[tracking_combined_displacement['game_play_nflId'].value_counts() > 1].index)]
  tracking_combined_displacement['x_displacement']=tracking_combined_displacement.groupby('game_play_nflId')['x'].diff()
  tracking_combined_displacement['y_displacement']=tracking_combined_displacement.groupby('game_play_nflId')['y'].diff()
  tracking_combined_displacement=tracking_combined_displacement[tracking_combined_displacement['frameId']==tracking_combined_displacement['ballsnapframe']]
  tracking_combined_displacement['total_displacement']=(tracking_combined_displacement['x_displacement']**2+tracking_combined_displacement['y_displacement']**2)**0.5
  tracking_combined_displacement=tracking_combined_displacement[tracking_combined_displacement['total_displacement']>0.5]

  tracking_combined_filtered=tracking_combined_filtered.merge(tracking_combined_displacement[['game_play_nflId','x_displacement','y_displacement','total_displacement']],on=['game_play_nflId'])


  #plays combined
  plays_combined=plays_with_motion_by_coverage[plays_with_motion_by_coverage['game_and_playId'].isin(tracking_combined_filtered.game_and_playId)]

  #add defensive team
  merged_data = tracking_combined_filtered.merge(
      plays_combined[['game_and_playId', 'defensiveTeam']],
      on=['game_and_playId'],  # Merge on these keys
      how='left'               # Use 'left' to keep all rows in the tracking data
  )
  tracking_combined_filtered['defensiveTeam'] = merged_data['defensiveTeam'].values

  #add defensive team
  merged_data = tracking_combined_displacement.merge(
      plays_combined[['game_and_playId', 'defensiveTeam']],
      on=['game_and_playId'],  # Merge on these keys
      how='left'               # Use 'left' to keep all rows in the tracking data
  )
  tracking_combined_displacement['defensiveTeam'] = merged_data['defensiveTeam'].values

  return tracking_combined_filtered, tracking_combined_displacement, plays_combined


#tracking data for defenders prep
def tracking_defenders(tracking_combined,tracking_combined_filtered,plays_combined):
  #defense distance
  tracking_combined_defenders = tracking_combined[tracking_combined['game_and_playId'].isin(tracking_combined_filtered.game_and_playId.unique())]
  merge_temp = tracking_combined_defenders.merge(
      plays_combined[['game_and_playId', 'defensiveTeam']],
      on=['game_and_playId'], how='left')
  tracking_combined_defenders = merge_temp[merge_temp['club'] == merge_temp['defensiveTeam']]

  tracking_combined_defenders['game_play_frameId'] = tracking_combined_defenders['game_and_playId'] + '_' + tracking_combined_defenders['frameId'].astype(str)
  tracking_combined_filtered['game_play_frameId'] = tracking_combined_filtered['game_and_playId'] + '_' + tracking_combined_filtered['frameId'].astype(str)
  #merge by motion frame
  tracking_combined_defenders['original_order']=tracking_combined_defenders.index
  motion_frames = tracking_combined_filtered[['game_play_frameId']].drop_duplicates()
  tracking_combined_defenders = tracking_combined_defenders.merge(motion_frames,on='game_play_frameId',sort='False')
  tracking_combined_defenders = tracking_combined_defenders.sort_values('original_order')
  tracking_combined_defenders = tracking_combined_defenders.set_index('original_order')

  tracking_combined_defenders['nflId'] = tracking_combined_defenders['nflId'].fillna(0).astype(int)
  tracking_combined_defenders['game_play_nflId'] = tracking_combined_defenders['game_and_playId'] + '_' + tracking_combined_defenders['nflId'].astype(str)
  tracking_combined_defenders['x_diff']=tracking_combined_defenders['x'].diff().fillna(0)
  tracking_combined_defenders['y_diff']=tracking_combined_defenders['y'].diff().fillna(0)
  tracking_combined_defenders.loc[
      tracking_combined_defenders.groupby('game_play_nflId').head(1).index,
      ['x_diff', 'y_diff']
  ] = 0
  tracking_combined_defenders['distance'] = (tracking_combined_defenders['x_diff']**2 + tracking_combined_defenders['y_diff']**2)**0.5

  return tracking_combined_defenders


def temp_df_function(tracking_combined_displacement,tracking_combined_defenders):
  #temp dataframe for defense
  tracking_combined_displacement['offenseId'] = tracking_combined_displacement['nflId']
  temp_df = tracking_combined_defenders.merge(tracking_combined_displacement[['game_and_playId','offenseId','first_occur','ballsnapframe']],on=['game_and_playId'],how='inner')
  temp_df=temp_df[(temp_df['frameId']==temp_df['first_occur']) | (temp_df['frameId']==temp_df['ballsnapframe'])]
  temp_df['game_play_nfl_offenseId']=temp_df['game_play_nflId']+'_'+temp_df['offenseId'].astype(str)
  temp_df['x_displacement']=temp_df.groupby('game_play_nfl_offenseId')['x'].diff()
  temp_df['y_displacement']=temp_df.groupby('game_play_nfl_offenseId')['y'].diff()
  temp_df=temp_df[temp_df['frameId']==temp_df['ballsnapframe']]
  temp_df['total_displacement']=(temp_df['x_displacement']**2+temp_df['y_displacement']**2)**0.5
  temp_df2 = tracking_combined_defenders.merge(temp_df[['game_play_nflId','offenseId','first_occur','ballsnapframe']],on=['game_play_nflId'],how='inner')
  return temp_df, temp_df2
  # Dataframe without duplicates
  # deduplicated_df = temp_df.sort_values(by=['game_play_nflId', 'first_occur'])
  # deduplicated_df = deduplicated_df.drop_duplicates(subset=['game_play_nflId'], keep='first')



    