import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import nfl_data_py as nfl
from IPython.display import HTML, display
import statsmodels.api as sm
from scipy import stats 
import certifi
import ssl

def plot_distance_ratio(tracking_combined_filtered,tracking_combined_defenders,plays_combined):
    #offense distance
    off_motion_by_defense=tracking_combined_filtered.groupby('game_play_nflId').agg({'distance':'sum','defensiveTeam':'first'})
    off_motion_by_defense['distance_squared']=off_motion_by_defense['distance']**2
    off_motion_by_defense=off_motion_by_defense.groupby('defensiveTeam').sum()

    
    #defensive motion
    defense_distance=tracking_combined_defenders.groupby('game_play_nflId').agg({'distance':'sum','club':'first'})
    defense_distance['distance_squared']=defense_distance['distance']**2

    defense_distance=defense_distance.groupby('club').sum()
    # print(defense_distance['distance_squared'])
    distance_ratio=defense_distance['distance_squared']/off_motion_by_defense['distance_squared']
    epa_allowed=plays_combined.groupby('defensiveTeam')['expectedPointsAdded'].mean()

    #plot data
    # plt.figure(figsize=(10,6))
    seaborn_distance_ratio = pd.DataFrame({'distance_ratio': distance_ratio, 'epa_allowed':epa_allowed})
    sns.lmplot(x='distance_ratio',y='epa_allowed',data=seaborn_distance_ratio,ci=None)
    plt.title('EPA/Play Allowed Based on Distance Ratio Score By Team (Plays With Motion)')
    plt.xlabel('Distance Ratio')
    plt.ylabel('Average EPA Allowed')
    for i in range(32):
        plt.text(seaborn_distance_ratio.distance_ratio[i] + 0.005, seaborn_distance_ratio.epa_allowed[i], seaborn_distance_ratio.index[i], fontsize=8)
    slope, intercept = np.polyfit(seaborn_distance_ratio.distance_ratio, seaborn_distance_ratio.epa_allowed, 1)
    x_values = np.array([seaborn_distance_ratio.distance_ratio.min(), seaborn_distance_ratio.distance_ratio.max()])
    y_values = slope * x_values + intercept
    plt.plot(x_values, y_values, color='red', label='Best-fit line')
    # model=LinearRegression()
    # model.fit(seaborn_distance_ratio[['distance_ratio']],seaborn_distance_ratio['epa_allowed'])
    # r2 = r2_score(seaborn_distance_ratio['epa_allowed'], model.predict(seaborn_distance_ratio[['distance_ratio']]))
    # print(f"R-squared: {r2}")
    equation_text = f"y = {slope:.2f}x + {intercept:.2f}"
    plt.text(0.01, 0.97, equation_text, fontsize=10, weight='bold',transform=plt.gca().transAxes)
    plt.legend(loc='upper left',bbox_to_anchor=(0.85,1.05))
    # plt.tight_layout()
    return distance_ratio

#distance projection ratio
def plot_distance_projection_ratio(tracking_combined_filtered,temp_df2,plays_combined):
    #defense distance
  # five_largest=temp_df.groupby(['game_and_playId','offenseId'],group_keys=False).apply(lambda group: group.nlargest(3,'distance'))
  # temp_df2 = tracking_combined_defenders.merge(temp_df[['game_play_nflId','offenseId','first_occur','ballsnapframe']],on=['game_play_nflId'],how='inner')
  temp_df2=temp_df2[(temp_df2['frameId']>=temp_df2['first_occur']) & (temp_df2['frameId']<=temp_df2['ballsnapframe'])]
  tracking_combined_filtered['off_xdiff'] = tracking_combined_filtered['x_diff']
  tracking_combined_filtered['off_ydiff'] = tracking_combined_filtered['y_diff']
  tracking_combined_filtered['off_distance'] = tracking_combined_filtered['distance']
  temp_df2['game_play_offense_frameId']=temp_df2['game_and_playId']+'_'+temp_df2['offenseId'].astype(str)+temp_df2['frameId'].astype(str)
  tracking_combined_filtered['game_play_offense_frameId']=tracking_combined_filtered['game_and_playId']+'_'+tracking_combined_filtered['nflId'].astype(str)+tracking_combined_filtered['frameId'].astype(str)
  tracking_combined_filtered['off_y'] = tracking_combined_filtered['y']

  # Optimize merge
  temp_df2['game_play_nfl_offenseId']=temp_df2['game_play_nflId']+'_'+temp_df2['offenseId'].astype(str)
  temp_df2['original_order'] = temp_df2.index
  temp_df2.set_index(['game_play_offense_frameId'], inplace=True)
  temp_df2=temp_df2[['club','x_diff','y_diff', 'game_play_nfl_offenseId','original_order','y']]
  tracking_combined_filtered.set_index(['game_play_offense_frameId'], inplace=True)

  # Merge and reset index
  distance_projection = temp_df2.join(tracking_combined_filtered[['off_xdiff', 'off_ydiff','off_distance','off_y']], how='inner').reset_index()
  distance_projection=distance_projection.sort_values('original_order')
  distance_projection=distance_projection.set_index('original_order')
  tracking_combined_filtered.reset_index(inplace=True)
  # Compute dot product using NumPy
  distance_projection['ydiff_inv'] = (1 / ((abs(distance_projection['off_y'] - distance_projection['y'])).replace(0,2))).clip(upper=0.5)
  distance_projection['dot_product_projection'] = np.einsum(
      'ij,ij->i',
      distance_projection[['x_diff', 'y_diff']].values,
      distance_projection[['off_xdiff', 'off_ydiff']].values
  ) / distance_projection['off_distance']
  distance_projection['dot_product_projection']=distance_projection['dot_product_projection'].clip(lower=0)
  distance_projection['dot_product_projection_squared']=distance_projection['dot_product_projection']**2
  distance_projection['weighted_projection_squared']=distance_projection['dot_product_projection_squared']*distance_projection['ydiff_inv']

  # distance_projection=distance_projection.groupby('game_play_nfl_offenseId').agg({'dot_product_projection':'sum','club':'first'})
  # distance_projection['dot_product_projection_squared']=distance_projection['dot_product_projection']**2
  # distance_projection=distance_projection.groupby('club').sum()
  # distance_projection_sums=distance_projection['dot_product_projection_squared']
  distance_projection_sums=distance_projection.groupby('club')['weighted_projection_squared'].sum()
  #offense distance
  tracking_combined_filtered['distance_squared']=tracking_combined_filtered['distance']**2

  # off_motion_by_defense=tracking_combined_filtered.groupby('game_play_nflId').agg({'distance':'sum','defensiveTeam':'first'})
  # off_motion_by_defense['distance_squared']=off_motion_by_defense['distance']**2
  distance=tracking_combined_filtered.groupby('defensiveTeam')['distance_squared'].sum()
  # distance=off_motion_by_defense['distance_squared']
  projection_ratio=distance_projection_sums/distance
  epa_allowed=plays_combined.groupby('defensiveTeam')['expectedPointsAdded'].mean()

  #plot results
  # plt.figure()
  seaborn_projection_ratio = pd.DataFrame({'projection_ratio': projection_ratio, 'epa_allowed':epa_allowed})
  sns.lmplot(x='projection_ratio',y='epa_allowed',data=seaborn_projection_ratio,ci=None)
  #plot details
  plt.title('EPA/Play Allowed Based on Distance Scalar Projection Ratio By Team (Plays With Motion)')
  plt.xlabel('Dist-SPR (Distance Scalar Projection Ratio)')
  plt.ylabel('Average EPA Allowed')
  for i in range(32):
      plt.text(seaborn_projection_ratio.projection_ratio[i] + 0.002, seaborn_projection_ratio.epa_allowed[i], seaborn_projection_ratio.index[i], fontsize=8)
  slope, intercept = np.polyfit(seaborn_projection_ratio.projection_ratio, seaborn_projection_ratio.epa_allowed, 1)
  x_values = np.array([seaborn_projection_ratio.projection_ratio.min(), seaborn_projection_ratio.projection_ratio.max()])
  y_values = slope * x_values + intercept
  plt.plot(x_values, y_values, color='red', label='Best-fit line')
  # model=LinearRegression()
  # model.fit(seaborn_projection_ratio[['projection_ratio']],seaborn_projection_ratio['epa_allowed'])
  # r2 = r2_score(seaborn_projection_ratio['epa_allowed'], model.predict(seaborn_projection_ratio[['projection_ratio']]))
  # print(f"R-squared: {r2}")
  equation_text = f"y = {slope:.2f}x + {intercept:.2f}"
  plt.text(0.01, 0.97, equation_text, fontsize=10, weight='bold', transform=plt.gca().transAxes)
  plt.legend(loc='upper left',bbox_to_anchor=(0.85,1.05))
  # plt.tight_layout()
  return projection_ratio


#displacement projection ratio
def plot_disp_projection_ratio(tracking_combined_displacement, temp_df,plays_combined):
  tracking_combined_displacement['offense_xdisplacement'] = tracking_combined_displacement['x_displacement']
  tracking_combined_displacement['offense_ydisplacement'] = tracking_combined_displacement['y_displacement']
  tracking_combined_displacement['offense_displacement'] = tracking_combined_displacement['total_displacement']
  tracking_combined_displacement['off_y'] = tracking_combined_displacement['y']

  temp_df['original_index']=temp_df.index
  temp_df=temp_df.merge(tracking_combined_displacement[['game_and_playId','offenseId','offense_xdisplacement','offense_ydisplacement','offense_displacement','off_y']],on=['game_and_playId','offenseId'],sort=False)
  temp_df=temp_df.sort_values('original_index')
  temp_df=temp_df.set_index('original_index')
  dot_product_df = temp_df.copy()
  dot_product_df['projection'] = (dot_product_df['x_displacement'] * dot_product_df['offense_xdisplacement'] + dot_product_df['y_displacement'] * dot_product_df['offense_ydisplacement']) / dot_product_df['offense_displacement']
  dot_product_df['projection']=dot_product_df['projection'].clip(lower=0)
  dot_product_df['projection_squared'] = dot_product_df['projection']**2
  dot_product_df['ydiff_inv'] = (1 / ((abs(dot_product_df['off_y'] - dot_product_df['y'])).replace(0,1))).clip(upper=1)
  dot_product_df['weighted_projection_squared']=dot_product_df['projection_squared']*dot_product_df['ydiff_inv']


  #defense over offense
  tracking_combined_displacement['displacement_squared']=tracking_combined_displacement['total_displacement']**2
  off_displacement=tracking_combined_displacement.groupby('defensiveTeam')['total_displacement'].sum()
  off_displacement_squared=tracking_combined_displacement.groupby('defensiveTeam')['displacement_squared'].sum()
  defense_dotproduct=dot_product_df.groupby('club')['weighted_projection_squared'].sum()
  dotproduct_ratio=defense_dotproduct/off_displacement_squared
  epa_allowed=plays_combined.groupby('defensiveTeam')['expectedPointsAdded'].mean()

  #plot results
  # plt.figure()
  seaborn_dotproduct_ratio = pd.DataFrame({'dotproduct_ratio': dotproduct_ratio, 'epa_allowed':epa_allowed})
  sns.lmplot(x='dotproduct_ratio',y='epa_allowed',data=seaborn_dotproduct_ratio,ci=None)
  #add plot details
  #plot details
  plt.title('EPA/Play Allowed Based on Displacement Scalar Projection Ratio Score By Team (Plays With Motion)')
  plt.xlabel('Disp-SPR (Displacement Scalar Projection Ratio)')
  plt.ylabel('Average EPA Allowed')
  for i in range(32):
      plt.text(seaborn_dotproduct_ratio.dotproduct_ratio[i] + 0.005, seaborn_dotproduct_ratio.epa_allowed[i], seaborn_dotproduct_ratio.index[i], fontsize=8)
  slope, intercept = np.polyfit(seaborn_dotproduct_ratio.dotproduct_ratio, seaborn_dotproduct_ratio.epa_allowed, 1)
  x_values = np.array([seaborn_dotproduct_ratio.dotproduct_ratio.min(), seaborn_dotproduct_ratio.dotproduct_ratio.max()])
  y_values = slope * x_values + intercept
  plt.plot(x_values, y_values, color='red', label='Best-fit line')

  # model=LinearRegression()
  # model.fit(seaborn_dotproduct_ratio[['dotproduct_ratio']],seaborn_dotproduct_ratio['epa_allowed'])
  # r2 = r2_score(seaborn_dotproduct_ratio['epa_allowed'], model.predict(seaborn_dotproduct_ratio[['dotproduct_ratio']]))
  # print(f"R-squared: {r2}")
  equation_text = f"y = {slope:.2f}x + {intercept:.2f}"
  plt.text(0.01, 0.97, equation_text, fontsize=10, weight='bold',transform=plt.gca().transAxes)
  plt.legend(loc='upper left',bbox_to_anchor=(0.92,1.05))
  # plt.tight_layout()
  return dotproduct_ratio

  

def plot_sum_scaled(ratio1, ratio2,plays_combined):
  #distance + projection ratio scaled
  ratio1_scaled = (ratio1 - ratio1.min()) / (ratio1.max() - ratio1.min())
  ratio2_scaled = (ratio2 - ratio2.min()) / (ratio2.max() - ratio2.min())
  epa_allowed=plays_combined.groupby('defensiveTeam')['expectedPointsAdded'].mean()

  #plot results
  # plt.figure()
  sum_ratios=ratio1_scaled+ratio2_scaled
  seaborn_projection_ratio = pd.DataFrame({'sum_ratios': sum_ratios, 'epa_allowed':epa_allowed})
  sns.lmplot(x='sum_ratios',y='epa_allowed',data=seaborn_projection_ratio,ci=None)
  #plot details
  plt.title('EPA/Play Allowed Based on Total Reaction Score By Team (Plays With Motion)')
  plt.xlabel('TRS (Total Reaction Score)')
  plt.ylabel('Average EPA Allowed')
  for i in range(32):
      plt.text(seaborn_projection_ratio.sum_ratios[i] + 0.005, seaborn_projection_ratio.epa_allowed[i], seaborn_projection_ratio.index[i], fontsize=8)
  slope, intercept = np.polyfit(seaborn_projection_ratio.sum_ratios, seaborn_projection_ratio.epa_allowed, 1)
  x_values = np.array([seaborn_projection_ratio.sum_ratios.min(), seaborn_projection_ratio.sum_ratios.max()])
  y_values = slope * x_values + intercept
  plt.plot(x_values, y_values, color='red', label='Best-fit line')
  model=LinearRegression()
  model.fit(seaborn_projection_ratio[['sum_ratios']],seaborn_projection_ratio['epa_allowed'])
  r2 = r2_score(seaborn_projection_ratio['epa_allowed'], model.predict(seaborn_projection_ratio[['sum_ratios']]))
  r = r2**0.5
  print(f"R-value: {r}")
  equation_text = f"y = {slope:.2f}x + {intercept:.2f}"
  plt.text(0.01, 0.97, equation_text, fontsize=10, weight='bold',transform=plt.gca().transAxes)
  plt.legend(loc='upper left',bbox_to_anchor=(0.85,1.05))
  # plt.tight_layout()
  return sum_ratios

#plot bar graph
def plot_bargraph_nested(values):
  x = [0,0.35,1,1.35,2,2.35]
  colors=['blue','red','blue','red','blue','red']
  labels_shortened = ['Top 5','Bottom 5','Top 10','Bottom 10','Top Half','Bottom Half']
  width = 0.3
  fig,ax = plt.subplots(figsize=(10,5))
  bars1=ax.bar(x, values, width, color=colors )
  ax.set_xlabel('Rankings Based On Total Reaction Score')
  ax.set_ylabel('EPA/Play Allowed')
  ax.set_xticks(x)
  ax.set_xticklabels(labels_shortened)
  ax.set_title('Average EPA/Play allowed based on Total Reaction Score Ranking (Plays With Motion)')
  # ax.legend()

  for bar in bars1:
    if bar.get_height() < 0:
      ax.text(bar.get_x()+bar.get_width()/2, 0, f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)
    else:
      ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)
  # ax.spines['bottom'].set_position(('data',0))
#   plt.show()

def plot_bargraph(x_ratios,plays_combined):
  dr_med=x_ratios.median()
  lower_half=x_ratios[x_ratios < dr_med]
  upper_half=x_ratios[x_ratios > dr_med]
  #bottom ten and top ten
  top_ten=x_ratios.nlargest(10)
  bottom_ten=x_ratios.nsmallest(10)

  #bottom and top five
  top_five=x_ratios.nlargest(5)
  bottom_five=x_ratios.nsmallest(5)

  #epa by all plays 
  #bottom and top ten
  total_topten_mean=plays_combined[plays_combined['defensiveTeam'].isin(top_ten.index)]['expectedPointsAdded'].mean()
  total_botten_mean=plays_combined[plays_combined['defensiveTeam'].isin(bottom_ten.index)]['expectedPointsAdded'].mean()
  #bottom and top five
  total_topfive_mean=plays_combined[plays_combined['defensiveTeam'].isin(top_five.index)]['expectedPointsAdded'].mean()
  total_botfive_mean=plays_combined[plays_combined['defensiveTeam'].isin(bottom_five.index)]['expectedPointsAdded'].mean()
  #bottom and top half
  total_tophalf_mean=plays_combined[plays_combined['defensiveTeam'].isin(upper_half.index)]['expectedPointsAdded'].mean()
  total_bothalf_mean=plays_combined[plays_combined['defensiveTeam'].isin(lower_half.index)]['expectedPointsAdded'].mean()

  values=[total_topfive_mean,total_botfive_mean,total_topten_mean,total_botten_mean,total_tophalf_mean,total_bothalf_mean]
  # plt.figure()
  plot_bargraph_nested(values)

def render_logo(logo_url):
    return f'<img src="{logo_url}" width="18">'


def trs_order(distspr, dispspr, sum_ratios):
  ssl_context = ssl.create_default_context(cafile=certifi.where())

  team_logos = nfl.import_team_desc()[['team_abbr', 'team_logo_espn']]
  team_logos = team_logos.set_index('team_abbr')['team_logo_espn'].to_dict()
  distspr = (distspr - distspr.min()) / (distspr.max() - distspr.min())
  dispspr = (dispspr - dispspr.min()) / (dispspr.max() - dispspr.min())
  df = pd.DataFrame({'TRS': sum_ratios.round(2), "DispSPR Scaled": dispspr.round(2),"DistSPR Scaled": distspr.round(2), 'Team':distspr.index})
  df['Logo'] = df['Team'].map(team_logos)
  df['Logo'] = df['Logo'].apply(render_logo)
  # Move the logo column to the front for better display
  df = df[['Team', 'Logo', 'TRS', 'DispSPR Scaled', 'DistSPR Scaled']]
  df.sort_values('TRS',ascending=False,inplace=True)
  # Convert the DataFrame to an HTML table with images
  # display(HTML(df.to_html(escape=False)))
  html_code=df.to_html(escape=False,index=False,border=1)
  styled_html = f"""
  {html_code}
  <style>
    table {{
        border-collapse: collapse;  /* Fixed by using double braces */
        width: 50%;
    }}
    th, td {{
        text-align: center;
        padding: 2px;
        font-weight: bold;
        font-size: 11px;
        line-height: 1.2;
    }}
  </style>
  """
 
  with open('styled_nfl_table.html', 'w') as file:
      file.write(styled_html)

  HTML(styled_html)
  print("HTML file successfully exported as 'styled_nfl_table.html'")
