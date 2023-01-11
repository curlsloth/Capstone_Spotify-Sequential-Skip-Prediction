import pandas as pd
import numpy as np


def get_ground_truth(test_output):

    ground_truths = [] 
    df = test_output.copy()
    df[['session_position','session_length']] = df[['session_position','session_length']].astype('int64')
    df = df[['session_id','skip_2','session_position','session_length']].loc[df['session_position']*2 > df['session_length']]
    df = df.reset_index()
    current_index = 0
    while current_index < len(df):
        partial_length = df['session_length'].iloc[current_index]-df['session_position'].iloc[current_index]+1
        session_skips = list(df.loc[current_index:current_index+partial_length-1, 'skip_2'])
        ground_truths.append(session_skips)
        current_index += partial_length 
    return ground_truths


def get_submission(test_output):
    
    submission = []
    for s in test_output['session_id'].unique():
        submission.append(np.array(test_output['pred'][test_output['session_id']==s]*1))
    return submission


def evaluate(submission,groundtruth):
    ap_sum = 0.0
    first_pred_acc_sum = 0.0
    counter = 0
    for sub, tru in zip(submission, groundtruth):
        if len(sub) != len(tru):
            raise Exception('Line {} should contain {} predictions, but instead contains '
                            '{}'.format(counter+1,len(tru),len(sub)))
        ap_sum += ave_pre(sub,tru,counter)
        first_pred_acc_sum += sub[0] == tru[0]
        counter+=1
    ap = ap_sum/counter
    first_pred_acc = first_pred_acc_sum/counter
    return ap,first_pred_acc


def ave_pre(submission,groundtruth,counter):
    s = 0.0
    t = 0.0
    c = 1.0
    for x, y in zip(submission, groundtruth):
        if x != 0 and x != 1:
            raise Exception('Invalid prediction in line {}, should be 0 or 1'.format(counter))
        if x==y:
            s += 1.0
            t += s / c
        c += 1
    return t/len(groundtruth)


def precise_weight_perSession(df_test_pred):
    # calculate the average weight per session position (counted from the 1st track being predicted) * session length
    def new_weight(n):
            import itertools
            t = [1] * n
            p_list = list(itertools.product([0, 1], repeat=n))
            dd = pd.DataFrame(pd.Series(p_list).tolist())

            ap_list = []
            for p in p_list:
                ap_list.append(ave_pre(p,t,0))
            dd['ap'] = ap_list

            mean_diff = []
            for nn in range(n):
                mean_diff.append(dd.groupby(by = nn)['ap'].mean().diff()[1])
            
            return mean_diff
    
    # calculate the average weight per session position (counted from the 1st track being predicted), average over session length
    pp = pd.DataFrame([[np.nan] * 10 for i in range(6)])
    c = 0
    for n in range(5,11):
        pp.iloc[c,:n] = new_weight(n)
        c += 1

    ave_weight_list = pp.iloc[-1]
    
    a = df_test_pred['session_position'] - np.floor(df_test_pred['session_length']/2) - 1 # for example, the 6th track in a 10-track session, a will be 6-(10/2)-1 = 0
    if (a<0).sum()>0:
        raise Exception('***a number equal or below 0***')
        
    # as the final AA is calculated by averaging the ap per session, each track in the shorter session will have higher contribution to the final AA
    weight = [i / j for i, j in zip(ave_weight_list[a], np.floor(df_test_pred['session_length']/2))]
    return weight




def get_sim(df_hist, df_lookup, sim_file_list, score_name_list):
    df_hist['ListenYes'] = (df_hist['skip_2'] == False)*1
    df_hist['ListenYes'].replace(0, -1, inplace = True)
    df_hist = df_hist.groupby(['session_id', 'clus']).agg({'ListenYes':['sum']})
    df_hist = df_hist.reset_index()
    df_hist.columns = df_hist.columns.droplevel(level = 1) # take out the unwanted level
    df_pivot = pd.pivot_table(df_hist, values = 'ListenYes',index='session_id', columns='clus')
    df_pivot = df_pivot.fillna(0)
    
    
    for sim_file, score_name in zip(sim_file_list, score_name_list):
        sim_matrix = pd.read_csv(sim_file).drop(columns=['Unnamed: 0'])
        sim_matrix.columns = list(map(str, range(0,len(sim_matrix))))
        df_sim_session = df_pivot.dot(sim_matrix)/sim_matrix.sum()
        
        df_lookup[score_name] = df_sim_session.lookup(df_lookup['session_id'],df_lookup['clus'].astype(str))
    
    return df_lookup


def prep_dfs(file, tf_df, kmean_df):
    log_df = pd.read_csv(file)
    log_df = log_df.merge(kmean_df)

    log_df_1 = log_df.loc[log_df['session_position']<=(log_df['session_length']/2)]
    log_df_1['hour_of_day'] = log_df_1['hour_of_day'].astype('float')
    log_df_1['premium'] = log_df_1['premium'].astype('bool')
    log_df_1['weekday'] = log_df_1['date'].astype('datetime64[ns]').dt.dayofweek
    log_df_1 = log_df_1.drop(columns = ['date'])
    log_df_1 = pd.get_dummies(log_df_1, columns=['hist_user_behavior_reason_end', 'hist_user_behavior_reason_start', 'context_type','weekday'], dtype = 'bool')
    log_df_1 = log_df_1.merge(tf_df.drop(columns = ['time_signature','mode','key']))
    
                     
    col_bool = log_df_1.columns[log_df_1.dtypes=='bool']
    col_nonbool = log_df_1.columns[log_df_1.dtypes!='bool'].drop(['session_id','track_id_clean','clus'])
    
    # the non-convertable values will be set to 0
    log_df_1[col_nonbool] = log_df_1[col_nonbool].apply(pd.to_numeric, errors='coerce', downcast = 'float').fillna(0).astype('float32')

    # aggregate the track history where ['skip_2']==True
    log_df_1_summary_skip2True = pd.concat([log_df_1.loc[log_df_1['skip_2']==True].groupby(['session_id'])[col_bool].agg(['mean']), 
                                            log_df_1.loc[log_df_1['skip_2']==True].groupby(['session_id'])[col_nonbool].agg(['mean', 'std', 'median'])],
                                            axis = 1)
    log_df_1_summary_skip2True.columns = log_df_1_summary_skip2True.columns.get_level_values(0)+'_sk2True_'+log_df_1_summary_skip2True.columns.get_level_values(1)
    
    # aggregate the track history where ['skip_2']==False
    log_df_1_summary_skip2False = pd.concat([log_df_1.loc[log_df_1['skip_2']==False].groupby(['session_id'])[col_bool].agg(['mean']), 
                                             log_df_1.loc[log_df_1['skip_2']==False].groupby(['session_id'])[col_nonbool].agg(['mean', 'std', 'median'])],
                                             axis = 1)
    log_df_1_summary_skip2False.columns = log_df_1_summary_skip2False.columns.get_level_values(0)+'_sk2False_'+log_df_1_summary_skip2False.columns.get_level_values(1)
    
    
    log_df_history = log_df_1[['session_id','track_id_clean','skip_2','clus']]


    half_cut = log_df['session_length']/2

    # need to at least include 2 trials, otherwise the log_df_1_summary will confound with all the tracks in the same session

    #1st trial in the 2nd half
    log_df_2_1 = log_df.loc[(log_df['session_position']>half_cut) & (log_df['session_position']<=half_cut+1)]
    log_df_2_1 = log_df_2_1[['session_id','track_id_clean','skip_2','session_position','session_length','clus']]
    log_df_2_1['weight'] = 1

    #2nd trial in the 2nd half
    log_df_2_2 = log_df.loc[(log_df['session_position']>half_cut+1) & (log_df['session_position']<=half_cut+2)]
    log_df_2_2 = log_df_2_2[['session_id','track_id_clean','skip_2','session_position','session_length','clus']]
    log_df_2_2['weight'] = 0.75
    
    #3rd trial in the 2nd half
    log_df_2_3 = log_df.loc[(log_df['session_position']>half_cut+2) & (log_df['session_position']<=half_cut+3)]
    log_df_2_3 = log_df_2_3[['session_id','track_id_clean','skip_2','session_position','session_length','clus']]
    log_df_2_3['weight'] = 0.62
    
    #4th trial in the 2nd half
    log_df_2_4 = log_df.loc[(log_df['session_position']>half_cut+3) & (log_df['session_position']<=half_cut+4)]
    log_df_2_4 = log_df_2_4[['session_id','track_id_clean','skip_2','session_position','session_length','clus']]
    log_df_2_4['weight'] = 0.53
    
    #5th trial in the 2nd half
    log_df_2_5 = log_df.loc[(log_df['session_position']>half_cut+4) & (log_df['session_position']<=half_cut+5)]
    log_df_2_5 = log_df_2_5[['session_id','track_id_clean','skip_2','session_position','session_length','clus']]
    log_df_2_5['weight'] = 0.47
    
#     #remaining trials in the 2nd half
#     log_df_2_6 = log_df.loc[(log_df['session_position']>half_cut+5)]
#     log_df_2_6 = log_df_2_6[['session_id','track_id_clean','skip_2','session_position','session_length','clus']]
#     log_df_2_6['weight'] = 0.35

    log_df_2 = pd.concat([log_df_2_1,log_df_2_2,log_df_2_3,log_df_2_4,log_df_2_5])
    log_df_2 = log_df_2.merge(log_df_1_summary_skip2True, on='session_id')
    log_df_2 = log_df_2.merge(log_df_1_summary_skip2False, on='session_id')

    sim_file_list = ['../models/SVD/all_tracks/similarity_for20180918/k300_CanbDist.csv',
                     '../models/SVD/all_tracks/similarity_for20180918/k300_CosSim.csv',
                     '../models/SVD/all_tracks/similarity_for20180918/k300_LinCorr.csv',
                     '../models/SVD/all_tracks/similarity_for20180918/k300_ManhDist.csv',
                     '../models/SVD/all_tracks/similarity_for20180918/k300_HammDist.csv',
                     '../models/SVD/all_tracks/similarity_for20180918/k300_SpearCorr.csv',
                     '../models/SVD/all_tracks/similarity_for20180918/k300_KendCorr.csv',
                     '../models/SVD/all_tracks/similarity_for20180918/k300_ChebDist.csv',
                     '../models/SVD/all_tracks/similarity_for20180918/k300_BrayDist.csv']
    score_name_list = ['CanbDist300', 'CosSim300','LinCorr300','ManhDist300','HammDist300','SpearCorr300','KendCorr300','ChebDist','BrayDist']

    return get_sim(log_df_history, log_df_2, sim_file_list, score_name_list)
