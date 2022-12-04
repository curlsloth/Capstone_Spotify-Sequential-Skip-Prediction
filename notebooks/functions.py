#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def cal_similarMat(df_train):
    import numpy as np
    import pandas as pd
    
    more_sim_index = False # turn it as False to only calculate a few indexes
    
    
#     sessions = list(np.sort(df_train['session_id'].unique())) 
#     tracks = list(df_train['clus'].unique()) 
#     no_skip_2 = (list(df_train['skip_2']==False))*1 # use *1 to convert bool to integer
    
#     DfSessionUnique = []
#     DfSessionUnique = pd.DataFrame(sessions,columns=['sessions'])
    
#     from scipy import sparse
#     from pandas.api.types import CategoricalDtype

#     rows = df_train['session_id'].astype(CategoricalDtype(categories=sessions)).cat.codes # unique sessions (index)

#     # Get the associated row indices
#     cols = df_train['clus'].astype(CategoricalDtype(categories=tracks)).cat.codes # unique tracks (column)
    
    
#     # Get the associated column indices
#     #Compressed Sparse Row matrix
#     listeningSparse = []
#     listeningSparse = sparse.csr_matrix((no_skip_2, (rows, cols)), shape=(len(sessions), len(tracks)))
#     #csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
#     #where data, row_ind and col_ind satisfy the relationship a[row_ind[k], col_ind[k]] = data[k]. , see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html

#     listeningSparse
#     #a sparse matrix is not a pandas dataframe, but sparse matrices are efficient for row slicing and fast matrix vector products
    
    
    df_train['ListenYes'] = (df_train['skip_2'] == False)*1
    
    data2=df_train[['session_id','clus','ListenYes']]

    data2['ListenYes'].replace(0, -1, inplace = True)

    data3 = data2.groupby(['session_id', 'clus']).agg({'ListenYes':['sum']})
    data3 = data3.reset_index()
    data3.columns = data3.columns.droplevel(level = 1) # take out the unwanted level
    
    
    DfMatrix = pd.pivot_table(data3, values='ListenYes', index='session_id', columns='clus')

    DfMatrix=DfMatrix.fillna(0) #NaN values need to get replaced by 0, meaning they have not been listened yet.
    
    DfResetted = DfMatrix.reset_index().rename_axis(None, axis=1) 

    DfTracksListen = DfResetted.drop(columns=['session_id'])

    #Normalization
    import numpy as np
    DfTracksListenNorm = DfTracksListen / np.sqrt(np.square(DfTracksListen).sum(axis=0)) 

    #### similarity and correlation
    # Calculating with Vectors to compute Cosine Similarities
    TrackTrackSim = DfTracksListenNorm.transpose().dot(DfTracksListenNorm) 

    #Another approach to the above would be using correlation
    TrackTrackCorr = DfTracksListenNorm.corr()
    
    
    from scipy.spatial.distance import cdist
    
    #### distances
    # Euclidean distance
    TrackTrackEuclDist = pd.DataFrame(cdist(DfTracksListenNorm.T,DfTracksListenNorm.T, 'euclidean'), index = TrackTrackSim.index, columns = TrackTrackSim.columns)

    
    # Manhattan distance
    TrackTrackManhDist = pd.DataFrame(cdist(DfTracksListenNorm.T,DfTracksListenNorm.T, 'cityblock'), index = TrackTrackSim.index, columns = TrackTrackSim.columns)


   
 

    # Create a place holder matrix for similarities, and fill in the session column
    SessTrackSimilarity = pd.DataFrame(index=DfResetted.index, columns=DfResetted.columns)
    SessTrackSimilarity.iloc[:,:1] = DfResetted.iloc[:,:1]
    SessTrackCorrelation = SessTrackSimilarity.copy()
    SessTrackEuclDist = SessTrackSimilarity.copy()
    SessTrackManhDist = SessTrackSimilarity.copy()
    
    
    
    if more_sim_index:
        #Spearman correlation
        TrackTrackSpearCorr = DfTracksListenNorm.corr(method = 'spearman')

        #Kendall correlation
        TrackTrackKendCorr = DfTracksListenNorm.corr(method = 'kendall')

        # Squared Euclidean distance
        TrackTrackSqEuclDist = pd.DataFrame(cdist(DfTracksListenNorm.T,DfTracksListenNorm.T, 'sqeuclidean'), index = TrackTrackSim.index, columns = TrackTrackSim.columns)
        
        # Canberra distance
        TrackTrackCanbDist = pd.DataFrame(cdist(DfTracksListenNorm.T,DfTracksListenNorm.T, 'canberra'), index = TrackTrackSim.index, columns = TrackTrackSim.columns)

        #### boolean distances
        # Hamming distance
        TrackTrackHammDist = pd.DataFrame(cdist(DfTracksListenNorm.T>0,DfTracksListenNorm.T>0, 'hamming'), index = TrackTrackSim.index, columns = TrackTrackSim.columns)
        
        SessTrackSpearCorr = SessTrackSimilarity.copy()
        SessTrackKendCorr = SessTrackSimilarity.copy()
        SessTrackSqEuclDist = SessTrackSimilarity.copy()
        SessTrackCanbDist = SessTrackSimilarity.copy()
        SessTrackHammDist = SessTrackSimilarity.copy()

    #We now loop through the rows and columns filling in empty spaces with similarity scores.
    
    SessionListening = []
    TrackTopSimilarity = []

    for i in range(0,len(SessTrackSimilarity.index)):
        for j in range(1,len(SessTrackSimilarity.columns)):

            ses = SessTrackSimilarity.index[i]
            tra = SessTrackSimilarity.columns[j]

            SessionListening = DfTracksListen.loc[ses,]
            TrackSimilarity = TrackTrackSim[tra]
            TrackCorrelation = TrackTrackCorr[tra]
            TrackEuclDist = TrackTrackEuclDist[tra]
            TrackManhDist = TrackTrackManhDist[tra]
            
            SessTrackSimilarity.loc[i][j] = sum(SessionListening*TrackSimilarity)/sum(TrackSimilarity)
            SessTrackCorrelation.loc[i][j] = sum(SessionListening*TrackCorrelation)/sum(TrackCorrelation)
            SessTrackEuclDist.loc[i][j] = sum(SessionListening*TrackEuclDist)/sum(TrackEuclDist)
            SessTrackManhDist.loc[i][j] = sum(SessionListening*TrackManhDist)/sum(TrackManhDist)
            
            
            if more_sim_index:
                TrackSpearCorr = TrackTrackSpearCorr[tra]
                TrackKendCorr = TrackTrackKendCorr[tra]
                TrackSqEuclDist = TrackTrackSqEuclDist[tra]
                TrackCanbDist = TrackTrackCanbDist[tra]
                TrackHammDist = TrackTrackHammDist[tra]
                
                SessTrackSpearCorr.loc[i][j] = sum(SessionListening*TrackSpearCorr)/sum(TrackSpearCorr)
                SessTrackKendCorr.loc[i][j] = sum(SessionListening*TrackKendCorr)/sum(TrackKendCorr)
                SessTrackSqEuclDist.loc[i][j] = sum(SessionListening*TrackSqEuclDist)/sum(TrackSqEuclDist)
                SessTrackCanbDist.loc[i][j] = sum(SessionListening*TrackCanbDist)/sum(TrackCanbDist)
                SessTrackHammDist.loc[i][j] = sum(SessionListening*TrackHammDist)/sum(TrackHammDist)


    
    
    SessTrackSimilarity.set_index('session_id', inplace = True)
    SessTrackCorrelation.set_index('session_id', inplace = True)
    SessTrackEuclDist.set_index('session_id', inplace = True)
    SessTrackManhDist.set_index('session_id', inplace = True)
    
    if more_sim_index:
        SessTrackSpearCorr.set_index('session_id', inplace = True)
        SessTrackKendCorr.set_index('session_id', inplace = True)
        SessTrackSqEuclDist.set_index('session_id', inplace = True)
        SessTrackCanbDist.set_index('session_id', inplace = True)
        SessTrackHammDist.set_index('session_id', inplace = True)
        
    
    if more_sim_index:
        sim_output = [SessTrackSimilarity, SessTrackCorrelation, SessTrackSpearCorr, SessTrackKendCorr, SessTrackEuclDist, SessTrackSqEuclDist, SessTrackManhDist, SessTrackCanbDist, SessTrackHammDist]
    else:
        sim_output = [SessTrackSimilarity, SessTrackCorrelation, SessTrackEuclDist, SessTrackManhDist]
    
    
    return sim_output




def featureAugment(df_history, cols):
    df_fa = pd.DataFrame()
    
    # split the data based on skip_2
    df_history_T = df_history.loc[df_history['skip_2']==True]
    df_history_F = df_history.loc[df_history['skip_2']==False]
        
    for c in cols:
        df_fa[c+'_mean'] = df_history.groupby(['session_id'])[c].mean()
        df_fa[c+'_std'] = df_history.groupby(['session_id'])[c].std()
        df_fa[c+'_skew'] = df_history.groupby(['session_id'])[c].skew()
        df_fa[c+'_median'] = df_history.groupby(['session_id'])[c].median()
        
        df_fa[c+'_T_mean'] = df_history_T.groupby(['session_id'])[c].mean()
        df_fa[c+'_T_std'] = df_history_T.groupby(['session_id'])[c].std()
        df_fa[c+'_T_skew'] = df_history_T.groupby(['session_id'])[c].skew()
        df_fa[c+'_T_median'] = df_history_T.groupby(['session_id'])[c].median()
        
        df_fa[c+'_F_mean'] = df_history_F.groupby(['session_id'])[c].mean()
        df_fa[c+'_F_std'] = df_history_F.groupby(['session_id'])[c].std()
        df_fa[c+'_F_skew'] = df_history_F.groupby(['session_id'])[c].skew()
        df_fa[c+'_F_median'] = df_history_F.groupby(['session_id'])[c].median()
        
        # correlations
        df_fa[c+'_PearsonTrend'] = df_history.groupby(['session_id'])[c, 'session_position'].corr().iloc[0::2,-1].reset_index(level=1, drop=True)
        df_fa[c+'_SpearTrend'] = df_history.groupby(['session_id'])[c, 'session_position'].corr('spearman').iloc[0::2,-1].reset_index(level=1, drop=True)
        df_fa[c+'_KendallTrend'] = df_history.groupby(['session_id'])[c, 'session_position'].corr('kendall').iloc[0::2,-1].reset_index(level=1, drop=True)

        df_fa[c+'_T_PearsonTrend'] = df_history_T.groupby(['session_id'])[c, 'session_position'].corr().iloc[0::2,-1].reset_index(level=1, drop=True)
        df_fa[c+'_T_SpearTrend'] = df_history_T.groupby(['session_id'])[c, 'session_position'].corr('spearman').iloc[0::2,-1].reset_index(level=1, drop=True)
        df_fa[c+'_T_KendallTrend'] = df_history_T.groupby(['session_id'])[c, 'session_position'].corr('kendall').iloc[0::2,-1].reset_index(level=1, drop=True)

        df_fa[c+'_F_PearsonTrend'] = df_history_F.groupby(['session_id'])[c, 'session_position'].corr().iloc[0::2,-1].reset_index(level=1, drop=True)
        df_fa[c+'_F_SpearTrend'] = df_history_F.groupby(['session_id'])[c, 'session_position'].corr('spearman').iloc[0::2,-1].reset_index(level=1, drop=True)
        df_fa[c+'_F_KendallTrend'] = df_history_F.groupby(['session_id'])[c, 'session_position'].corr('kendall').iloc[0::2,-1].reset_index(level=1, drop=True)
        
        if df_history[c].dtypes.name != 'bool':
            df_fa[c+'_max'] = df_history.groupby(['session_id'])[c].max()
            df_fa[c+'_min'] = df_history.groupby(['session_id'])[c].min()
            df_fa[c+'_range'] = np.subtract(df_fa[c+'_max'], df_fa[c+'_min'], dtype=np.float32)
            df_fa[c+'_q25'] = df_history.groupby(['session_id'])[c].quantile(.25)
            df_fa[c+'_q75'] = df_history.groupby(['session_id'])[c].quantile(.75)
            df_fa[c+'_iqr'] = np.subtract(df_fa[c+'_q75'], df_fa[c+'_q25'], dtype=np.float32)
            
            df_fa[c+'_T_max'] = df_history_T.groupby(['session_id'])[c].max()
            df_fa[c+'_T_min'] = df_history_T.groupby(['session_id'])[c].min()
            df_fa[c+'_T_range'] = np.subtract(df_fa[c+'_T_max'], df_fa[c+'_T_min'], dtype=np.float32)
            df_fa[c+'_T_q25'] = df_history_T.groupby(['session_id'])[c].quantile(.25)
            df_fa[c+'_T_q75'] = df_history_T.groupby(['session_id'])[c].quantile(.75)
            df_fa[c+'_T_iqr'] = np.subtract(df_fa[c+'_T_q75'], df_fa[c+'_T_q25'], dtype=np.float32)
            
            df_fa[c+'_F_max'] = df_history_F.groupby(['session_id'])[c].max()
            df_fa[c+'_F_min'] = df_history_F.groupby(['session_id'])[c].min()
            df_fa[c+'_F_range'] = np.subtract(df_fa[c+'_F_max'], df_fa[c+'_F_min'], dtype=np.float32)
            df_fa[c+'_F_q25'] = df_history_F.groupby(['session_id'])[c].quantile(.25)
            df_fa[c+'_F_q75'] = df_history_F.groupby(['session_id'])[c].quantile(.75)
            df_fa[c+'_F_iqr'] = np.subtract(df_fa[c+'_F_q75'], df_fa[c+'_F_q25'], dtype=np.float32)
            
#             df_fa[c+'_mode'] = df_history.groupby(['session_id'])[c].agg(pd.Series.mode)

    return df_fa





# evaludation functions

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



def spotify_eval(y_true, y_pred, input_df):
    df_temp = input_df.loc[y_true.index.values,['session_id','skip_2','session_position','session_length']].copy()
    df_temp['pred'] = y_pred
    ground_truths = get_ground_truth(df_temp)
    submission = get_submission(df_temp)
    ap,first_pred_acc = evaluate(submission,ground_truths)
    return ap




def weight_perSession(df_test_pred):
    a = df_test_pred['session_position'] - np.floor(df_test_pred['session_length']/2)
    if (a<=0).sum()>0:
        raise Exception('***a number equal or below 0***')
    weight = [(1 / x) for x in a]
    return weight


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

            return mean_diff/mean_diff[0]
    
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

