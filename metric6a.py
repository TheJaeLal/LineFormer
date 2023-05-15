import os
import sys
import json
import math
import itertools
# import editdistance
import numpy as np
import scipy.optimize
import scipy.spatial.distance

def check_groups(ds):
    try:
        _i = ds[0][0]
        return 1
    except Exception:
        return 0

def pprint(obj):
    print(json.dumps(obj, indent=4, sort_keys=True))

def get_dataseries(json_obj):
    if 'task6_output' in json_obj:
        return json_obj['task6_output']['visual elements']
    elif 'task6' in json_obj:
        return json_obj['task6']['output']['visual elements']
    return None

def euclid(p1, p2):
    x1 = float(p1['x'])
    y1 = float(p1['y'])
    x2 = float(p2['x'])
    y2 = float(p2['y'])
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# def box_to_discrete(ds):
#     out = []
#     for it_name in ['first_quartile', 'max', 'min', 'median', 'third_quartile']: 
#         out.append( {'name': it_name, 'x': ds[it_name]['x'], 'y': ds[it_name]['y']} )
#     return out

def box_arr_to_np(ds):
    n = np.zeros( (1, 8))
    cnt_q = 0
    for _i,p in enumerate(ds):      
        n[0,cnt_q] = float(ds[p]['y'])
        n[0,cnt_q+1] = float(ds[p]['x'])
        cnt_q = cnt_q+1
    return n

def compare_box(pred_ds, gt_ds, min_dim):
    pred_ds = box_arr_to_np(pred_ds)
    gt_ds = box_arr_to_np(gt_ds)
    cost_mat = np.minimum(1, scipy.spatial.distance.cdist(pred_ds, gt_ds, metric='cityblock') /(min_dim*0.05))
    return cost_mat

def scatt_arr_to_np(ds):
    n = np.zeros((len(ds), 2))
    for i, p in enumerate(ds):
        n[i,0] = float(p['x'])
        n[i,1] = float(p['y'])
    return n

def bar_arr_to_np(ds):
    n = np.zeros([1,4])
    n[0,0] = float(ds['y0'])
    n[0,1] = float(ds['x0'])
    n[0,2] = float(ds['height']) + float(ds['y0'])
    n[0,3] = float(ds['width']) + float(ds['x0'])
    return n

def compare_bar(pred_ds, gt_ds, min_dim):
    pred_ds = bar_arr_to_np(pred_ds)
    gt_ds = bar_arr_to_np(gt_ds)
    
    cost_mat = np.minimum(1, scipy.spatial.distance.cdist(pred_ds, gt_ds, metric='cityblock') /(min_dim*0.05))
    return cost_mat

def compare_scatter(pred_ds, gt_ds, min_dim, gamma, beta):

    is_grouped = check_groups(gt_ds)
    
    if is_grouped:
        len_seq = len(gt_ds)
    else:
        len_seq = 1
        pred_ds = [pred_ds]
        gt_ds = [gt_ds]

    score = np.zeros((len(gt_ds), len(pred_ds)))
    for iter_seq1 in range(len(gt_ds)):
        gt_seq = scatt_arr_to_np(gt_ds[iter_seq1])

        for iter_seq2 in range(len(pred_ds)):
            pred_seq = scatt_arr_to_np(pred_ds[iter_seq2])
        
            # V = np.cov(gt_ds.T)
            # VI = np.linalg.inv(V).T
            
            #cost_mat = np.minimum(1, scipy.spatial.distance.cdist(pred_ds, gt_ds, metric='mahalanobis', VI=VI) / gamma)
            cost_mat = np.minimum(1, scipy.spatial.distance.cdist(pred_seq, gt_seq, metric='euclidean') / (min_dim*gamma))
        
            score[iter_seq1, iter_seq2] = get_score(cost_mat)

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-score)
    score = score[row_ind, col_ind].sum()/(float(len(gt_ds))*beta)

    return score

def get_score(cost_mat):
    cost_mat = pad_mat(cost_mat)
    k = cost_mat.shape[0]
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_mat)
    
    cost = cost_mat[row_ind, col_ind].sum()
    score = 1 - (cost / k)
    return score

def get_cont_recall(p_xs, p_ys, g_xs, g_ys, epsilon):
    total_score = 0
    total_interval = 0

    for i in range(g_xs.shape[0]):
        x = g_xs[i]
        
        if g_xs.shape[0] == 1:
            interval = 1
        elif i == 0:
            interval = (g_xs[i+1] - x) / 2
        elif i == (g_xs.shape[0] - 1):
            interval = (x - g_xs[i-1]) / 2
        else:
            interval = (g_xs[i+1] - g_xs[i-1]) / 2

        y = g_ys[i]
        y_interp = np.interp(x, p_xs, p_ys)
        error = min(1, abs( (y - y_interp) / (abs(y) + epsilon)))
        total_score += (1 - error) * interval
        total_interval += interval

    if g_xs.shape[0] != 1:
        assert np.isclose(total_interval, g_xs[-1] - g_xs[0])
    return total_score / total_interval

def compare_continuous(pred_ds, gt_ds):
    pred_ds = sorted(pred_ds, key=lambda p: float(p['x']))
    gt_ds = sorted(gt_ds, key=lambda p: float(p['x']))

    if not pred_ds and not gt_ds:
        # empty matches empty
        return 1.0
    elif not pred_ds and gt_ds:
        # empty does not match non-empty
        return 0.0
    elif pred_ds and not gt_ds:
        # empty does not match non-empty
        return 0.0

    p_xs = np.array([float(ds['x']) for ds in pred_ds])
    p_ys = np.array([float(ds['y']) for ds in pred_ds])
    g_xs = np.array([float(ds['x']) for ds in gt_ds])
    g_ys = np.array([float(ds['y']) for ds in gt_ds])

    epsilon = (g_ys.max() - g_ys.min()) / 100.
    recall = get_cont_recall(p_xs, p_ys, g_xs, g_ys, epsilon)
    precision = get_cont_recall(g_xs, g_ys, p_xs, p_ys, epsilon)

    return (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.

# def norm_edit_dist(s1, s2):
    # return editdistance.eval(s1, s2) / float(max(len(s1), len(s2), 1))

def create_dist_mat(pred_seq, gt_seq, compare, beta):
    is_grouped = check_groups(gt_seq)

    if not is_grouped:
        len_seq = 1
        gt_seq = [gt_seq]
        pred_seq = [pred_seq]

    score = 0
    for iter_seq1 in range(len(gt_seq)):
        l1 = len(gt_seq[iter_seq1])
        tmp_score = 0
        
        for iter_seq2 in range(len(pred_seq)):
            l2 = len(pred_seq[iter_seq2])
            mat = np.full( (l1, l2), -1.)
            for i in range(l1):
                for j in range(l2):
                    mat[i,j] = compare(gt_seq[iter_seq1][i], pred_seq[iter_seq2][j])
            tmp_score = max(tmp_score, get_score(1 - (mat/beta)))
        score += tmp_score
    score = score/float(len(gt_seq))

    return score

def pad_mat(mat):
    h,w = mat.shape
    if h == w:
        return mat
    elif h > w:
        new_mat = np.zeros( (h, h) )
        new_mat[:,:w] = mat
        return new_mat
    else:
        new_mat = np.zeros( (w, w) )
        new_mat[:h,:] = mat
        return new_mat
    
def compare_line_6b(pred_ds, gt_ds):
    is_grouped = check_groups(gt_ds)
    if is_grouped:
        score = np.zeros((len(gt_ds), len(pred_ds)))
        score = pad_mat(score)
        for iter_seq1 in range(len(gt_ds)):
            for iter_seq2 in range(len(pred_ds)):
                score[iter_seq1, iter_seq2] = compare_continuous(gt_ds[iter_seq1], pred_ds[iter_seq2])
        
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(-score)
        score = score[row_ind, col_ind].sum()/score.shape[0]
    else:
        # print(gt_ds)
        score = compare_continuous(pred_ds, gt_ds)

    return score

def compare_line_6a(pred_ds, gt_ds):
    is_grouped = check_groups(gt_ds)
    if is_grouped:
        score = np.zeros((len(gt_ds), len(pred_ds)))
        for iter_seq1 in range(len(gt_ds)):
            for iter_seq2 in range(len(pred_ds)):
                score[iter_seq1, iter_seq2] = compare_continuous(gt_ds[iter_seq1], pred_ds[iter_seq2])
        
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(-score)
        score = score[row_ind, col_ind].sum()/len(gt_ds)
    else:
        # print(gt_ds)
        score = compare_continuous(pred_ds, gt_ds)

    return score

def metric_6a(pred_data_series, gt_data_series, gt_type, alpha=1, beta=2, gamma=1, img_dim = [1280.0, 960.0], debug=False):
    if 'box' in gt_type.lower():
        compare = lambda ds1, ds2: compare_box(ds1, ds2, min(img_dim))
        pred_no_names = pred_data_series['boxplots']
        gt_no_names = gt_data_series['boxplots']
        ds_match_score = create_dist_mat(pred_no_names, gt_no_names, compare, beta)
    elif 'bar' in gt_type.lower():
        compare = lambda ds1, ds2: compare_bar(ds1, ds2, min(img_dim))
        pred_no_names = pred_data_series['bars']
        gt_no_names = gt_data_series['bars']
        ds_match_score = create_dist_mat(pred_no_names, gt_no_names, compare, beta)
    elif 'scatter' in gt_type.lower():
        pred_no_names = pred_data_series['scatter points']
        gt_no_names = gt_data_series['scatter points']
        ds_match_score = compare_scatter(pred_no_names, gt_no_names, min(img_dim), gamma, beta)
    elif 'line' in gt_type.lower():
        pred_no_names = pred_data_series['lines']
        gt_no_names = gt_data_series['lines']
        ds_match_score = compare_line(pred_no_names, gt_no_names)
    else:
        raise Exception("Odd Case")
  
    return ds_match_score

def metric_6a_indv(pred_data_series, gt_data_series, gt_type, alpha=1, beta=2, gamma=1, img_dim = [1280.0, 960.0], debug=False):
    # expects both pred_data_series and gt_data_series to be the list of lists. Only for line charts
    if 'line' in gt_type.lower():
        ds_match_score = compare_line_6a(pred_data_series, gt_data_series)
    else:
        raise Exception("Odd Case")
    return ds_match_score

def metric_6b_indv(pred_data_series, gt_data_series, gt_type, alpha=1, beta=2, gamma=1, img_dim = [1280.0, 960.0], debug=False):
    # expects both pred_data_series and gt_data_series to be the list of lists. Only for line charts
    if 'line' in gt_type.lower():
        ds_match_score = compare_line_6b(pred_data_series, gt_data_series)
    else:
        raise Exception("Odd Case")
    return ds_match_score

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("USAGE: python metric6a.py pred_file|pred_dir gt_file|gt_dir [alpha] [beta] [gamma] [img_dim] [debug]")
        exit()
    pred_infile = sys.argv[1]
    gt_infile = sys.argv[2]

    try:
        alpha = float(sys.argv[3])
    except:
        alpha = 1
    try:
        beta = float(sys.argv[4])
    except:
        beta = 1
    try:
        gamma = float(sys.argv[5])
    except:
        gamma = 1    
    try:
        img_dim = sys.argv[6]
    except:
        img_dim = [1280, 960.0]
    try:
        debug = sys.argv[7]
    except:
        debug = False

    if os.path.isfile(pred_infile) and os.path.isfile(gt_infile):
        pred_json = json.load(open(pred_infile))
        gt_json = json.load(open(gt_infile))

        pred_outputs = get_dataseries(pred_json)
        gt_outputs = get_dataseries(gt_json)
        gt_type = gt_json['task1']['output']['chart_type']

        score = metric_6a(pred_outputs, gt_outputs, gt_type, alpha, beta, gamma, img_dim, debug)
        print(score)
    elif os.path.isdir(pred_infile) and os.path.isdir(gt_infile):
        scores_type = {}
        scores = []
        for x in os.listdir(pred_infile):
            print("Processing: %s" %x)

            pred_file = os.path.join(pred_infile, x)
            gt_file = os.path.join(gt_infile, x)

            try:
                pred_json = json.load(open(pred_file))
                gt_json = json.load(open(gt_file))
            except Exception:
                continue

            pred_outputs = get_dataseries(pred_json)
            gt_outputs = get_dataseries(gt_json)
            gt_type = gt_json['task1']['output']['chart_type']

            score = metric_6a(pred_outputs, gt_outputs, gt_type, alpha, beta, gamma, img_dim, debug)
            
            if (gt_type in scores_type):
                scores_type[gt_type].append(score)
            else:
                scores_type[gt_type] = [score]
            scores.append(score)
            print("Score: %f" %score)
        avg_score = sum(scores) / len(scores)
        print("Average Score: %f" % avg_score)
        for types in scores_type:
            print("Average Score for %s: %f" %(types, sum(scores_type[types])/len(scores_type[types])))
    else:
        print("Error: pred_file and gt_file must both be files or both be directories")
        exit()
