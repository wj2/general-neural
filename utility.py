
import numpy as np
import os
import sys
import re
import shutil
import scipy.io as sio
import scipy.linalg as spla
import pystan as ps
import pickle
import itertools as it
import configparser
from pref_looking.eyes import analyze_eyemove
from pref_looking.bias import get_look_img

monthdict = {'01':'Jan', '02':'Feb', '03':'Mar', '04':'Apr', '05':'May', 
             '06':'Jun', '07':'Jul', '08':'Aug', '09':'Sep', '10':'Oct',
             '11':'Nov', '12':'Dec'}

class ConfigParserColor(configparser.ConfigParser):

    def getcolor(self, *args, zero_to_one=True, **kwargs):
        string = self.get(*args, **kwargs)
        if string is not None:
            col = np.array(list(int(x.strip()) for x in string.split(',')))
            assert len(col) == 3
            if zero_to_one:
                col = col/256
        else:
            col = string
        return col
    
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_stan_summary_col(summary, col):
    col_names = summary['summary_colnames']
    neff_col = col_names.index(col)
    targ = summary['summary'][:, neff_col]
    return targ

def stan_model_valid(sm, rhat_range=(.9, 1.1), min_n_eff=5000):
    if sm is not None:
        summary = sm.summary()
        rhat = get_stan_summary_col(summary, 'Rhat')
        rhat_constraint = (np.any(rhat_range[0] > rhat)
                           and np.any(rhat_range[1] < rhat))
        neff = get_stan_summary_col(summary, 'n_eff')
        neff_constraint = np.any(neff < min_n_eff)
        valid = not (rhat_constraint or neff_constraint)
    else:
        valid = False
    return valid

def filter_invalid_models(models, depth=2, rhat_range=(.9, 1.1),
                          min_n_eff=5000):
    new_models = []
    for m in models:
        if depth == 1:
            if stan_model_valid(m, rhat_range, min_n_eff):
                new_m = m
            else:
                new_m = None
        else:
            new_m = filter_invalid_models(m, depth - 1, rhat_range,
                                          min_n_eff)
        new_models.append(new_m)
    return new_models        

def generate_all_combinations(size, starting_order):
    combs = []
    for i in range(starting_order, size + 1):
        c = list(it.combinations(range(size), i))
        combs = combs + c
    return combs

def pickle_all_stan_models(folder='general/stan_models/', pattern='.*\.stan$',
                           decoder='utf-8'):
    fls = os.listdir(folder)
    print(fls)
    fls = filter(lambda x: re.match(pattern, x) is not None, fls)
    for f in fls:
        print(f)
        path = os.path.join(folder, f)
        pickle_stan_model(path, decoder)
        
def pickle_stan_model(path, decoder='utf-8'):
    name, ext = os.path.splitext(path)
    with open(path, 'rb') as f:
        s = f.read().decode(decoder)
    newname = name + '.pkl'
    sm = ps.StanModel(model_code=s)
    with open(newname, 'wb') as f:
        pickle.dump(sm, f)
    return newname

def h2(p):
    return -p*np.log2(p) - (1 - p)*np.log2(1 - p)

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    if len(v1.shape) == 1:
        v1 = np.expand_dims(v1, 0)
    if len(v2.shape) == 1:
        v2 = np.expand_dims(v2, 0)
    v1_lens = np.sqrt(np.sum(v1**2, axis=1))
    v2_lens = np.sqrt(np.sum(v2**2, axis=1))
    s = np.sum(v1*v2, axis=1)/(v1_lens*v2_lens)
    return s 

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    TAKEN FROM: https://gist.github.com/pv/8036995

    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def make_trial_constraint_func(fields, targets, relationships, 
                               combfunc=np.logical_and, begin=True):
    def dfunc(data):
        selection = np.array([begin]*data.shape[0])
        for i, field in enumerate(fields):
            target = targets[i]
            relat = relationships[i]
            b = relat(data[field], target)
            selection = combfunc(selection, b)
        return selection
    return dfunc

def make_time_field_func(fieldname, offset=0):
    def ffunc(data):
        ts = data[fieldname] + offset
        return ts
    return ffunc

def split_angles(angs, div):
    sangs = np.sort(angs)
    ind = np.argmin(np.abs(angs - div)) + 1
    halfmark = int(np.ceil(len(sangs)/2))
    rollby = halfmark - ind
    rolled = np.roll(sangs, rollby)
    ambig = ((rolled - div) % 180) == 0
    ambig_dirs = rolled[ambig]
    c1_mask = np.logical_and(np.arange(rolled.shape[0]) < halfmark, 
                             np.logical_not(ambig))
    c1 = rolled[c1_mask]
    c2_mask = np.logical_and(np.arange(rolled.shape[0]) >= halfmark,
                             np.logical_not(ambig))
    c2 = rolled[c2_mask]
    return c1, c2, ambig_dirs

def signed_vector_angle_2d(v1, v2, degrees=False):
    theta = np.arctan2(v1[0]*v2[1] - v1[1]*v2[0],
                       v1[0]*v2[0] + v1[1]*v2[1])
    if degrees:
        theta = theta*(180/np.pi)
    return theta

def vector_angle(v1, v2, degrees=True):
    v1 = np.array(v1)
    v2 = np.array(v2)
    costheta = np.dot(v1, v2)/(np.sqrt(np.sum(v1**2))*np.sqrt(np.sum(v2**2)))
    if costheta > 1.:
        costheta = 1
    theta = np.arccos(costheta)
    if degrees:
        theta = theta*(180/np.pi)
    return theta

def make_unit_vector(v):
    v = np.array(v)
    if len(v.shape) == 1:
        v = np.expand_dims(v, 0)
    v_len = np.expand_dims(np.sqrt(np.sum(v**2, axis=1)), 1)
    v_norm = v/v_len
    return np.squeeze(v_norm)

def demean_unit_std(data, collapse_dims=(), axis=0, sig_eps=.00001):
    mask_shape = np.array(data.shape)
    mask_shape[axis] = 1
    collapse_dims = np.sort(collapse_dims)[::-1]
    coll_data = data
    for cd in collapse_dims:
        mask_shape[cd] = 1
        coll_data = collapse_array_dim(coll_data, cd, axis)
    mu = coll_data.mean(axis).reshape(mask_shape)
    sig = coll_data.std(axis).reshape(mask_shape)
    sig[sig < sig_eps] = 1.
    out = (data - mu)/sig
    return out

def collapse_array_dim(arr, col_dim, stack_dim=0):
    inds = [slice(0, arr.shape[i]) for i in range(len(arr.shape))]
    arrs = []
    for i in range(arr.shape[col_dim]):
        inds[col_dim] = i
        arrs.append(arr[tuple(inds)])
    return np.concatenate(arrs, axis=stack_dim)

def get_matching_files(datadir, expr):
    dirfiles = os.listdir(datadir)
    matchfiles = filter(lambda x: re.match(expr, x) is not None, dirfiles)
    return matchfiles

def load_collection_bhvmats(datadir, params, expr='.*\.mat',
                            forget_imglog=False, repl_logpath=None,
                            log_suffix='_imglog.txt', make_log_name=True,
                            trial_cutoff=300, max_trials=None, dates=None,
                            use_data_name=False):
    dirfiles = os.listdir(datadir)
    matchfiles = filter(lambda x: re.match(expr, x) is not None, dirfiles)
    ld = {}
    full_bhv = None
    for i, mf in enumerate(matchfiles):
        name, ext = os.path.splitext(mf)
        if make_log_name:
            imglog = os.path.join(datadir, name + log_suffix)
        else:
            imglog = None
        full_mf = os.path.join(datadir, mf)
        if forget_imglog:
            ld = {}
        try:
            bhv, ld = load_bhvmat_imglog(full_mf, imglog, datanum=i, 
                                         prevlog_dict=ld, dates=dates,
                                         repl_logpath=repl_logpath,
                                         use_data_name=use_data_name,
                                         **params)
            if max_trials is not None:
                bhv = bhv[:max_trials]
            if len(bhv) > trial_cutoff:
                if full_bhv is None:
                    full_bhv = bhv
                else:
                    full_bhv = np.concatenate((full_bhv, bhv), axis=0)
            else:
                print('file {} has less than {} trials and was '
                      'excluded'.format(mf, trial_cutoff))
        except Exception as ex:
            print('error on {}'.format(full_mf))
            print(ex)
    return full_bhv

code_to_direc_saccdmc = {201:0, 219:270, 213:180, 207:90}

def load_saccdmc(datapaths, pattern='rcl[mj][0-9]*.mat', varname='data'):
    rawd = load_separate(datapaths, pattern, varname=varname)
    combed = []
    for i, ud in enumerate(rawd):
        try:
            c = merge_neuro_bhv_saccdmc(ud['NEURO'][0], ud['BHV'][0],
                                          datanum=i)
            combed.append(c)
        except ValueError:
            msg = 'error formatting file {} (ind {})'
            print(msg.format(ud['BHV'][0]['DataFileName'][0,0], i))
    d = np.concatenate(combed)
    return d 

def merge_neuro_bhv_saccdmc(neuro, bhv, noerr=True, fixation_on=35, 
                            fixation_off=36, start_trial=9, datanum=0, 
                            lever_release=4, lever_hold=7, reward_given=48,
                            end_trial=18, fixation_acq=8, sample_on_code=23,
                            sample_off_code=24, test1_on_code=25, 
                            test1_off_code=26, test2_on_code=27, 
                            test2_off_code=28, spks_buff=1000, saccdms_block=3,
                            ms_block=1, cat1sampdir=(90, 180), 
                            cat2sampdir=(270, 0), cat1testdir=(75, 135, 195), 
                            cat2testdir=(255, 315, 15), datanumber=None,
                            code_to_direc=code_to_direc_saccdmc):
    samp_catdict = {}
    samp_catdict.update([(d, 1) for d in cat1sampdir])
    samp_catdict.update([(d, 2) for d in cat2sampdir])
    test_catdict = {}
    test_catdict.update([(d, 1) for d in cat1testdir])
    test_catdict.update([(d, 2) for d in cat2testdir])
    trls = bhv['TrialNumber'][0,0]
    ntrs = len(trls)
    fields = ['trial_type', 'TrialError', 'block_number', 'code_numbers',
              'code_times', 'trial_starts', 'datafile', 'datanum',
              'sample_on','sample_off','test1_on', 'test1_off', 'fixation_move',
              'trialnum','fixation_on','fixation_off',
              'lever_release','lever_hold','reward_time','ISI_start',
              'ISI_end','fixation_acquired','eyepos','spike_times','LFP',
              'task_cond_num', 'angle', 'saccade', 'saccade_intoRF',
              'saccade_towardRF', 'sample_direction', 'test_direction', 'match',
              'sample_category', 'test_category']
    dt = {'names':fields, 'formats':['O']*len(fields)}
    neuro_starts, neurons = split_spikes(neuro['Neuron'][0,0], 
                                         neuro['CodeNumbers'][0,0],
                                         neuro['CodeTimes'][0,0], 
                                         start_code=start_trial,
                                         end_code=end_trial, buff=spks_buff)
    x = np.zeros(ntrs, dtype=dt)
    condinf = bhv['InfoByCond'][0,0]
    for i, t in enumerate(trls):
        cn = int(bhv['ConditionNumber'][0,0][i, 0])
        x[i]['trial_type'] = cn
        x[i]['task_cond_num'] = cn
        x[i]['TrialError'] = bhv['TrialError'][0,0][i, 0]
        x[i]['block_number'] = bhv['BlockNumber'][0,0][i, 0]
        x[i]['code_numbers'] = bhv['CodeNumbers'][0,0][0, i]
        x[i]['code_times'] = bhv['CodeTimes'][0,0][0, i]
        strt = get_bhvcode_time(start_trial, x[i]['code_numbers'],
                                x[i]['code_times'], 
                                first=False)
        x[i]['trial_starts'] = 0
        x[i]['ISI_end'] = get_bhvcode_time(start_trial, x[i]['code_numbers'],
                                           x[i]['code_times'], 
                                           first=True) - strt
        x[i]['ISI_start'] = get_bhvcode_time(end_trial, x[i]['code_numbers'],
                                             x[i]['code_times'], 
                                             first=True) - strt
        x[i]['datafile'] = bhv['DataFileName']
        x[i]['datanum'] = datanum
        x[i]['trialnum'] = bhv['TrialNumber'][0,0][i, 0]
        x[i]['sample_on'] = get_bhvcode_time(sample_on_code, 
                                             x[i]['code_numbers'], 
                                             x[i]['code_times'], 
                                             first=True) - strt
        x[i]['sample_off'] = get_bhvcode_time(sample_off_code, 
                                              x[i]['code_numbers'],
                                              x[i]['code_times'], 
                                              first=True) - strt
        x[i]['test1_on'] = get_bhvcode_time(test1_on_code, 
                                           x[i]['code_numbers'],
                                           x[i]['code_times'], 
                                           first=True) - strt
        x[i]['test1_off'] = get_bhvcode_time(test1_off_code, 
                                            x[i]['code_numbers'],
                                            x[i]['code_times'], 
                                            first=True) - strt
        if cn == 9: # some trials are condition number nine (flash map), 
                    # but saccdms block
            pass
        elif x[i]['block_number'] == saccdms_block:
            x[i]['saccade'] = condinf[cn-1, 0]['Saccade']
            x[i]['saccade_intoRF'] = condinf[cn-1, 0]['IntoRF']
            x[i]['saccade_towardRF'] = condinf[cn-1, 0]['TowardsRF']
            try:
                x[i]['sample_direction'] = condinf[cn-1, 0]['SampleDirection']
                x[i]['test_direction'] = condinf[cn-1, 0]['TestDirection']
                x[i]['test_category'] = test_catdict[int(x[i]['test_direction'])]
                x[i]['match'] = condinf[cn-1, 0][0,0]['Match']
            except ValueError:
                direc = code_to_direc[int(condinf[cn-1, 0][0,0]['SampleCode'])]
                x[i]['sample_direction'] = direc
            x[i]['sample_category'] = samp_catdict[int(x[i]['sample_direction'])]
        elif x[i]['block_number'] == ms_block:
            x[i]['angle'] = condinf[cn-1,0]['Angle']
        ep = bhv['AnalogData'][0, 0][0, i]['EyeSignal'][0, 0]
        x[i]['eyepos'] = ep[int(strt):,:]
        x[i]['spike_times'] = dict((k, neurons[k][i]) for k in neurons)
        if x[i]['saccade']:
            x[i]['fixation_move'] = get_bhvcode_time(fixation_off, 
                                                     x[i]['code_numbers'], 
                                                     x[i]['code_times'], 
                                                     first=True) - strt
        else:
            x[i]['fixation_move'] = np.nan
    if noerr:
        x = x[x['TrialError'] == 0]
    return x

def split_spikes_startdur(neuron, starts, durs, buff=1000):
    neurs = neuron.dtype.fields.keys()
    ns = dict((n, []) for n in neurs)
    for i, s in enumerate(starts):
        end = s + durs[i]
        for n in neurs:
            n_spks = neuron[n][0, 0]
            t_spks = np.logical_and(n_spks > s - buff,
                                    n_spks < end + buff)
            spks_t = n_spks[t_spks] - s
            ns[n].append(spks_t)
    return ns

def split_spikes(neuron, codes, code_times, start_code=9, end_code=18, 
                 buff=1000):
    neurs = neuron.dtype.fields.keys()
    ns = dict((n, []) for n in neurs)
    start_times = []
    starts = np.where(codes == start_code)[0]
    for i, s in enumerate(starts):
        if len(starts) > i + 1 and s + 1 == starts[i+1]:
            pass
        else:
            end = np.where(codes[s:] == end_code)[0][0]
            end_time = code_times[s:][end]
            start_time = code_times[s]
            for n in neurs:
                n_spks = neuron[n][0,0]
                t_spks = np.logical_and(n_spks > start_time - buff, 
                                        n_spks < end_time + buff)
                spks_t = n_spks[t_spks] - start_time
                ns[n].append(spks_t)
            start_times.append(start_time)
    return start_times, ns

def merge_timecourses(ns, xs):
    """
    concatenates two separate timecourses that have the same number of trials
    and neurons (eg, can be used to subtract some mid-block that has variable 
    length)
    
    ns - list<list<dict>>, all dicts should have the same keys; keys can be 
    arbitrary, but the value should always be an array of shape KxT, where K 
    (trials) for a particular key is constant across the all dicts in ns

    xs - list<float>, should be the same length as ns, with length T
    """
    merge_ns = [{} for _ in range(len(ns[0]))]
    merge_xs = np.concatenate(xs)
    splits = [len(x) for x in xs]
    for i in range(len(ns)):
        for j in range(len(ns[i])):
            for k in ns[i][j].keys():
                if k in merge_ns[j].keys():
                    v = merge_ns[j][k]
                    merge_ns[j][k] = np.concatenate((v, ns[i][j][k]), axis=1)
                else:
                    merge_ns[j][k] = ns[i][j][k]
    return merge_ns, merge_xs, splits

def merge_trials(ns):
    merge_ns = {}
    for i in range(len(ns)):
        for k in ns[i].keys():
            if k in merge_ns.keys():
                v = merge_ns[k]
                merge_ns[k] = np.concatenate((v, ns[i][k]), axis=0)
            else:
                merge_ns[k] = ns[i][k]
    return merge_ns

def generate_orthonormal_vectors(v, n):
    v = np.array(v)
    if len(v.shape) == 1:
        v = np.expand_dims(v, 0)
    vecs = spla.null_space(v)
    weightings = np.random.randn(vecs.shape[1], n)
    out = np.dot(vecs, weightings).T
    out = make_unit_vector(out)
    return out

def generate_orthonormal_basis(d):
    mat = np.random.randn(d, d)
    basis = spla.orth(mat)
    return basis

def get_orthogonal_basis(q):
    q = np.array(q)
    q = q/np.sqrt(np.sum(q**2))
    v_star = np.ones_like(q)
    v_star[0] = np.sqrt((1 - q[0])/2)
    v_star[1:] = -q[1:]/(2*v_star[0])
    p = np.identity(len(q)) - 2*np.outer(v_star, v_star)
    return p    

def load_data(path, varname='superx'):
    data = sio.loadmat(path, mat_dtype=True)
    return data[varname]

def load_separate(paths, pattern=None, varname='x'):
    files = []
    pattern = '.*' + pattern
    for p in paths:
        if os.path.isdir(p):
            f = os.listdir(p)[1:]
            f = [os.path.join(p, x) for x in f]
            files = files + f
        else:
            files.append(p)
    if pattern is not None:
        files = filter(lambda x: re.match(pattern, x) is not None, files)
    for i, f in enumerate(files):
        d = load_data(f, varname=varname)
        if i == 0:
            alldata = d
        else:
            alldata = np.concatenate((alldata, d), axis=0)
    return alldata    

def load_bhvmat_imglog(path_bhv, path_log=None, noerr=True, 
                       prevlog_dict=None,
                       trial_field='TrialNumber', fixation_on=35, 
                       fixation_off=36, start_trial=9, datanum=0, 
                       lever_release=4, lever_hold=7, reward_given=48,
                       end_trial=18, fixation_acq=8, left_img_on=191,
                       left_img_off=192, right_img_on=195, right_img_off=196,
                       default_wid=5, default_hei=5, default_img1_xy=(-5, 0),
                       default_img2_xy=(5, 0), spks_buff=1000,
                       plt_conds=(7,8,9,10,11,12,13,14,15,16), 
                       sdms_conds=(1,2,3,4,5,6,7,8), ephys=False,
                       centimgon=25, centimgoff=26, eyedata_len=500,
                       eye_params={}, dates=None, xy1_loc_dict=None,
                       xy2_loc_dict=None, repl_logpath=None,
                       use_data_name=False, lum_conds=None,
                       correct_sdms_position=True):
    if prevlog_dict is None:
        log_dict = {}
    else:
        log_dict = prevlog_dict
    data = sio.loadmat(path_bhv)
    if ephys:
        data = data['data'][0]
        bhv = data['BHV'][0]
    else:
        data = data['bhv']
        bhv = data
    if path_log is None and 'imglog' in data.dtype.names:
        path_log = data['imglog'][0][0]
        path_log = path_log.replace('uc/freedman/', '')
    elif path_log is None and use_data_name:
        filename = bhv['DataFileName'][0][0][0]
        fn, ext = os.path.splitext(filename)
        logname = fn + '_imglog.txt'
        folder, end = os.path.split(path_bhv)
        path_log = os.path.join(folder, logname)
    log = None
    if 'imglog_data' in data.dtype.names:
        raw_log = data['imglog_data'][0][0]
        log = list(x[0].encode() for x in raw_log)
    if ephys:
        neuro = data['NEURO'][0]
        neurons = split_spikes_startdur(neuro['Neuron'][0,0], 
                                        neuro['TrialTimes'][0,0],
                                        neuro['TrialDurations'][0,0], 
                                        buff=spks_buff)
    if path_log is not None and log is None:
        if repl_logpath is not None:
            p, name = os.path.split(path_log)
            path_log = os.path.join(repl_logpath, name)
        if os.path.exists(path_log):
            log = open(path_log, 'rb').readlines()
    if path_log is not None and not os.path.exists(path_log) and log is None:
        print('imglog path does not exist')
        print(path_log)
        print('and no log data provided; not attempting to load')
        path_log = None
        cns = bhv['ConditionNumber'][0,0][:, 0]
        n_pltrials = sum(c in plt_conds for c in cns)
        print('there are {} pref looking trials'.format(n_pltrials))
    if dates is not None:
        date_pattern = '[0-9]{2}[-_]?[0-9]{2}[_-]?[0-9]{4}'
        m = re.search(date_pattern, path_bhv)
        if m is not None:
            included_date = m[0] in dates
        else:
            print('no date found in file name')
            included_date = False
    else:
        included_date = None
    trls = bhv[trial_field][0,0]
    fields = ['trial_type', 'TrialError', 'block_number', 'code_numbers',
              'code_times', 'trial_starts', 'datafile', 'datanum',
              'samp_img_on','samp_img_off','trialnum','image_nos','leftimg',
              'rightimg', 'leftviews', 'rightviews','fixation_on','fixation_off',
              'lever_release','lever_hold','reward_time','ISI_start',
              'ISI_end','fixation_acquired','left_img_on','left_img_off',
              'right_img_on','right_img_off','eyepos','spike_times','LFP',
              'task_cond_num', 'img1_xy', 'img2_xy', 'img_wid', 'img_hei',
              'test_array_on', 'test_array_off', 'left_first', 'right_first',
              'first_sacc_time', 'angular_separation', 'included_date',
              'saccade_begs', 'saccade_ends', 'saccade_lens', 'saccade_targ',
              'leftimg_type', 'rightimg_type', 'first_look', 'on_left_img',
              'on_right_img', 'not_on_img']
    dt = {'names':fields, 'formats':['O']*len(fields)}
    x = np.zeros(len(trls), dtype=dt)
    for i, t in enumerate(trls):
        x[i]['included_date'] = included_date
        x[i]['code_numbers'] = bhv['CodeNumbers'][0,0][0, i]
        x[i]['code_times'] = bhv['CodeTimes'][0,0][0, i]
        x[i]['trial_starts'] = get_bhvcode_time(start_trial, 
                                                x[i]['code_numbers'],
                                                x[i]['code_times'], first=True)
        x[i]['ISI_end'] = get_bhvcode_time(start_trial, x[i]['code_numbers'],
                                           x[i]['code_times'], first=True)
        if ephys:
            x[i]['spike_times'] = dict((k, neurons[k][i] + x[i]['trial_starts']) 
                                       for k in neurons)
        else:
            x[i]['spike_times'] = None
        x[i]['trial_type'] = bhv['ConditionNumber'][0,0][i, 0]
        x[i]['task_cond_num'] = bhv['ConditionNumber'][0,0][i, 0]
        x[i]['TrialError'] = bhv['TrialError'][0,0][i, 0]
        x[i]['block_number'] = bhv['BlockNumber'][0,0][i, 0]
        x[i]['ISI_start'] = get_bhvcode_time(end_trial, x[i]['code_numbers'],
                                             x[i]['code_times'], first=True)
        # x[i]['datafile'] = bhv['DataFileName'][0][0][0]
        x[i]['datanum'] = datanum
        x[i]['samp_img_on'] = get_bhvcode_time(centimgon,
                                               x[i]['code_numbers'],
                                               x[i]['code_times'], first=True)
        x[i]['samp_img_off'] = get_bhvcode_time(centimgoff,
                                                x[i]['code_numbers'],
                                                x[i]['code_times'], first=True)
        x[i]['test_array_on'] = get_bhvcode_time(centimgon,
                                                 x[i]['code_numbers'],
                                                 x[i]['code_times'], 
                                                 first=False)
        x[i]['test_array_off'] = get_bhvcode_time(centimgoff,
                                                  x[i]['code_numbers'],
                                                  x[i]['code_times'], 
                                                  first=False)
        x[i]['trialnum'] = bhv['TrialNumber'][0,0][i, 0]
        x[i]['image_nos'] = []
        if (x[i]['trial_type'] in plt_conds 
            or x[i]['trial_type'] in sdms_conds) and log is not None:
            entry1 = log.pop(0)
            entry1 = entry1.strip(b'\r\n').split(b'\t')
            tn1, s1, _, cond1, vs1, cat1, img1 = entry1
            entry2 = log.pop(0)
            entry2 = entry2.strip(b'\r\n').split(b'\t')
            tn2, s2, _, cond2, vs2, cat2, img2 = entry2
            if int(tn1) != int(x[i]['trialnum']):
                print(entry1)
                print(entry2)
                print(x[i]['trialnum'])
                print(x[0]['trialnum'])
                # print(filename)
                print(tn1, x[i]['trialnum'])
            assert tn1 == tn2
            assert int(tn1) == int(x[i]['trialnum'])
            img1_n, ext1 = os.path.splitext(img1)
            img2_n, ext2 = os.path.splitext(img2)
            log_dict[img1_n] = log_dict.get(img1_n, 0) + 1
            log_dict[img2_n] = log_dict.get(img2_n, 0) + 1
            x[i]['leftimg'] = img1_n
            x[i]['rightimg'] = img2_n
            x[i]['leftviews'] = log_dict[img1_n]
            x[i]['rightviews'] = log_dict[img2_n]
            x[i]['leftimg_type'] = cat1
            x[i]['rightimg_type'] = cat2
        else:
            x[i]['leftimg'] = ''
            x[i]['rightimg'] = ''
            x[i]['leftviews'] = 0
            x[i]['rightviews'] = 0
        x[i]['fixation_on'] = get_bhvcode_time(fixation_on, 
                                               x[i]['code_numbers'],
                                               x[i]['code_times'], first=True)
        x[i]['fixation_off'] = get_bhvcode_time(fixation_off, 
                                                x[i]['code_numbers'],
                                                x[i]['code_times'], first=True)
        x[i]['lever_release'] = get_bhvcode_time(lever_release,
                                                 x[i]['code_numbers'],
                                                 x[i]['code_times'], first=True)
        x[i]['lever_hold'] = get_bhvcode_time(lever_hold,
                                              x[i]['code_numbers'],
                                              x[i]['code_times'], first=True)
        x[i]['reward_time'] = get_bhvcode_time(reward_given,
                                               x[i]['code_numbers'],
                                               x[i]['code_times'], first=True)
        x[i]['fixation_acquired'] = get_bhvcode_time(fixation_acq,
                                                     x[i]['code_numbers'],
                                                     x[i]['code_times'], 
                                                     first=True)
        x[i]['left_img_on'] = get_bhvcode_time(left_img_on, 
                                               x[i]['code_numbers'],
                                               x[i]['code_times'], first=True)
        x[i]['left_img_off'] = get_bhvcode_time(left_img_off,
                                                x[i]['code_numbers'],
                                                x[i]['code_times'], first=True)
        x[i]['right_img_on'] = get_bhvcode_time(right_img_on, 
                                                x[i]['code_numbers'],
                                                x[i]['code_times'], first=True)
        x[i]['right_img_off'] = get_bhvcode_time(right_img_off,
                                                 x[i]['code_numbers'],
                                                 x[i]['code_times'], first=True)
        ep = bhv['AnalogData'][0,0][0, i]['EyeSignal'][0,0]
        x[i]['eyepos'] = ep
        if ('UserVars' in bhv.dtype.names 
            and 'img1_xy' in bhv['UserVars'][0,0].dtype.names):
            tmp1 = bhv['UserVars'][0,0]['img1_xy']
            tmp2 = bhv['UserVars'][0,0]['img2_xy']
            if tmp1.shape == (1, 1) and tmp1[0,0].shape[1] > i:
                if tmp1[0, 0][0, i].shape[1] > 0:
                    x[i]['img1_xy'] = tmp1[0,0][0, i][0]
                    x[i]['img2_xy'] = tmp2[0,0][0, i][0]
                else:
                    x[i]['img1_xy'] = default_img1_xy
                    x[i]['img2_xy'] = default_img2_xy
            elif tmp1.shape[1] > i and tmp1[0, i].shape[1] > 0:
                x[i]['img1_xy'] = tmp1[0, i]
                x[i]['img2_xy'] = tmp2[0, i]
            else:
                x[i]['img1_xy'] = default_img1_xy
                x[i]['img2_xy'] = default_img2_xy
        elif (xy1_loc_dict is not None
              and not np.any(np.isnan(xy1_loc_dict[datanum]))):
            x[i]['img1_xy'] = xy1_loc_dict[datanum]
            x[i]['img2_xy'] = xy2_loc_dict[datanum]
        else:
            x[i]['img1_xy'] = default_img1_xy
            x[i]['img2_xy'] = default_img2_xy
        if ('UserVars' in bhv.dtype.names 
            and 'img_wid' in bhv['UserVars'][0,0].dtype.names
            and bhv['UserVars'][0,0]['img_wid'].shape[1] > i):
            x[i]['img_wid'] = bhv['UserVars'][0,0]['img_wid'][0, i]
            x[i]['img_hei'] = bhv['UserVars'][0,0]['img_hei'][0, i]
        else:
            x[i]['img_wid'] = default_wid
            x[i]['img_hei'] = default_hei
        _add_eye_info(x[i], eye_params, eyedata_len)

    if correct_sdms_position:
        t_mask = np.isin(x['trial_type'], plt_conds)
        ang_mask = x['angular_separation'] == 180
        full_mask = t_mask*ang_mask
        filt = x[full_mask]
        xy1 = filt[0]['img1_xy']
        xy2 = filt[0]['img2_xy']
        sdms_mask = np.isin(x['trial_type'], sdms_conds)
        for i in np.where(sdms_mask)[0]:
            x[i]['img1_xy'] = xy1
            x[i]['img2_xy'] = xy2
            _add_eye_info(x[i], eye_params, eyedata_len)
    if noerr:
        x = x[x['TrialError'] == 0]
    return x, log_dict

def _add_eye_info(x_i, eye_params, eyedata_len=500):
    ep = x_i['eyepos']
    ang = compute_angular_separation(x_i['img1_xy'],
                                     x_i['img2_xy'])
    x_i['angular_separation'] = np.round(ang)
    if ep.shape[0] > eyedata_len:
        sbs, ses, l, look = analyze_eyemove(ep, x_i['img1_xy'], 
                                            x_i['img2_xy'],
                                            wid=x_i['img_wid'],
                                            hei=x_i['img_hei'],
                                            postthr=x_i['fixation_off'],
                                            readdpost=False, **eye_params)
        x_i['saccade_begs'] = sbs
        x_i['saccade_ends'] = ses
        x_i['saccade_lens'] = l
        x_i['saccade_targ'] = look
        if len(look) > 0:
            x_i['left_first'] = look[0] == b'l'
            x_i['right_first'] = look[0] == b'r'
            x_i['first_sacc_time'] = sbs[0] + x_i['fixation_off']
            x_i['first_look'] = look[0]
        if np.isnan(x_i['fixation_off']):
            x_i['on_left_img'] = np.nan
            x_i['on_right_img'] = np.nan
            x_i['not_on_img'] = np.nan
        else:
            tlen = int(x_i['eyepos'].shape[0] - x_i['fixation_off'])
            all_eyes = get_look_img((x_i,), 2, 1, 0, x_i['img1_xy'],
                                    x_i['img2_xy'], x_i['img_wid'],
                                    x_i['img_hei'], fix_time=0, tlen=tlen,
                                    eye_field='eyepos',
                                    eyemove_flag='fixation_off')
            x_i['on_left_img'] = all_eyes[0, :, 2]
            x_i['on_right_img'] = all_eyes[0, :, 1]
            x_i['not_on_img'] = all_eyes[0, :, 0]

def copy_struct_array(new_arr, old_arr):
    names = old_arr.dtype.names
    for n in names:
        new_arr[n] = old_arr[n]
    return new_arr

def get_bhvcode_time(codenum, trial_codenums, trial_codetimes, first=True):
    i = np.where(codenum == trial_codenums)[0]
    if len(i) > 0:
        if first:
            i = i[0]
        else:
            i = i[-1]
        ret = int(np.round(trial_codetimes[i][0]))
    else:
        ret = np.nan
    return ret

def get_only_vplt(data, condrange=(7, 20), condfield='trial_type'):
    mask = np.logical_and(data[condfield] >= condrange[0], 
                         data[condfield] <= condrange[1])
    return data[mask]

def make_ratio_function(func1, func2):
    def _ratio_func(ts):
        one = func1(ts)
        two = func2(ts)
        norm = one / (one + two)
        return norm
    return _ratio_func

def compute_angular_separation(xy1, xy2):
    theta1 = np.rad2deg(np.arctan2(xy1[1], xy1[0]))
    theta2 = np.rad2deg(np.arctan2(xy2[1], xy2[0]))
    diff = np.abs(theta1 - theta2) % 360
    sep = np.min([diff, 360 - diff])
    return sep    

def get_only_conds(data, condlist, condfield='trial_type'):
    mask = np.zeros(data.shape[0])
    if max(data[condfield].shape) == 1:
        for_masking = data[condfield][0,0]
    else:
        for_masking = data[condfield]
    for c in condlist:
        c_mask = for_masking == c
        mask = np.logical_or(mask, c_mask)
    return data[mask]

def nan_array(size, dtype=None):
    arr = np.ones(size, dtype=dtype)
    arr[:] = np.nan
    return arr

def euclidean_distance(pt1, pt2):
    pt1 = np.array(pt1)
    if len(pt1.shape) == 1:
        pt1 = pt1.reshape((1, pt1.shape[0]))
    pt2 = np.array(pt2)
    if len(pt2.shape) == 1:
        pt2 = pt2.reshape((1, pt2.shape[0]))
    return np.sqrt(np.sum((pt1 - pt2)**2, axis=1))

def distribute_imglogs(il_path, out_path):
    il_list = os.listdir(il_path)
    nomatch = []
    for il in il_list:
        m = re.findall('-([0-9][0-9])(?=-)', il)
        if len(m) == 2:
            mo = monthdict[m[0]]
            da = m[1]
            if da[0] == '0':
                da = da[1]
            foldname = mo+da
            fpath = os.path.join(out_path, foldname)
            if foldname not in os.listdir(out_path):
                os.mkdir(fpath)
            shutil.copy(os.path.join(il_path, il),
                        os.path.join(out_path, foldname))
        else:
            nomatch.append(il)
    return nomatch

def iterate_function(func, args, kv_argdict):
    """
    Facilitates doing parameter sweeps for functions, recording results
    as different arguments to the function are varied.

    Parameters
    ----------
    func : function
        The function to be swept, func(*args) must be a valid function call.
    args : list
        The default values for each function argument; note that when a 
        particular parameter is varied, the other arguments will have the
        value given here.
    kv_argdict : dict
        A dictionary with keys (k, ind) where k is the text name of a parameter
        (used for output) and ind is the index of that parameter in args -- the
        values are the different values of k to sweep.

    Returns
    -------
    res_dict : dict
        A dictionary with keys k from kv_argdict, the values are the lists of
        results from the parameter sweep for that argument.
    """
    res_dict = {}
    for kvs in kv_argdict.keys():
        print(kvs)
        k = kvs[0]
        inds = kvs[1:]
        vals = kv_argdict[kvs]
        res_dict[k] = (vals, [])
        for j in range(len(vals[0])):
            new_args = args[:]
            new_args = list(new_args)
            for i, ind in enumerate(inds):
                new_args[ind] = vals[i][j]
            res = func(*new_args)
            res_dict[k][1].append(res)
    return res_dict

def get_data_run_nums(data, drunfield):
    return np.unique(np.concatenate(data[drunfield], axis=0))

def index_func(a, b, axis=0):
    m_a = np.nanmean(a, axis=axis)
    m_b = np.nanmean(b, axis=axis)
    ind = (m_a - m_b)/(m_a + m_b)
    mask = (m_a == 0)*(m_b == 0)
    ind[mask] = 0
    return ind

def bootstrap_test(a, b, func=np.nanmean, n=1000):
    a_m, b_m = func(a), func(b)
    a_l, b_l = len(a), len(b)
    a_s, b_s = np.var(a)/a_l, np.var(b)/b_l
    t = (a_m - b_m)/np.sqrt((a_s/a_l) + (b_s/b_l))
    z = func(np.concatenate((a,b)))
    a_i = a - a_m + z
    b_i = b - b_m + z
    a_stars, a_sems = bootstrap_list(a_i, func, n=n, ret_sem=True)
    b_stars, b_sems = bootstrap_list(b_i, func, n=n, ret_sem=True)
    t_stars = (a_stars - b_stars)/np.sqrt((a_sems/a_l) + (b_sems/b_l))
    p = np.sum(t_stars >= t)/n
    return p

def bootstrap_diff(a, b, func, n=1000):
    stats = np.zeros(n)
    for i in range(n):
        a_inds = np.random.choice(np.arange(len(a)), len(a))
        a_choice = a[a_inds]
        b_inds = np.random.choice(np.arange(len(b)), len(b))
        b_choice = b[b_inds]
        stats[i] = func(a_choice) - func(b_choice)
    return stats

def bootstrap_list(l, func, n=1000, out_shape=None, ret_sem=False):
    if out_shape is None:
        stats = np.zeros(n)
    else:
        stats = np.zeros((n,)+out_shape)
    if ret_sem:
        sems = np.zeros_like(stats)
    for i in range(n):
        inds = np.random.choice(np.arange(len(l)), len(l))
        samp = l[inds]
        stats[i] = func(samp)
        if ret_sem:
            sems[i] = np.var(samp)/len(l)
    if ret_sem:
        out = (stats, sems)
    else:
        out = stats
    return out

def resample_on_axis(d, n, axis=0, with_replace=True):
    if d.shape[axis] >= n  and n > 0:
        inds = np.random.choice(d.shape[axis], n, replace=with_replace)
        ref = [slice(0, d.shape[i]) for i in range(len(d.shape))]
        ref[axis] = inds
        samp = d[tuple(ref)]
    else:
        samp_shp = list(d.shape)
        samp_shp[axis] = 1
        samp = np.zeros(samp_shp)
        samp[:] = np.nan
    return samp
    
def mean_axis0(x):
    return np.mean(x, axis=0)

def bootstrap_on_axis(d, func, axis=0, n=500, with_replace=True):
    stats = np.zeros((n,) + d.shape[1:])
    for i in range(n):
        samp = resample_on_axis(d, n, axis=axis)
        stats[i] = func(samp)
    return stats

def collapse_list_dict(ld):
    for i, k in enumerate(ld.keys()):
        if i == 0:
            l = ld[k]
        else:
            l = np.concatenate((l, ld[k]))
    return l

def gen_img_list(famfolder=None, fam_n=None, novfolder=None, nov_n=None, 
                 intfamfolder=None, intfam_n=None):
    if famfolder is not None:
        fams = os.listdir(famfolder)[1:]
    else:
        fams = ['F {}'.format(i+1) for i in np.arange(fam_n)]
    if novfolder is not None:
        novs = os.listdir(novfolder)[1:]
    else:
        novs = ['N {}'.format(i+1) for i in np.arange(nov_n)]
    if intfamfolder is not None:
        intfams = os.listdir(intfamfolder)[1:]
    else:
        intfams = ['IF {}'.format(i+1) for i in np.arange(intfam_n)]
    return fams, novs, intfams
        

def get_img_names(codes, famfolder='/Users/wjj/Dropbox/research/uc/freedman/'
                  'pref_looking/famimgs', if_ns=25, n_ns=50):
    f, n, i = gen_img_list(famfolder=famfolder, nov_n=n_ns, intfam_n=if_ns)
    all_imnames = np.array(f + n + i)
    cs = (codes - 1).astype(np.int)
    return all_imnames[cs]

def get_cent_codes(tcodes, imgcodebeg=56, imgcodeend=180):
    return tcodes[(tcodes >= imgcodebeg)*(tcodes <= imgcodeend)]

def get_code_member(code, fambeg=56, famend=105, intbeg=106, intend=130, 
                    novbeg=131, novend=180):
    if novbeg <= code <= novend:
        member = np.array([[1, 0, 0]])
    elif intbeg <= code <= intend:
        member = np.array([[0, 1, 0]])
    elif fambeg <= code <= famend:
        member = np.array([[0, 0, 1]])
    return member

def get_code_views(code, fambeg=56, famend=105, intbeg=106, intend=130, 
                   novbeg=131, novend=180, novviews=25., intviews=450., 
                   famviews=10000.):
    memb = get_code_member(code, novbeg=novbeg, novend=novend, intbeg=intbeg, 
                           intend=intend, fambeg=fambeg, famend=famend)
    return np.sum(np.array([novviews, intviews, famviews])*memb)

def views_to_member(views, novbeg=0, novend=200, intbeg=201, intend=1000, 
                    fambeg=1001, famend=100000):
    return get_code_member(views, novbeg=novbeg, novend=novend, intbeg=intbeg,
                           intend=intend, fambeg=fambeg, famend=famend)
    
def bins_tr(spks, beg, end, binsize, column=False, usetype=np.float):
    bs = np.arange(beg, end + binsize, binsize)
    spk_bs, _ = np.histogram(spks, bins=bs)
    if column:
        spk_bs = np.reshape(spk_bs, (-1, 1))
    return spk_bs.astype(usetype)

def get_neuron_spks_list(data, zerotimecode=8, drunfield='datanum', 
                         spktimes='spike_times'):
    undruns = get_data_run_nums(data, drunfield)
    neurons = []
    for i, dr in enumerate(undruns):
        d = data[data[drunfield] == dr]
        drneurs = []
        for j, tr in enumerate(d):
            t = get_code_time(tr, zerotimecode)
            for k, spks in enumerate(tr[spktimes][0, :]):
                if len(spks) > 0:
                    tspks = spks - t
                    if j == 0:
                        drneurs.append([tspks])
                    else:
                        drneurs[k].append(tspks)
                else:
                    if j == 0:
                        drneurs.append([])
        neurons = neurons + drneurs
    return neurons

def euler_integrate(func, beg, end, step):
    accumulate = 0
    for t in np.arange(beg, end, step):
        accumulate = accumulate + step*func(t)
    return accumulate

def evoked_st_cumdist(spkts, t, lam):
    out = 1 - (1 - empirical_fs_cumdist(spkts, t))*np.exp(lam*t)
    return out

def empirical_fs_cumdist(spkts, t):
    spk_before = lambda x: np.any(x < t) or (x.size == 0)
    successes = np.sum(map(spk_before, spkts))
    x = map(spk_before, spkts)
    return successes / float(len(spkts))

def get_spks_window(dat, begwin, endwin):
    makecut = lambda x: (np.sum(np.logical_and(begwin <= x, x <= endwin)) 
                         / (endwin - begwin))
    stuff = map(makecut, dat)
    return stuff

def get_code_time(trl, code, codenumfield='code_numbers', 
                  codetimefield='code_times'):
    ct = trl[codetimefield][trl[codenumfield] == code]
    return ct

def estimate_latency(neurspks, backgrd_window, latenwindow, integstep=.5):
    bckgrd_spks = get_spks_window(neurspks, backgrd_window[0], 
                                  backgrd_window[1])
    bgd_est = np.mean(bckgrd_spks)
    expect_func = lambda x: 1 - evoked_st_cumdist(neurspks, x, bgd_est)
    est_lat = euler_integrate(expect_func, latenwindow[0], latenwindow[1], 
                              integstep)
    sm_func = lambda x: 2*x*(1 - evoked_st_cumdist(neurspks, x, bgd_est))
    sm_latency = euler_integrate(sm_func, latenwindow[0], latenwindow[1], 
                                 integstep)
    est_std = np.sqrt(sm_latency - est_lat**2)
    return est_lat, est_std

def get_trls_with_neurnum(dat, neurnum, neurfield='spike_times', 
                          drunfield='datanum'):
    druns = get_data_run_nums(dat, drunfield)
    count_neurs = 0
    for i, dr in enumerate(druns):
        drdat = dat[dat[drunfield] == dr]
        new_neurs = drdat[0][neurfield].shape[1]
        if count_neurs <= neurnum and new_neurs + count_neurs > neurnum:
            neur_i = neurnum - count_neurs
            for trls in drdat:
                trls[neurfield] = trls[neurfield][:, [neur_i]]
            keeptrls = drdat
            return keeptrls
        else:
            count_neurs = count_neurs + new_neurs
    raise Exception('only {} neurons in this dataset, which is less '
                    'than {}'.format(count_neurs, neurnum))
            
def get_condnum_trls(dat, condnums, condnumfield='trial_type'):
    keeps = np.zeros((dat.shape[0], len(condnums)))
    for i, c in enumerate(condnums):
        keeps[:, i] = dat[condnumfield] == c
    flatkeep = np.sum(keeps, 1)
    return dat[flatkeep > 0]
