from term import *
import hashlib, miceforest as mf, flaml as fl
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, log_loss#, fbeta_score, f1_score
# from sklearn.model_selection import train_test_split
from sklearn import set_config
set_config(transform_output="pandas")
warnings.filterwarnings("ignore", message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`")
seed = 42
code_desc = lambda x: [x+'_code', x+'_desc']

crse_codes = [
    '_allcrse',
    'agec2317',
    # 'ansc1119',
    # 'ansc1319',
    # 'anth2302',
    # 'anth2351',
    # 'arts1301',
    # 'arts1303',
    # 'arts1304',
    # 'arts3331',
    # 'biol1305',
    # 'biol1406',
    # 'biol1407',
    # 'biol2401',
    # 'biol2402',
    # 'busi1301',
    # 'busi1307',
    # 'chem1111',
    # 'chem1112',
    # 'chem1302',
    # 'chem1311',
    # 'chem1312',
    # 'chem1407',
    # 'chem1411',
    # 'chem1412',
    # 'comm1311',
    # 'comm1315',
    # 'comm2302',
    # 'crij1301',
    # 'dram1310',
    # 'dram2361',
    # 'dram4304',
    # 'easc2310',
    # 'econ1301',
    # 'econ2301',
    # 'engl1301',
    # 'engl1302',
    # 'engl2307',
    # 'engl2320',
    # 'engl2321',
    # 'engl2326',
    # 'engl2340',
    # 'engl2350',
    # 'engl2360',
    # 'engl2362',
    # 'engl2364',
    # 'engl2366',
    # 'engl2368',
    # 'engr2303',
    # 'envs1302',
    # 'fina1360',
    # 'geog1303',
    # 'geog1320',
    # 'geog1451',
    # 'geog2301',
    # 'geol1403',
    # 'geol1404',
    # 'geol1407',
    # 'geol1408',
    # 'govt2305',
    # 'govt2306',
    # 'hist1301',
    # 'hist1302',
    # 'hist2321',
    # 'hist2322',
    # 'huma1315',
    # 'kine2315',
    # 'math1314',
    # 'math1316',
    # 'math1324',
    # 'math1332',
    # 'math1342',
    # 'math2412',
    # 'math2413',
    # 'musi1303',
    # 'musi1310',
    # 'musi1311',
    # 'musi2350',
    # 'musi3325',
    # 'phil1301',
    # 'phil1304',
    # 'phil2303',
    # 'phil3301',
    # 'phys1302',
    # 'phys1401',
    # 'phys1402',
    # 'phys1403',
    # 'phys1410',
    # 'phys1411',
    # 'phys2425',
    # 'phys2426',
    # 'psyc2301',
    # 'soci1301',
    # 'soci1306',
    # 'soci2303',
    # 'univ0200',
    # 'univ0204',
    # 'univ0301',
    # 'univ0314',
    # 'univ0324',
    # 'univ0332',
    # 'univ0342',
]

@dataclasses.dataclass
class AMP(MyBaseClass):
    cycle_day : int = 0
    proj_code : int = 202408
    pred_codes: tuple = (202108, 202208, 202308, 202408)
    crse_code : str = '_allcrse'
    styp_code : str = 'n'
    # stats: tuple = [pctl(0), pctl(25), pctl(50), pctl(75), pctl(100)]
    stats: tuple = ('mean',)
    show: set = dataclasses.field(default_factory=set)
    param: dict = dataclasses.field(default_factory=dict)    
    root_path: str = f"/home/scook/institutional_data_analytics/admitted_matriculation_projection/resources/rslt"
    dependence: dict = dataclasses.field(default_factory=lambda: {'adm':'raw', 'flg':'raw', 'raw':'X', 'reg':'X', 'X':'X_proc', 'X_proc':'Y'})
    aggregations: tuple = (
        'coll_desc',
        'dept_desc',
        'majr_desc',
        'camp_desc',
        'stat_desc',
        'cnty_desc',
        'gender',
        *[f'race_{r}' for r in ['american_indian','asian','black','pacific','white','hispanic']],
        'waiver',
        'hs_qrtl',
        'international',
        'resd_desc',
        'lgcy',
        'lgcy_desc',
        'admt_desc',
        'math',
        'reading',
        'writing',
        'ssb',
        'oriented',
    )

    def __post_init__(self):
        super().__post_init__()
        self.root_path /= rjust(self.cycle_day,3,0)


    def get_terms(self):
        def func():
            print()
            self.terms = [{pred_code:
                Term(term_code=pred_code, cycle_day=cycle_day, overwrite=self.overwrite, show=self.show).get_reg().get_raw()
            for pred_code in self.pred_codes} for cycle_day in [0, self.cycle_day]]
            self.cycle_date = self.terms[1][self.proj_code].cycle_date
        return self.get(func, "terms.pkl")


    def get_X(self):
        def func():
            ren = {'term_code':'pred_code', 'term_desc':'pred_desc', 'crse_code':'value'}
            idx = ['id','pidm','pred_code','levl_code','styp_code','variable','value']
            def get_raw(dct):
                R = pd.concat([T.raw.rename(columns=ren) for pred_code, T in dct.items()]).copy().dropna(axis=1, how='all').reset_index(drop=True).prep()
                R['pidm'] = encrypt(R['pidm'])
                R['id'] = encrypt(R['id'])
                repl = {'ae':0, 'n1':1, 'n2':2, 'n3':3, 'n4':4, 'r1':1, 'r2':2, 'r3':3, 'r4':4}
                R['hs_qrtl'] = pd.cut(R['hs_pctl'], bins=[-1,25,50,75,90,101], labels=[4,3,2,1,0], right=False).combine_first(R['apdc_code'].map(repl))
                R['remote'] = R['camp_code'] != 's'
                R['resd'] = R['resd_code'] == 'r'
                R['oriented'] = R['oriented'] != 'n'
                R['lgcy'] = ~R['lgcy_code'].isin(['o',pd.NA])
                R['majr_code'] = R['majr_code'].replace({'0000':pd.NA, 'und':pd.NA, 'eled':'eted', 'agri':'unda'})
                R['coll_code'] = R['coll_code'].replace({'ae':'an', 'eh':'ed', 'hs':'hl', 'st':'sm', '00':pd.NA})
                R['coll_desc'] = R['coll_code'].map({
                    'an': 'ag & natural_resources',
                    'ba': 'business',
                    'ed': 'education',
                    'en': 'engineering',
                    'hl': 'health sciences',
                    'la': 'liberal & fine arts',
                    'sm': 'science & mathematics',
                    pd.NA: 'no college designated',
                })
                majr = ['majr_desc','dept_code','dept_desc','coll_code','coll_desc']
                S = R.sort_values('cycle_date').drop_duplicates(subset='majr_code', keep='last')[['majr_code',*majr]]
                X = R.drop(columns=majr).merge(S, on='majr_code', how='left').prep(bool=True)
                checks = [
                    'cycle_day >= 0',
                    'apdc_day >= cycle_day',
                    'appl_day >= apdc_day',
                    'birth_day >= appl_day',
                    'birth_day >= 5000',
                    'distance >= 0',
                    'hs_pctl >=0',
                    'hs_pctl <= 100',
                    'hs_qrtl >= 0',
                    'hs_qrtl <= 4',
                    'act_equiv >= 1',
                    'act_equiv <= 36',
                    'gap_score >= 0',
                    'gap_score <= 100',
                ]
                for check in checks:
                    mask = X.eval(check)
                    assert mask.all(), [check,X[~mask].disp(5)]

                fill = {
                    'birth_day': ['median',['pred_code','styp_code']],
                    'remote': False,
                    'international': False,
                    **{f'race_{r}': False for r in ['american_indian','asian','black','pacific','white','hispanic']},
                    'lgcy': False,
                    'resd': False,
                    'waiver': False,
                    'fafsa_app': False,
                    'schlship_app': False,
                    'finaid_accepted': False,
                    'ssb': False,
                    'math': False,
                    'reading': False,
                    'writing': False,
                    'gap_score': 0,
                    'oriented': False,
                }
                for k, v in fill.items():
                    X[k] = X.impute(k, *listify(v))
                M = X.isnull().rename(columns=lambda x:x+'_missing')
                return X.join(M).sample(frac=1, random_state=self.param['imp'][2]['random_state']).rename(columns=ren).prep(bool=True).sindex(idx)

            with warnings.catch_warnings(action='ignore'):
                X = [get_raw(dct) for dct in self.terms]
            self.aggregations = sorted(c for c in X[0].select_dtypes(['string','boolean']).columns if "_missing" not in c and "_code" not in c)

            Y = [pd.concat([
                    T.reg
                    .rename(columns=ren)
                    .assign(pidm=lambda x: encrypt(x['pidm']), variable='crse_code')
                    .fillna({'credit_hr':0})
                    .query('credit_hr>0')
                    .prep(bool=True)
                    .sindex(idx)[['credit_hr']]
                    # .sindex(idx)[['variable','credit_hr']]
                for pred_code, T in dct.items()]) for dct in self.terms]
            E = [y.query("value=='_allcrse'").droplevel(['variable','value']).squeeze() for y in Y]  # only rows for _allcrse
            E[0] = E[0].rename('final').to_frame()
            E[1] = E[1].rename('current').to_frame()

            col = ['levl_code','styp_code']
            X[0] = X[0].droplevel(col).join(E[0], how='outer')  # E[0] determines final levl_code, styp_code, & credit_hr
            E[0] = E[0].droplevel(col).join(X[0][[]], how='outer')  # copy id from X[0] to E[0]
            E[1] = E[1].droplevel(col).join(X[1][[]], how='outer')  # X[1] determines current levl_code & styp_code
            X[1] = X[1].droplevel(col).join(E[1], how='outer')  # copy current credit_hr from E[1] to X[1]
            X[1] = X[1].join(E[0].droplevel(col), how='outer')  # copy final credit_hr from E[0] to X[1]
            # X[0] = X[0].join(E[1].droplevel(col), how='outer')  # copy current credit_hr from E[1] to X[0]
            Y = [y[[]].droplevel(col).join(e[[]]) for y,e in zip(Y,E)]  # copy levl_code & stype_code from E to Y
            qry = f"levl_code == 'ug' & styp_code in ('n','r','t')"
            aggy = lambda y: y.query(qry).groupby(['pred_code','levl_code','styp_code','variable','value']).size()
            aggx = lambda x: aggy(x[self.aggregations].melt(ignore_index=False))
            def get_df(dct):
                Y = pd.concat(dct, axis=1).prep().fillna(0)
                Y['mlt'] = Y['final'] / Y['admitted']
                Y[np.isinf(Y)] = pd.NA
                return Y.rename(columns=lambda x: "pred_code_"+x)

            AY = get_df({
                'current': aggy(Y[1]),
                'admitted': aggy(X[1][[]].join(Y[0].droplevel(col))),
                'final': aggy(Y[0]),
            })
            AX = get_df({
                'current': aggx(X[1].query("current.notnull()")),
                'admitted': aggx(X[1].query("final.notnull()")),
                'final': aggx(X[0].query("final.notnull()")),
            })
            self.mlt = pd.concat([AY,AX])
            self.X = X[1].assign(credit_hr=X[1]['current'].fillna(0)).drop(columns=['current','final']).query(qry)
            self.y = [y.rsindex(['pidm','pred_code','value']) for y in Y]
        return self.get(func, "X.pkl", pre="terms", drop=["terms"])


    def get_X_proc(self):
        def func():
            X = self.X.query(f"styp_code=='{self.styp_code}'").rename(columns=lambda x:'__'+x)
            trf = ColumnTransformer(self.param['trf'][2], remainder='drop', verbose_feature_names_out=False)
            X_trf = trf.fit_transform(X).prep(bool=True, cat=True)
            imp_dct = self.param['imp'][2].copy()
            iterations = imp_dct.pop('iterations')
            tune = imp_dct.pop('tune')
            if tune:
                ds = imp_dct.pop('datasets')
                imp = mf.ImputationKernel(X_trf, datasets=1, **imp_dct)
                imp.mice(iterations)
                variable_parameters, losses = imp.tune_parameters(dataset=0)
                imp_dct['datasets'] = ds
            else:
                variable_parameters = None
            imp = mf.ImputationKernel(X_trf, **imp_dct)
            imp.mice(iterations, variable_parameters=variable_parameters)
            self.X_proc = pd.concat([
                    imp.complete_data(k)
                    .addlevel({'trf_hash':self.param['trf'][0], 'imp_hash':self.param['imp'][0], 'sim':k})
                    .prep(bool=True, cat=True)
                for k in range(imp.dataset_count())])
        return self.get(func, f"X_proc/{self.styp_code}/{self.param['trf'][0]}/{self.param['imp'][0]}.pkl", pre="X", drop=["terms","X"])


    def get_Y(self):
        def func():
            dct = self.param['clf'][2].copy()
            g = lambda y, nm: y.query(f"value=='{self.crse_code}'").droplevel('value').assign(**{nm:True})
            Z = (
                self.X_proc
                .join(g(self.y[1],'registered'))
                .join(g(self.y[0],'actual'))
                .prep(bool=True, cat=True)
                .fillna({'registered':False,'actual':False})
            )
            self.clf = dict()
            self.Y = dict()
            self.train_score = dict()
            for train_code in self.pred_codes:
            # for train_code in [self.proj_code]:
                print(f"...{train_code}", end="")
                strat = ['actual','__act_equiv_missing','pidm']
                grp = ['pred_code','sim']
                qry = f"pred_code=={train_code}" if train_code != self.proj_code else f"pred_code!={train_code}"
                X_model = Z.query(qry).sort_values(strat, ascending=False).copy()
                X_model['msk'] = X_model.groupby(grp).cumcount() % 5 > 0
                X_model = X_model.groupby(grp).filter(lambda x: x['actual'].sum() >= 5)
                if len(X_model) == 0:
                    # print(train_code, 'not enough')
                    pred = False
                    proba = 0.0
                    self.train_score[train_code] = pd.NA
                else:
                    msk = X_model.pop('msk')
                    y_model = X_model.pop('actual')
                    dct |= {
                        'X_train':X_model[msk],
                        'y_train':y_model[msk],
                        'X_val':X_model[~msk],
                        'y_val':y_model[~msk],
                        'task':'classification',
                        'verbose':0,
                    }
                    clf = fl.AutoML(**dct)
                    with warnings.catch_warnings(action='ignore'):
                        clf.fit(**dct)
                    pred = clf.predict(Z.drop(columns='actual'))
                    proba = clf.predict_proba(Z.drop(columns='actual'))[:,1]
                    self.clf[train_code] = clf._trained_estimator
                    self.train_score[train_code] = clf.best_result['val_loss'] * 100
                self.Y[train_code] = Z[['actual']].assign(pred=pred, proba=proba).addlevel({'crse_code':self.crse_code, 'train_code':train_code, 'clf_hash':self.param['clf'][0]}).prep(bool=True).copy()
            self.Y = pd.concat(self.Y.values())
            self.train_score = pd.Series(self.train_score, name='train_score').rename_axis('train_code')
        return self.get(func,
                        f"Y/{self.styp_code}/{self.crse_code}/{self.param['trf'][0]}/{self.param['imp'][0]}/{self.param['clf'][0]}.pkl",
                        pre=["X","X_proc"],
                        drop=["terms","X","y","mlt","X_proc"])


    def get_result(self, variable='crse_code'):
        nm = f"rslt_{variable}"
        def func():
            if 'Y' not in self:
                return
            proj_current = f'{self.proj_code}_current'
            proj_predict = f'{self.proj_code}_predict'
            prior_current = f'{self.proj_code-100}_current'
            prior_final = f'{self.proj_code-100}_final'
            proj_change = f'{self.proj_code}_change_pct'
            grp = uniquify([variable,'levl_code','styp_code','train_code','pred_code','trf_hash','imp_hash','clf_hash','sim'])
            S = (
                self.X
                .join(self.Y, how='inner')
                .groupby(grp).apply(lambda y: pd.Series({
                        'pred_code_predict': y['proba'].sum(),
                        'test_score': log_loss(y['actual'], y['proba'], labels=[False,True]) * 100,
                    }), include_groups=False)
                .join(self.train_score)
                .join(self.mlt.query(f"variable=='{variable}'").droplevel('variable').rename_axis(index={'value':variable}))
            )
            alpha = 1
            S['overall_score'] = (S['train_score'] + alpha * S['test_score']) / (1 + alpha)
            qry = lambda q: S.query(f"pred_code=={q}").droplevel('pred_code')
            S = (S
                .join(qry(self.proj_code-100)['pred_code_current'].rename(prior_current))
                .join(qry(self.proj_code-100)['pred_code_final'  ].rename(prior_final))
                .join(qry(self.proj_code    )['pred_code_current'].rename(proj_current))
                .join(qry(self.proj_code    )['pred_code_predict'].rename(proj_predict))
                .query(f"pred_code!={self.proj_code}")
                .astype('Float64').fillna(0)
            )
            S['pred_code_predict'] *= S['pred_code_final'] > 0
            S.loc[S.eval('pred_code_predict==0'), ['pred_code_predict',proj_predict,'train_score','test_score','overall_score']] = pd.NA
            for k in ['pred_code_predict',proj_predict]:
                S[k] *= S['pred_code_mlt']
            S[proj_change] = (S[proj_predict] / S[prior_final] - 1) * 100
            S['pred_code_error'] = S['pred_code_predict'] - S['pred_code_final']
            S['pred_code_error_pct'] = S['pred_code_error'] / S['pred_code_final'] * 100
            S = (
                S[[prior_current,proj_current,prior_final,proj_predict,proj_change,'pred_code_final','pred_code_predict','pred_code_error','pred_code_error_pct','pred_code_mlt','train_score','test_score','overall_score']]
                .reset_index()
                .sort_values([variable,'levl_code','styp_code','train_code','pred_code','trf_hash','imp_hash','clf_hash'], ascending=[True,True,True,False,False,True,True,True])
                .prep()
            )
            S['train_code'] = S['train_code'].astype('string').replace(str(self.proj_code), 'all')
            grp.remove('sim')
            with warnings.catch_warnings(action='ignore'):
                self[nm] = {'summary': S} | {str(stat): S.drop(columns='sim').groupby(grp, sort=False).agg(stat).prep() for stat in listify(self.stats)}
            self[nm]['mean'].disp(40)
        return self.get(func, f"{nm}/{self.styp_code}/{self.crse_code}/{self.param['trf'][0]}/{self.param['imp'][0]}/{self.param['clf'][0]}.pkl",
                        pre=["Y","X"], drop=["terms","X","y","mlt","X_proc","clf","Y"])


pwrtrf = make_pipeline(StandardScaler(), PowerTransformer())
param_grds = {
    'trf': {
        'act_equiv': 'passthrough',
        'act_equiv_missing': 'passthrough',
        'admt_code': 'drop',
        'apdc_day': 'passthrough',
        'appl_day': 'drop',
        'birth_day': 'passthrough',
        'camp_code': 'drop',
        'coll_code': 'passthrough',
        # 'distance': ['passthrough', pwrtrf],
        'distance': 'passthrough',
        'fafsa_app': 'drop',
        'finaid_accepted': 'drop',
        'gap_score': 'passthrough',
        'gender': 'passthrough',
        'hs_qrtl': 'passthrough',
        'international': 'passthrough',
        'lgcy': 'passthrough',
        'math': 'passthrough',
        'oriented': 'passthrough',
        'pred_code': 'drop',
        **{f'race_{r}': 'passthrough' for r in ['american_indian','asian','black','pacific','white','hispanic']},
        'reading': 'passthrough',
        'remote': 'passthrough',
        'resd': 'passthrough',
        'schlship_app': 'passthrough',
        'ssb': 'passthrough',
        'styp_code': 'drop',
        'waiver': 'passthrough',
        'writing': 'passthrough',
        'credit_hr': 'passthrough',
    },
    'imp': {
        'random_state': seed,
        'datasets': 10,
        'iterations': 10,
        'tune': False,
        # 'tune': [False, True],
    },
    'clf': {
        'seed': seed,
        'metric': 'log_loss',
        'early_stop': True,
        'time_budget': 2,
        # 'time_budget': 120,
        # 'time_budget': np.arange(10,500,10),
        # 'time_budget': [*np.arange(1,20),*np.arange(20,100,10),*np.arange(100,200,25),*np.arange(200,401,50)],
        # 'time_budget': 120,
        # 'estimator_list': [['lgbm','histgb']],#'xgboost']],#'catboost']],#'histgb','extra_tree','xgb_limitdepth','rf']],#'lrl1','lrl2','kneighbor'
        # 'estimator_list': [['xgboost']],#'xgboost']],#'catboost']],#'histgb','extra_tree','xgb_limitdepth','rf']],#'lrl1','lrl2','kneighbor'
        'estimator_list': [['xgboost']],#'catboost']],#'histgb','extra_tree','xgb_limitdepth','rf']],#'lrl1','lrl2','kneighbor'
        # 'estimator_list': [['xgboost']],#'histgb','extra_tree','xgb_limitdepth','rf']],#'lrl1','lrl2','kneighbor'
        # 'ensemble': [False, True],
        'ensemble': False,
        # 'ensemble': True,
    },
}


formatter = lambda x: str(x).replace('\n','').replace(' ','')
hasher = lambda x, d=2: hashlib.shake_128(formatter(x).encode()).hexdigest(d)
param_dcts = dict()
for key, val in param_grds.items():
    lst = cartesian(val, sort=True, key=str)
    if key == 'trf':
        lst = [[(c,t,["__"+c]) for c,t in trf.items() if t not in ['drop', None, '']] for trf in lst]
    param_dcts[key] = [[hasher(k), formatter(k), k] for k in lst]


def run_amp(cycle_day, styp_codes=['n'], overwrite=[]):
    for kwargs in cartesian({'cycle_day': cycle_day, 'styp_code': styp_codes, 'crse_code': sorted(crse_codes, reverse=False), 'param': cartesian(param_dcts), 'overwrite': [listify(overwrite)]}):
        self = AMP(**kwargs)
        self.get_result()
        self.get_Y()
        if self.proj_code in self.clf:
            print(self.param['clf'][0], 'time_budget', self.param['clf'][2]['time_budget'], self.clf[self.proj_code].estimator)
        if self.crse_code == '_allcrse':
            for variable in self.aggregations:
                self.get_result(variable)
    def func():
        write_csv = lambda nm: self[nm].to_csv(self.root_path / f'AMP_{nm}_{self.cycle_date.date()}.csv')
        self.stack = dict()
        append = lambda k, crse_code, df: self.stack.setdefault(k, dict()).setdefault(crse_code, df.copy())
        for fn in sorted((self.root_path / 'rslt_crse_code').rglob('*.pkl')):
            self.load(fn, force=True)
            self.load(str(fn).replace('rslt_crse_code','Y'), force=True)
            # print(self.crse_code)
            for k in ['Y']:
                append(k, self.crse_code, self[k])
            for k, v in self.rslt_crse_code.items():
                append(k, self.crse_code, v)
        self.summary = pd.concat(self.stack['mean'].values()).droplevel(['trf_hash','imp_hash','clf_hash']).round(2).prep()
        write_csv('summary')
        Y = self.stack['Y']['_allcrse'].groupby(['pidm','pred_code']).agg(actual=('actual','mean'), proba_mean=('proba','mean'), proba_stdev=('proba','std'))
        self.details = self.X.join(Y).prep(bool=True)
        write_csv('details')
    return self.get(func, f"stack.pkl", pre="X", drop=["terms","X","y","mlt","X_proc","clf","Y","rslt_crse_code"])


if __name__ == "__main__":
    print(pd.Timestamp.now())
    delattr(pd.Series, 'disp')
    delattr(pd.DataFrame, 'disp')
    @pd_ext
    def disp(df, max_rows=4, max_cols=200, **kwargs):
        print()
        print(df.reset_index().drop(columns='index', errors='ignore').head(max_rows).to_markdown(tablefmt='psql'))

    from IPython.utils.io import Tee
    with contextlib.closing(Tee('/home/scook/institutional_data_analytics/admitted_matriculation_projection/admitted_matriculation_predictor/log.txt', "w", channel="stdout")) as outputstream:
        run_amp(119)
        