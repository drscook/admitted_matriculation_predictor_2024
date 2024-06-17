from term import *
import hashlib, miceforest as mf, flaml as fl
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, log_loss
from sklearn import set_config
set_config(transform_output="pandas")
code_desc = lambda x: [x+'_code', x+'_desc']

crse_codes = [
    '_anycrse',
    # 'agec2317',
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
    term_codes: tuple = (202108, 202208, 202308, 202408)
    crse_code : str = '_anycrse'
    styp_code : str = 'n'
    stats: tuple = (pctl(0), pctl(25), pctl(50), pctl(75), pctl(100))
    show: set = dataclasses.field(default_factory=set)
    param: dict = dataclasses.field(default_factory=dict)    
    root_path: str = f"/scratch/user/u.gm197559"
    dependence: dict = dataclasses.field(default_factory=lambda: {'adm':'raw', 'flg':'raw', 'raw':'X', 'reg':'X', 'X':'X_proc', 'X_proc':'Y'})
    aggregations: tuple = (
        'crse_code',
        'coll_desc',
        'dept_desc',
        'majr_desc',
        'camp_desc',
        # # 'stat_desc',
        # # 'cnty_desc',
        'gender',
        *[f'race_{r}' for r in ['american_indian','asian','black','pacific','white','hispanic']],
        # 'waiver',
        'hs_qrtl',
        'international',
        # 'resd_desc',
        # 'lgcy',
        'lgcy_desc',
        # # 'admt_desc',
        'math',
        'reading',
        'writing',
        # # 'ssb',
        # 'oriented',
    )

    def __post_init__(self):
        super().__post_init__()
        self.root_path /= rjust(self.cycle_day,3,0)


    def get_terms(self):
        def func():
            print()
            self.terms = {key: {term_code:
                Term(term_code=term_code, cycle_day=cycle_day, overwrite=self.overwrite, show=self.show).get_reg().get_raw()
            for term_code in self.term_codes} for key, cycle_day in {'current':self.cycle_day, 'actual':0}.items()}
            del self.terms['actual'][self.proj_code]
            self.cycle_date = self.terms['current'][self.proj_code].cycle_date
        return self.get(func, fn="terms.pkl")


    def get_X(self):
        def func():
            def get_raw(dct):
                R = pd.concat([T.raw for term_code, T in dct.items()]).dropna(axis=1, how='all').reset_index(drop=True).prep().copy()
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
                return X

            idx = ['term_code','pidm']
            with warnings.catch_warnings(action='ignore'):
                X = {key: get_raw(dct).set_index(idx) for key, dct in self.terms.items()}
            Y = {key:
                pd.concat([T.reg for term_code, T in dct.items()])
                .dropna().sort_values(['crse_code',*idx]).set_index(idx).astype('string')
                # .assign(credit_hr=lambda y: y.eval("crse_code=='_allcrse'")*y['credit_hr'] + y.eval("crse_code!='_allcrse'")*y['crse_code'])
                .assign(credit_hr=lambda y: np.where(y.eval("crse_code=='_allcrse'"), y['credit_hr'], y['crse_code']))
                for key, dct in self.terms.items()}

            idx = ['variable','term_code','pidm']
            attr = ['id','levl_code','styp_code','admit','enroll','matric']
            crse = {'crse_code':'variable','credit_hr':'value'}
            def g(key):
                g = lambda variable, df: df[[]].assign(variable=variable, value=True)
                h = lambda D: D[key].drop(columns=crse, errors='ignore').melt(ignore_index=False)
                Z = pd.concat([
                    g('admit' , X['current']),
                    g('enroll', Y['current'].query("crse_code=='_anycrse'")),
                    g('matric', Y['actual' ].query("crse_code=='_anycrse'")),
                    Y[key].filter(crse).rename(columns=crse),
                    h(Y),
                    h(X),
                ]).dropna().astype('string').groupby(idx, sort=False).first()
                mask = Z.eval("variable in @attr")
                Z = Z[mask].unstack(0).droplevel(0,1).prep(bool=True).join(Z[~mask]).query("levl_code=='ug' & styp_code in ('n','r','t')").reset_index()
                Z['pidm'] = encrypt(Z['pidm'])
                Z['id'] = encrypt(Z['id'])
                Z.loc[Z.eval("variable==value"), "variable"] = "crse_code"
                return Z.set_index(idx+attr)
            self.Z = {'current':g('current').query('admit'), 'actual':g('actual').query('matric')}
            g = lambda df: df.groupby(['variable','term_code','levl_code','styp_code','value']).size()
            dct = {
                'admit' : g(self.Z['current']),
                'enroll': g(self.Z['current'].query('enroll')),
                'actual': g(self.Z['actual']),
            }
            dct['mlt'] = dct['actual'] / g(self.Z['actual'].query('admit'))
            A = pd.DataFrame(dct)
            mask = A.eval(f"term_code!={self.proj_code}")
            B = A[mask]
            C = A[~mask].drop(columns='mlt').join(B['mlt'].rename(index={k:self.proj_code for k in self.term_codes}))
            self.agg = pd.concat([B,C])
            self.y_true = pd.concat([z.loc['crse_code'].rsindex(['value',*idx]).assign(**{key:True}) for key,z in self.Z.items()], axis=1).prep(bool=True).fillna(False)
            X = self.Z['current'].query("variable!='crse_code'").unstack(0).droplevel(0,1).prep(bool=True)
            fill = {
                '_allcrse': 0,
                'birth_day': ['median',['term_code','styp_code']],
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
            for key, val in fill.items():
                X[key] = X.impute(key, *listify(val))
            M = X.isnull().rename(columns=lambda x:x+'_missing')
            self.X = X.join(M).prep(bool=True)
        return self.get(func, pre="terms", drop="terms", fn="X.pkl")


    def get_X_proc(self):
        def func():
            X = self.X.query(f"styp_code==@self.styp_code").rename(columns=lambda x:x+'_')
            trf = ColumnTransformer(self.param['trf'][2], remainder='drop', verbose_feature_names_out=False)
            X_trf = trf.fit_transform(X).prep(bool=True, cat=True)
            imp_dct = self.param['imp'][2].copy()
            iterations = imp_dct.pop('iterations')
            tune = imp_dct.pop('tune')
            if tune:
                ds = imp_dct.pop('datasets')
                imp = mf.ImputationKernel(self.X_trf, datasets=1, **imp_dct)
                imp.mice(iterations)
                variable_parameters, losses = imp.tune_parameters(dataset=0)
                imp_dct['datasets'] = ds
            else:
                variable_parameters = None
            imp = mf.ImputationKernel(X_trf, **imp_dct)
            imp.mice(iterations, variable_parameters=variable_parameters)
            self.X_proc = pd.concat([
                    imp.complete_data(k)
                    .addlevel({'sim':k, 'trf_hash':self.param['trf'][0], 'imp_hash':self.param['imp'][0]})
                    .prep(bool=True, cat=True)
                for k in range(imp.dataset_count())])
        return self.get(func, pre="X", fn=f"X_proc/{self.styp_code}/{self.param['trf'][0]}/{self.param['imp'][0]}.pkl")


    def get_y_pred(self):
        def func():
            Z = (
                self.X_proc
                .join(self.y_true.loc[self.crse_code])
                .fillna({c:False for c in self.y_true.columns})
                .sort_values(['actual','act_equiv_missing_','pidm'], ascending=False)
                .groupby(['term_code','sim']).filter(lambda x: x.eval(f"actual.sum()>=5 | term_code.max()=={self.proj_code}"))
                .assign(mask = lambda x: x.groupby(['term_code','sim']).cumcount() % 5 > 0)
            )
            self.clf = dict()
            y = dict()
            train_score = dict()
            for train_code in self.term_codes:
            # for train_code in [self.proj_code]:
                print(train_code, end="...")
                X_model = Z.query("term_code==@train_code" if train_code != self.proj_code else "term_code!=@train_code").copy()
                if len(X_model) > 0:
                    y_model = X_model.pop('actual')
                    mask = X_model.pop('mask')
                    dct = self.param['clf'][2] | {
                        'X_train':X_model[mask],
                        'y_train':y_model[mask],
                        'X_val':X_model[~mask],
                        'y_val':y_model[~mask],
                        'task':'classification',
                        'verbose':0,
                    }
                    clf = fl.AutoML(**dct)
                    with warnings.catch_warnings(action='ignore'):
                        clf.fit(**dct)
                    pred = clf.predict(Z.drop(columns=['actual','mask']))
                    proba = clf.predict_proba(Z.drop(columns=['actual','mask']))[:,1]
                    train_score[train_code] = clf.best_result['val_loss'] * 100
                    self.clf[train_code] = clf._trained_estimator
                    y[train_code] = Z[['actual']].assign(pred=pred, proba=proba, train_code=train_code, crse_code=self.crse_code, clf_hash=self.param['clf'][0]).prep(bool=True)
                    print('done', end="  ")
                else:
                    print('fail', end="  ")
            if y:
                self.y_pred = pd.concat(y.values()).rsindex(['crse_code','term_code','pidm','train_code','sim','trf_hash','imp_hash','clf_hash'])
                self.train_score = pd.Series(train_score, name='train_score').rename_axis('train_code')
            
        return self.get(func, pre="X_proc", drop=["terms","X","y_true","Z","agg","X_trf","X_proc"],
                        fn=f"y_pred/{self.styp_code}/{self.crse_code}/{self.param['trf'][0]}/{self.param['imp'][0]}/{self.param['clf'][0]}.pkl")


    def get_results(self):
        def func():
            self.Y_pred = pd.concat([read(fn).get('y_pred', pd.DataFrame()) for fn in sorted((self.root_path / 'y_pred').rglob('*.pkl'))]).prep().reset_index()
            self.results = dict()
            for variable in listify(self.aggregations):
                grp = [variable,'levl_code','styp_code','term_code','train_code','sim','trf_hash','imp_hash','clf_hash']
                S = (self.Y_pred.query("crse_code=='_anycrse'" if variable!="crse_code" else "crse_code.notnull()")
                    .merge(self.X[variable if variable!="crse_code" else []].reset_index(),'left')
                    .groupby(grp).apply(lambda y: pd.Series({
                            'predict': y['proba'].sum(),
                            'test_score': log_loss(y['actual'], y['proba'], labels=[False,True]) * 100,
                        }), include_groups=False)
                    .reset_index()
                    .merge(self.train_score.reset_index(),'left')
                    .merge(self.agg.loc[variable].reset_index().rename(columns={'value':variable}).prep(bool=True),'left')
                )
                P = S.rename(columns={'actual':'prior'})[[*grp,'prior']].copy()
                P['term_code'] += 100
                S = S.merge(P,'left')
                alpha = 1
                S['overall_score'] = (S['train_score'] + alpha * S['test_score']) / (1 + alpha)
                S['predict'] *= S['mlt']
                S['predict_error'] = S['predict'] - S['actual']
                S['predict_error_pct'] = S['predict_error'] / S['actual'] * 100
                S['predict_change'] = S['predict'] - S['prior']
                S['predict_change_pct'] = S['predict_change'] / S['prior'] * 100
                S['train_code'] = S['train_code'].astype('string').replace(str(self.proj_code), 'all')
                S = (S
                    .prep()
                    .sort_values(grp, ascending=[True,True,True,False,False,True,True,True,True])
                    .set_index(grp)
                    [['admit','enroll','predict','prior','predict_change','predict_change_pct','actual','predict_error','predict_error_pct','overall_score','test_score','train_score','mlt']]
                )
                grp.remove('sim')
                self.results[variable] = {'summary':S} | {str(stat):S.groupby(grp,sort=False).agg(stat).prep() for stat in listify(self.stats)}
        return self.get(func, pre=["y_pred","X"], fn="results.pkl")


    def get_report(self):
        self.get_results()
        with pd.ExcelWriter(self.root_path / f'AMP_{self.cycle_date.date()}.xlsx', mode='w', engine='openpyxl') as writer:
            for variable, dct in self.results.items():
                # dct['mean'].to_excel(writer, sheet_name=variable)
                dct[' 50%'].to_excel(writer, sheet_name=variable)
        return self


pwrtrf = make_pipeline(StandardScaler(), PowerTransformer())
param_grds = {
    'trf': {
        '_allcrse': 'passthrough',
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
        'term_code': 'drop',
        **{f'race_{r}': 'passthrough' for r in ['american_indian','asian','black','pacific','white','hispanic']},
        'reading': 'passthrough',
        'remote': 'passthrough',
        'resd': 'passthrough',
        'schlship_app': 'passthrough',
        'ssb': 'passthrough',
        'waiver': 'passthrough',
        'writing': 'passthrough',
    },
    'imp': {
        'random_state': seed,
        'datasets': 10,
        'iterations': 10,
        # 'datasets': 2,
        # 'iterations': 2,
        'tune': False,
        # 'tune': [False, True],
    },
    'clf': {
        'seed': seed,
        'metric': 'log_loss',
        # 'early_stop': False,
        # 'time_budget': 2,
        # 'time_budget': 120,
        'max_iter': 75,
        'hpo_method': 'bs',
        # 'estimator_list': [['xgboost']],
        'ensemble': False, 
        'custom_hp': {
                    "xgboost": {
                        'tree_method': {
                            'domain': 'hist'
                        },
                        'device': {
                            'domain': 'cuda'
                        },
                    }
                },
        # 'n_jobs': -1,
        # 'use_ray': True,
        # 'n_concurrent_trials': 8,
        # 'ensemble': [False, True],
    },
}


formatter = lambda x: str(x).replace('\n','').replace(' ','')
hasher = lambda x, d=2: hashlib.shake_128(formatter(x).encode()).hexdigest(d)
param_dct = dict()
for key, val in param_grds.items():
    lst = cartesian(val, sort=True, key=str)
    if key == 'trf':
        lst = [[(c,t,[c+'_']) for c,t in trf.items() if t not in ['drop', None, '']] for trf in lst]
    param_dct[key] = [[hasher(k), formatter(k), k] for k in lst]
param_lst = cartesian(param_dct)


def run_amp(cycle_day, *styp_codes):
    cycle_day = int(cycle_day)
    styp_codes = styp_codes if styp_codes else ['n','r','t']
    self = AMP(cycle_day=cycle_day).get_X()
    for kwargs in cartesian({'crse_code':intersection(crse_codes, self.y_true.reset_index()['value'], sort=True, reverse=True), 'cycle_day':cycle_day, 'styp_code':styp_codes, 'param':param_lst}):
        self = AMP(**kwargs).get_y_pred()
    # return self.get_report()


if __name__ == "__main__":
    delattr(pd.Series, 'disp')
    delattr(pd.DataFrame, 'disp')
    @pd_ext
    def disp(df, max_rows=4, max_cols=200, **kwargs):
        print()
        print(df.reset_index().drop(columns='index', errors='ignore').head(max_rows).to_markdown(tablefmt='psql'))

    # sys.stdout = sys.stderr  # unbuffer output - emulates -u
    print(pd.Timestamp.now())
    run_amp(98, 'n')