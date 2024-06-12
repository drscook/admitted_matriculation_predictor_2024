from term import *
import hashlib, miceforest as mf, flaml as fl
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, log_loss
from sklearn import set_config
set_config(transform_output="pandas")
warnings.filterwarnings("ignore", message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`")
seed = 42
code_desc = lambda x: [x+'_code', x+'_desc']

crse_codes = [
    '_allcrse',
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
    # stats: tuple = [pctl(0), pctl(25), pctl(50), pctl(75), pctl(100)]
    stats: tuple = ('mean',)
    show: set = dataclasses.field(default_factory=set)
    param: dict = dataclasses.field(default_factory=dict)    
    root_path: str = f"C:\\Users\\gavin\\Documents\\admitted_matriculation_predictor"
    dependence: dict = dataclasses.field(default_factory=lambda: {'adm':'raw', 'flg':'raw', 'raw':'X', 'reg':'X', 'X':'X_proc', 'X_proc':'Y'})
    aggregations: tuple = (
        'crse_code',
        'coll_desc',
        'dept_desc',
        'majr_desc',
        'camp_desc',
        # 'stat_desc',
        # 'cnty_desc',
        'gender',
        *[f'race_{r}' for r in ['american_indian','asian','black','pacific','white','hispanic']],
        # 'waiver',
        'hs_qrtl',
        'international',
        'resd_desc',
        'lgcy',
        # 'lgcy_desc',
        # 'admt_desc',
        'math',
        'reading',
        'writing',
        # 'ssb',
        'oriented',
    )

    def __post_init__(self):
        super().__post_init__()
        self.root_path /= rjust(self.cycle_day,3,0)
        self.overwrite.add('stack')
        self.aggregations = listify(self.aggregations)


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
                .sort_values(['crse_code',*idx])
                .set_index(idx)
                .astype('string')
                .assign(credit_hr = lambda y: y.eval("crse_code=='_allcrse'")*y['credit_hr'] + y.eval("crse_code!='_allcrse'")*y['crse_code'])
                # .assign(credit_hr = lambda y: (y['crse_code']=='_allcrse')*y['credit_hr'] + (y['crse_code']!='_allcrse')*y['crse_code'])
                # .assign(temp      = 'crse_code',
                #         credit_hr = lambda x: (x['crse_code']=='_allcrse')*x['credit_hr'] + (x['crse_code']!='_allcrse')*x['crse_code'],
                #         crse_code = lambda x: (x['crse_code']=='_allcrse')*x['crse_code'] + (x['crse_code']!='_allcrse')*x['temp'],)
                # .drop(columns='temp')
                .dropna()
                for key, dct in self.terms.items()}

            idx = ['variable','term_code','pidm']
            attr = ['id','levl_code','styp_code','admit','enroll','matric']
            crse = {'crse_code':'variable','credit_hr':'value'}
            def g(key):
                qry = "crse_code=='_anycrse'"
                h = lambda D: D[key].drop(columns=crse, errors='ignore').melt(ignore_index=False)
                Z = pd.concat([
                    X['current']           [[]].assign(variable='admit' , value=True),
                    Y['current'].query(qry)[[]].assign(variable='enroll', value=True),
                    Y['actual' ].query(qry)[[]].assign(variable='matric', value=True),
                    Y[key].filter(crse).rename(columns=crse),
                    h(Y),
                    h(X),
                ]).astype('string').dropna().groupby(idx, sort=False).first()
                # mask = Z.eval("variable==value")
                # Z.loc[mask,'variable'] = 'crse_code'
                # Z[mask].disp(10)
                # assert 1==2
                mask = Z.eval("variable in @attr")
                Z = Z[mask].unstack(0).droplevel(0,1).prep(bool=True).join(Z[~mask]).reset_index().query("levl_code=='ug' & styp_code in ('n','r','t')")
                Z['pidm'] = encrypt(Z['pidm'])
                Z['id'] = encrypt(Z['id'])
                Z.loc[Z.eval("variable==value"), "variable"] = "crse_code"
                return Z.set_index(idx+attr)

                #     .assign(pidm=lambda z: encrypt(z['pidm']), id=lambda z: encrypt(z['id']))
                #     .set_index(idx+attr)
                # )

                # return (
                #     Z[mask].unstack(0).droplevel(0,1).prep(bool=True)
                #     .join(Z[~mask])
                #     .query("levl_code=='ug' & styp_code in ('n','r','t')")
                #     .reset_index()
                #     .assign(pidm=lambda z: encrypt(z['pidm']), id=lambda z: encrypt(z['id']))
                #     .set_index(idx+attr)
                # )
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

            courses = sorted(pd.concat(Y.values())['crse_code'].unique())
            courses.remove('_allcrse')
            L = [z.query(f"value in {courses}").droplevel(attr).rename(columns={'value':key}) for key,z in self.Z.items()]
            self.y_true = L[0].join(L[1], how='outer').prep(bool=True)            
            
            X = self.Z['current'].query(f"value not in {courses}").unstack(0).droplevel(0,1).prep(bool=True)
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


            # Y = {key:
            #     pd.concat([T.reg for term_code, T in dct.items()])
            #     .assign(credit_hr=lambda y: (y['crse_code']=='_allcrse')*(y['credit_hr']) + (y['crse_code']!='_allcrse')*(y['credit_hr']>0))
            #     .sort_values(['crse_code',*idx])
            #     .set_index(idx)
            #     for key, dct in self.terms.items()}

            # courses = sorted(pd.concat(Y.values())['crse_code'].unique())
            # courses.remove('_allcrse')
            # # attr = ['id','levl_code','styp_code','admit','enroll','matric','variable']
            # attr = ['id','levl_code','styp_code','admit','enroll','matric']
            # crse = {'crse_code':'variable','credit_hr':'value'}
            # def g(key):
            #     qry = "crse_code=='_anycrse'"
            #     h = lambda D: D[key].drop(columns=crse, errors='ignore').melt(ignore_index=False)
            #     Z = pd.concat([
            #         X['current']           [[]].assign(variable='admit' , value=True),
            #         Y['current'].query(qry)[[]].assign(variable='enroll', value=True),
            #         Y['final'  ].query(qry)[[]].assign(variable='matric', value=True),
            #         Y[key][crse.keys()].rename(columns=crse),
            #         h(Y),
            #         h(X),
            #     ]).dropna().astype('string').groupby([*idx,'variable'], sort=False).first()
            #     mask = Z.eval("variable in @attr")
            #     return (
            #         Z[mask].unstack().droplevel(0,1).prep(bool=True)
            #         .join(Z[~mask])
            #         .query("levl_code=='ug' & styp_code in ('n','r','t')")
            #         .reset_index()
            #         .assign(**{k: lambda z: encrypt(z[k]) for k in ['pidm','id']})
            #         .set_index([*idx,*attr,'variable'])
            #         # .set_index(idx+attr)
            #     )
            # self.Z = {'current':g('current').query('admit'), 'final':g('final').query('matric')}
            # self.X = self.Z['current'].query("variable not in @courses").unstack().droplevel(0,1).prep(bool=True)
            # self.Y = {key: z.query("variable in @courses").droplevel(attr).prep(bool=True)}

            # g = lambda df: df.groupby(['term_code','levl_code','styp_code','variable','value']).size()
            # dct = {
            #     'admit' : g(self.Z['current']),
            #     'enroll': g(self.Z['current'].query('enroll')),
            #     'matric': g(self.Z['final']),
            # }
            # dct['mlt'] = dct['matric'] / g(self.Z['final'].query('admit'))
            # self.agg = pd.DataFrame(dct)
            # id = pd.concat(X.values())[[]].groupby(['id','pidm']).first()
            # id = pd.concat(X.values()).reset_index()[['id','pidm']].drop_duplicates()
            # Y = [pd.concat([
            # Y = {key: pd.concat([
            #         T.reg
            #         .assign(pidm=lambda x: encrypt(x['pidm']), variable='crse_code', value=lambda x: x['crse_code'])
            #         .fillna({'credit_hr':0})
            #         .query('credit_hr>0')
            #         .merge(id, how='left')
            #         .prep(bool=True)
            #         .sindex(idx)
            #         [['credit_hr']]
            #     for term_code, T in dct.items()]) for key, dct in self.terms.items()}
            #     # for term_code, T in dct.items()]) for dct in self.terms]
            # Y = {key: pd.concat([
            #         T.reg
            #         .assign(pidm=lambda x: encrypt(x['pidm']), variable='crse_code', value=lambda x: x['crse_code'])
            #         .fillna({'credit_hr':0})
            #         .query('credit_hr>0')
            #         .merge(id, how='left')
            #         .prep(bool=True)
            #         .sindex(idx)
            #         ['credit_hr']
            #         .rename(key)
            #     for term_code, T in dct.items()]) for key, dct in self.terms.items()}
            #     # for term_code, T in dct.items()]) for dct in self.terms]
            # self.X = X
            # self.Y = Y
            # return self
            # E = [y.query("value=='_allcrse'").droplevel(['variable','value']).squeeze() for y in Y]  # only rows for _allcrse
            # E[0] = E[0].rename('final').to_frame()
            # E[1] = E[1].rename('current').to_frame()

            # col = ['levl_code','styp_code']
            # X[0] = X[0].droplevel(col).join(E[0], how='outer')  # E[0] determines final levl_code, styp_code, & credit_hr
            # E[0] = E[0].droplevel(col).join(X[0][[]], how='outer')  # copy id from X[0] to E[0]
            # E[1] = E[1].droplevel(col).join(X[1][[]], how='outer')  # X[1] determines current levl_code & styp_code
            # X[1] = X[1].droplevel(col).join(E[1], how='outer')  # copy current credit_hr from E[1] to X[1]
            # X[1] = X[1].join(E[0].droplevel(col), how='outer')  # copy final credit_hr from E[0] to X[1]
            # # X[0] = X[0].join(E[1].droplevel(col), how='outer')  # copy current credit_hr from E[1] to X[0]
            # Y = [y[[]].droplevel(col).join(e[[]]) for y,e in zip(Y,E)]  # copy levl_code & stype_code from E to Y
            # qry = f"levl_code == 'ug' & styp_code in ('n','r','t')"
            # cols = sorted(c for c in X[0].select_dtypes(['string','boolean']).columns if "_missing" not in c and "_code" not in c)
            # idx.remove('id')
            # idx.remove('pidm')
            # aggy = lambda y: y.query(qry).groupby(idx).size()
            # aggx = lambda x: aggy(x[cols].melt(ignore_index=False))
            # def get_df(dct):
            #     Y = pd.concat(dct, axis=1).prep().fillna(0)
            #     Y['mlt'] = Y['final'] / Y['admitted']
            #     Y[np.isinf(Y)] = pd.NA
            #     return Y#.rename(columns=lambda x: "term_code_"+x)

            # AY = get_df({
            #     'current': aggy(Y[1]),
            #     'admitted': aggy(X[1][[]].join(Y[0].droplevel(col))),
            #     'final': aggy(Y[0]),
            # })
            # AX = get_df({
            #     'current': aggx(X[1].query("current.notnull()")),
            #     'admitted': aggx(X[1].query("final.notnull()")),
            #     'final': aggx(X[0].query("final.notnull()")),
            # })
            # self.mlt = pd.concat([AY,AX])
            # self.X = X[1].assign(credit_hr=X[1]['current'].fillna(0)).drop(columns=['current','final']).query(qry)
            # self.y = [y.rsindex(['pidm','term_code','value']) for y in Y]
        return self.get(func, fn="X.pkl",
                        pre="terms", drop=["terms"])


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
        return self.get(func, fn=f"X_proc/{self.styp_code}/{self.param['trf'][0]}/{self.param['imp'][0]}.pkl", pre="X")
                        # pre="X", drop=["terms","X","y","Z","agg"])


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
                print(train_code, end=":")
                X_model = Z.query("term_code==@train_code" if train_code != self.proj_code else "term_code!=@train_code").copy()
                if len(X_model) == 0:
                    print('fail', end=", ")
                    # pred = False
                    # proba = 0.0
                    # train_score[train_code] = pd.NA
                else:
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
                    print('done', end=", ")
                    y[train_code] = Z[['actual']].assign(pred=pred, proba=proba, train_code=train_code, crse_code=self.crse_code, clf_hash=self.param['clf'][0]).prep(bool=True)
                    # y[train_code] = Z[['actual']].assign(pred=pred, proba=proba, train_code='all' if train_code==self.proj_code else train_code, crse_code=self.crse_code, clf_hash=self.param['clf'][0]).prep(bool=True)
                # y[train_code] = Z[['actual']].assign(pred=pred, proba=proba).addlevel({'clf_hash':self.param['clf'][0], 'train_code':train_code, 'crse_code':self.crse_code}).prep(bool=True)
            if y:
                self.y_pred = pd.concat(y.values()).rsindex(['crse_code','term_code','pidm','train_code','sim','trf_hash','imp_hash','clf_hash'])
            self.train_score = pd.Series(train_score, name='train_score').rename_axis('train_code')
        return self.get(func, fn=f"y_pred/{self.styp_code}/{self.crse_code}/{self.param['trf'][0]}/{self.param['imp'][0]}/{self.param['clf'][0]}.pkl",
                        pre="X_proc", drop=["terms","X","y_true","Z","agg","X_trf","X_proc"])


    # def get_result(self):
    #     def func():
    #         Z = self.X_proc.reset_index().merge(self.y_pred.reset_index(), how='outer')
    #         def g(variable):
    #             grp = [variable,'term_code','levl_code','styp_code','train_code','sim','trf_hash','imp_hash','clf_hash']
    #             S = (Z
    #                 .groupby(grp).apply(lambda y: pd.Series({
    #                         'code_predict': y['proba'].sum(),
    #                         'test_score': log_loss(y['actual'], y['proba'], labels=[False,True]) * 100,
    #                     }), include_groups=False)
    #                 .join(self.train_score)
    #                 .join(self.agg)
    #             )
    #             alpha = 1
    #             S['overall_score'] = (S['train_score'] + alpha * S['test_score']) / (1 + alpha)
    #             qry = lambda q: S.query("term_code==@q").droplevel('term_code')
    #             S = (S
    #                 .join(qry(self.proj_code-100)['term_code_current'].rename(prior_current))
    #                 .join(qry(self.proj_code-100)['term_code_final'  ].rename(prior_final))
    #                 .join(qry(self.proj_code    )['term_code_current'].rename(proj_current))
    #                 .join(qry(self.proj_code    )['term_code_predict'].rename(proj_predict))
    #                 .query("term_code!=@self.proj_code")
    #                 .astype('Float64').fillna(0)
    #             )
    #             S['term_code_predict'] *= S['term_code_final'] > 0
    #             S.loc[S.eval('term_code_predict==0'), ['term_code_predict',proj_predict,'train_score','test_score','overall_score']] = pd.NA
    #             for k in ['term_code_predict',proj_predict]:
    #                 S[k] *= S['term_code_mlt']
    #             S[proj_change] = (S[proj_predict] / S[prior_final] - 1) * 100
    #             S['term_code_error'] = S['term_code_predict'] - S['term_code_final']
    #             S['term_code_error_pct'] = S['term_code_error'] / S['term_code_final'] * 100
    #             S = (
    #                 S[[prior_current,proj_current,prior_final,proj_predict,proj_change,'term_code_final','term_code_predict','term_code_error','term_code_error_pct','term_code_mlt','train_score','test_score','overall_score']]
    #                 .reset_index()
    #                 .sort_values([variable,'levl_code','styp_code','term_code','train_code','trf_hash','imp_hash','clf_hash'], ascending=[True,True,True,False,False,True,True,True])
    #                 .prep()
    #             )
    #             S['train_code'] = S['train_code'].astype('string').replace(str(self.proj_code), 'all')
    #         # if self.Y.shape[0] == 0:
    #         #     return
    #         # proj_current = f'{self.proj_code}_current'
    #         # proj_predict = f'{self.proj_code}_predict'
    #         # prior_current = f'{self.proj_code-100}_current'
    #         # prior_final = f'{self.proj_code-100}_final'
    #         # proj_change = f'{self.proj_code}_change_pct'
    #         # Z = self.X.join(self.Y, how='inner')



    #         self.result = dict()


    #         for variable in self.aggregations if self.crse_code == '_anycrse' else ['crse_code']:
    #             A = self.agg.query(f"variable==@variable")
    #             mask = A.eval(f"term_code!={self.proj_code}")
    #             B = A[mask]
    #             C = A[~mask].drop(columns='mlt').join(B['mlt'].rename(index={k:self.proj_code for k in self.term_codes}))
    #             M = pd.concat([B,C])

    #             # print(variable)
    #             grp = [variable,'levl_code','styp_code','term_code','train_code','sim','trf_hash','imp_hash','clf_hash']
    #             S = (Z
    #                 .groupby(grp).apply(lambda y: pd.Series({
    #                         'term_code_predict': y['proba'].sum(),
    #                         'test_score': log_loss(y['actual'], y['proba'], labels=[False,True]) * 100,
    #                     }), include_groups=False)
    #                 .join(self.train_score)
    #                 .join(self.mlt.query("variable==@variable").droplevel('variable').rename_axis(index={'value':variable}))
    #             )
    #             alpha = 1
    #             S['overall_score'] = (S['train_score'] + alpha * S['test_score']) / (1 + alpha)
    #             qry = lambda q: S.query("term_code==@q").droplevel('term_code')
    #             S = (S
    #                 .join(qry(self.proj_code-100)['term_code_current'].rename(prior_current))
    #                 .join(qry(self.proj_code-100)['term_code_final'  ].rename(prior_final))
    #                 .join(qry(self.proj_code    )['term_code_current'].rename(proj_current))
    #                 .join(qry(self.proj_code    )['term_code_predict'].rename(proj_predict))
    #                 .query("term_code!=@self.proj_code")
    #                 .astype('Float64').fillna(0)
    #             )
    #             S['term_code_predict'] *= S['term_code_final'] > 0
    #             S.loc[S.eval('term_code_predict==0'), ['term_code_predict',proj_predict,'train_score','test_score','overall_score']] = pd.NA
    #             for k in ['term_code_predict',proj_predict]:
    #                 S[k] *= S['term_code_mlt']
    #             S[proj_change] = (S[proj_predict] / S[prior_final] - 1) * 100
    #             S['term_code_error'] = S['term_code_predict'] - S['term_code_final']
    #             S['term_code_error_pct'] = S['term_code_error'] / S['term_code_final'] * 100
    #             S = (
    #                 S[[prior_current,proj_current,prior_final,proj_predict,proj_change,'term_code_final','term_code_predict','term_code_error','term_code_error_pct','term_code_mlt','train_score','test_score','overall_score']]
    #                 .reset_index()
    #                 .sort_values([variable,'levl_code','styp_code','term_code','train_code','trf_hash','imp_hash','clf_hash'], ascending=[True,True,True,False,False,True,True,True])
    #                 .prep()
    #             )
    #             S['train_code'] = S['train_code'].astype('string').replace(str(self.proj_code), 'all')
    #             grp.remove('sim')
    #             with warnings.catch_warnings(action='ignore'):
    #                 self.result[variable] = {'summary': S} | {str(stat): S.drop(columns='sim').groupby(grp, sort=False).agg(stat).prep() for stat in listify(self.stats)}
    #         self.result['crse_code']['mean'].disp(40)
    #     return self.get(func, fn=f"result/{self.styp_code}/{self.crse_code}/{self.param['trf'][0]}/{self.param['imp'][0]}/{self.param['clf'][0]}.pkl",
    #                     pre=["Y","X"], drop=["terms","X","y","mlt","X_trf","X_proc","clf","Y","train_score"])



    def get_result(self):
        def func():
            self.results = dict()
            self.Y_pred = pd.concat([read(fn).get('y_pred', pd.DataFrame()) for fn in sorted((self.root_path / 'y_pred').rglob('*.pkl'))]).prep().reset_index()
            with pd.ExcelWriter(self.root_path / f'AMP_{self.cycle_date.date()}.xlsx', mode='w', engine='openpyxl') as writer:
                for variable in self.aggregations:
                    grp = [variable,'levl_code','styp_code','term_code','train_code','sim','trf_hash','imp_hash','clf_hash']
                    S = (self.Y_pred.query("crse_code=='_anycrse'" if variable!="crse_code" else "crse_code.notnull()")
                        .merge(self.X[variable if variable!="crse_code" else []].reset_index())
                        .groupby(grp).apply(lambda y: pd.Series({
                                'predict': y['proba'].sum(),
                                'test_score': log_loss(y['actual'], y['proba'], labels=[False,True]) * 100,
                            }), include_groups=False)
                        .join(self.train_score)
                        .join(self.agg.loc[variable].rename_axis(index={'value':variable}))
                    )
                    P = S['actual'].rename('prior').reset_index()
                    P['term_code'] += 100
                    S = S.reset_index().merge(P, 'left')
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
                    M = S.groupby(grp, sort=False).mean().prep()
                    M.to_excel(writer, sheet_name=variable)
                    self.results[variable] = {'summary':S, 'mean':M}
        return self.get(func, f"results.pkl", pre=["y_pred","X_proc"])

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
        'early_stop': True,
        # 'time_budget': 5,
        # 'time_budget': 120,
        # 'time_budget': np.arange(100,200,10),
        # 'time_budget': [*np.arange(1,20),*np.arange(20,100,10),*np.arange(100,200,25),*np.arange(200,401,50)],
        'time_budget': 150,
        # 'estimator_list': [['lgbm','histgb']],#'xgboost']],#'catboost']],#'histgb','extra_tree','xgb_limitdepth','rf']],#'lrl1','lrl2','kneighbor'
        # 'estimator_list': [['xgboost']],#'xgboost']],#'catboost']],#'histgb','extra_tree','xgb_limitdepth','rf']],#'lrl1','lrl2','kneighbor'
        'estimator_list': [['xgboost']],#'catboost']],#'histgb','extra_tree','xgb_limitdepth','rf']],#'lrl1','lrl2','kneighbor'
        # 'estimator_list': [['xgboost']],#'histgb','extra_tree','xgb_limitdepth','rf']],#'lrl1','lrl2','kneighbor'
        # 'ensemble': [False, True],
        'ensemble': False,
        # 'ensemble': True,
        # 'gpu_per_trial': 1,
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

def run_amp(cycle_day, styp_codes=['n'], overwrite=[]):
    for kwargs in cartesian({'cycle_day':cycle_day, 'styp_code':styp_codes, 'crse_code':intersection(crse_codes, self.y_true.reset_index()['variable'], sort=True, reversed=True), 'param':param_lst, 'overwrite':[listify(overwrite)]}):
        self = AMP(**kwargs)
        self.get_y_pred()
        # if self.proj_code in self.clf:
        #     print(self.param['clf'][0], 'time_budget', self.param['clf'][2]['time_budget'], self.clf[self.proj_code].estimator)
    self.get_results()
    # def func():
    #     self.results = dict()
    #     self.Y_pred = pd.concat([read(fn).get('y_pred', pd.DataFrame()) for fn in sorted((self.root_path / 'y_pred').rglob('*.pkl'))]).prep().reset_index()
    #     # def g(variable):
    #     with pd.ExcelWriter(self.root_path / f'AMP_{self.cycle_date.date()}.xlsx', mode='w', engine='openpyxl') as writer:
    #         for variable in self.aggregations:
    #             grp = [variable,'levl_code','styp_code','term_code','train_code','sim','trf_hash','imp_hash','clf_hash']
    #             S = (self.Y_pred.query("crse_code=='_anycrse'" if variable!="crse_code" else "crse_code.notnull()")
    #                 .merge(self.X[variable if variable!="crse_code" else []].reset_index())
    #                 .groupby(grp).apply(lambda y: pd.Series({
    #                         'predict': y['proba'].sum(),
    #                         'test_score': log_loss(y['actual'], y['proba'], labels=[False,True]) * 100,
    #                     }), include_groups=False)
    #                 .join(self.train_score)
    #                 .join(self.agg.loc[variable].rename_axis(index={'value':variable}))
    #             )
    #             P = S['actual'].rename('prior').reset_index()
    #             P['term_code'] += 100
    #             S = S.reset_index().merge(P, 'left')
    #             alpha = 1
    #             S['overall_score'] = (S['train_score'] + alpha * S['test_score']) / (1 + alpha)
    #             S['predict'] *= S['mlt']
    #             S['predict_error'] = S['predict'] - S['actual']
    #             S['predict_error_pct'] = S['predict_error'] / S['actual'] * 100
    #             S['predict_change'] = S['predict'] - S['prior']
    #             S['predict_change_pct'] = S['predict_change'] / S['prior'] * 100
    #             S['train_code'] = S['train_code'].astype('string').replace(str(self.proj_code), 'all')
    #             S = (S
    #                 .prep()
    #                 .sort_values(grp, ascending=[True,True,True,False,False,True,True,True,True])
    #                 .set_index(grp)
    #                 [['admit','enroll','predict','prior','predict_change','predict_change_pct','actual','predict_error','predict_error_pct','overall_score','test_score','train_score','mlt']]
    #             )
    #             grp.remove('sim')
    #             M = S.groupby(grp, sort=False).mean().prep()
    #             M.to_excel(writer, sheet_name=variable)
    #             self.results[variable] = {'summary':S, 'mean':M}
    # return self.get(func, f"results.pkl", pre="X")
# drop=["terms","X","y","mlt","X_trf","X_proc","clf","Y","train_score","result"])
    
    
    # return self
    # def func():
    #     self.get_X()
    #     stack = {'Y': dict()}
    #     for fn in sorted((self.root_path / 'result').rglob('*.pkl')):
    #         self.load(fn, force=True)
    #         self.load(str(fn).replace('result','Y'), force=True)
    #         stack['Y'][self.crse_code] = self.Y
    #         for key, val in self.result.items():
    #             for stat, df in val.items():
    #                 stack.setdefault(key, dict()).setdefault(stat, dict())[self.crse_code] = df
    #     Y = stack['Y']['_allcrse'].groupby(['pidm','term_code']).agg(actual=('actual','mean'), proba_mean=('proba','mean'), proba_stdev=('proba','std'))
    #     self.stack = {'Z': self.X.join(Y).prep(bool=True)}
    #     self.stack['Z'].to_csv(self.root_path / f'AMP_details_{self.cycle_date.date()}.csv')
    #     with pd.ExcelWriter(self.root_path / f'AMP_{self.cycle_date.date()}.xlsx', mode='w', engine='openpyxl') as writer:
    #         for key, val in stack.items():
    #             if key == 'Y':
    #                 self.stack[key] = pd.concat(val.values())
    #                 continue
    #             if key == 'crse_code':
    #                 self.stack[key] = {stat: pd.concat(dct.values()) for stat, dct in val.items()}
    #             else:
    #                 self.stack[key] = {stat: dct['_allcrse'] for stat, dct in val.items()}
    #             self.stack[key]['mean'].droplevel(['trf_hash','imp_hash','clf_hash']).round(2).prep().to_excel(writer, sheet_name=k)
    # return self.get(func, f"stack.pkl", drop=["terms","X","y","mlt","X_trf","X_proc","clf","Y","train_score","result"])


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
        run_amp(98)