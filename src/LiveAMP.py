from term import *
import hashlib, miceforest as mf, flaml as fl
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, log_loss, fbeta_score, f1_score
from sklearn import set_config
set_config(transform_output="pandas")
warnings.filterwarnings("ignore", message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`")
pwrtrf = make_pipeline(StandardScaler(), PowerTransformer())
code_desc = lambda x: [x+'_code', x+'_desc']

################## Metrics that work for FLAML and sklearn ##################
class Metric(MyBaseClass):
    """Parent class for metrics that work with FLAML and sklearn"""
    def __str__(self):
        return self.__repr__()
    def __lt__(self, other):
        return str(self) < str(other)
    def __call__(self, X_val, y_val, estimator, labels, X_train, y_train, weight_val=None, weight_train=None, *args):
        """for FLAML"""
        start = time.time()
        y_pred = estimator.predict(X_val)
        pred_time = (time.time() - start) / len(X_val)
        val_loss = self.score(y_val, y_pred, labels=labels, sample_weight=weight_val)
        y_pred = estimator.predict(X_train)
        train_loss = self.score(y_train, y_pred, labels=labels, sample_weight=weight_train)
        return val_loss, {
            "val_loss": val_loss,
            "train_loss": train_loss,
            "pred_time": pred_time,
        }

class F_beta(Metric):
    """Implements F_beta for FLAML https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html"""
    def __init__(self, beta):
        self.beta = float(beta)
    def __repr__(self):
        return f"F_{self.beta:.2f}"
    def score(self, y_true, y_pred, **kwargs):
        return 1 - fbeta_score(y_true, y_pred, beta=self.beta, **kwargs)

class Accuracy(Metric):
    def __repr__(self):
        return "accuracy"
    def score(self, y_true, y_pred, **kwargs):
        return 1 - accuracy_score(y_true, y_pred, **kwargs)

class LogLoss(Metric):
    def __repr__(self):
        return "log loss"
    def score(self, y_true, y_pred, **kwargs):
        return log_loss(y_true, y_pred, **kwargs)

################## AMP ##################
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
    pred_codes: tuple = (202108, 202208, 202308, 202408)
    crse_code : str = '_allcrse'
    styp_code : str = 'n'
    stats: tuple = (pctl(0), pctl(25), pctl(50), pctl(75), pctl(100), )
    show: set = dataclasses.field(default_factory=set)
    param: dict = dataclasses.field(default_factory=dict)    
    root_path: str = r"C:\\Users\gavin\\Documents\\AMP\\admitted_matriculation_predictor"
    dependence: dict = dataclasses.field(default_factory=lambda: {'adm':'raw', 'flg':'raw', 'raw':'X', 'reg':'X', 'X':'X_proc', 'X_proc':'Y'})

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
            ren = {'term_code':'pred_code', 'term_desc':'pred_desc', 'index':'idx'}
            R = pd.concat([T.raw.rename(columns=ren) for pred_code, T in self.terms[1].items()]).copy().dropna(axis=1, how='all').reset_index(drop=True).prep()
            repl = {'ae':0, 'n1':1, 'n2':2, 'n3':3, 'n4':4, 'r1':1, 'r2':2, 'r3':3, 'r4':4}
            R['hs_qrtl'] = pd.cut(R['hs_pctl'], bins=[-1,25,50,75,90,101], labels=[4,3,2,1,0], right=False).combine_first(R['apdc_code'].map(repl))
            R['remote'] = R['camp_code'] != 's'
            R['resd'] = R['resd_code'] == 'r'
            R['oriented'] = R['oriented'] != 'n'
            R['lgcy'] = ~R['lgcy_code'].isin(['n','o'])
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

            attr = ['idx','id','pidm','levl_code','styp_code','pred_code']
            X = X.join(M).sample(frac=1).reset_index().rename(columns=ren).set_index(attr, drop=False).prep(bool=True)
            self.pii = X[[]].copy()
            anon = lambda df: df.rindex(['id','pidm'], drop=True).copy()
            get_pii = lambda df: self.pii.join(df.rindex(['levl_code','styp_code'], drop=True), how='inner').copy()
            self.X = anon(X)
            
            Y = [pd.concat([
                    T.reg
                    .rename(columns=ren)
                    .set_index(['pidm','levl_code','styp_code','pred_code','crse_code'])
                    ['credit_hr']
                    .fillna(0)
                    .query('credit_hr>0')
                for pred_code, T in dct.items()]) for dct in self.terms]
            self.y = {'credit': get_pii(Y[1].query("crse_code=='_allcrse'").droplevel("crse_code"))}
            Y = [(y>0).rename('enrolled') for y in Y]
            self.y['census'] = Y[0].copy()
            self.y['admit'] = get_pii(Y[0])
            self.y['regstr'] = get_pii(Y[1])
            self.y = {k: anon(v).squeeze().rename(k) for k,v in self.y.items()}
            self.y['actual'] = self.y['admit'].copy().rename('actual')
            agg = lambda y: y.groupby(['styp_code','pred_code','crse_code']).sum().query(f"styp_code in ('n','r','t')")
            self.mlt = (
                agg(self.y['regstr']).to_frame()
                .join(agg(self.y['admit']), how='outer')
                .join(agg(self.y['census']), how='outer')
                .fillna(0)
            )
            self.mlt['regstr_pct'] = self.mlt['regstr'] / self.mlt['census'] * 100
            self.mlt['admit_pct' ] = self.mlt['admit' ] / self.mlt['census'] * 100
            # self.mlt['mlt'       ] = self.mlt['census'] / self.mlt['admit' ]
            self.mlt['mlt'] = 100 / self.mlt['admit_pct']
            self.mlt[np.isnan(self.mlt) | np.isinf(self.mlt)] = pd.NA
            write(self.root_path / 'pii.parq', self.pii)
        return self.get(func, "X.pkl", pre="terms", drop=["terms","pii"])


    def get_X_proc(self):
        def func():
            attr = []
            X = self.X.query(f"styp_code=='{self.styp_code}'").set_index(attr, drop=False, append=True).rename(columns=lambda x:'__'+x)
            trf = ColumnTransformer(self.param['trf'][2], remainder='drop', verbose_feature_names_out=False)
            X_trf = trf.fit_transform(X).prep(bool=True, cat=True).sample(frac=1)
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
            g = lambda y: y.query(f"crse_code=='{self.crse_code}'").droplevel('crse_code')
            Z = self.X_proc.copy()
            if 'registration' in dct and dct.pop('registration'):
                Z = Z.join(self.y['credit']).join(g(self.y['regstr']))
            Z = Z.join(g(self.y['actual'])).fillna({'actual':False, 'regstr':False, 'credit':0}).prep(bool=True, cat=True)

            self.clf = dict()
            self.Y = dict()
            self.train_score = dict()
            for train_code in self.pred_codes:
                print(f"...{train_code}", end="")
                qry = f"pred_code=={train_code}" if train_code != self.proj_code else f"pred_code!={train_code}"
                X_model = Z.query(qry).groupby(['pred_code','sim']).filter(lambda x:x['actual'].sum()>=4).copy()
                if len(X_model) == 0:
                    # print(train_code, 'not enough')
                    pred = False
                    proba = 0.0
                    self.train_score[train_code] = pd.NA
                else:
                    G = X_model.query(f"sim==0").groupby(['pred_code','actual'], observed=True).ngroup().rename('grp').to_frame().rsindex('idx')
                    y_model = X_model.pop('actual')
                    for i in range(10000):
                        idx = G.groupby('grp').sample(frac=0.75, random_state=i).index
                        msk = X_model.eval('idx.isin(@idx)')
                        if y_model[msk].any() & y_model[~msk].any():
                            break
                        print(i, y_model[msk].sum(), y_model[~msk].sum())
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
            self.train_score = pd.Series(self.train_score, name='train_score').rename_axis('pred_code')
        return self.get(func,
                        f"Y/{self.styp_code}/{self.crse_code}/{self.param['trf'][0]}/{self.param['imp'][0]}/{self.param['clf'][0]}.pkl",
                        pre="X_proc",
                        drop=["terms","X","y","mlt","X_proc"])


    def get_summary(self):
        def func():
            if 'Y' not in self:
                return
            grp = ['crse_code','levl_code','styp_code','train_code','pred_code','trf_hash','imp_hash','clf_hash','sim']
            S = self.Y.groupby(grp).apply(lambda y: pd.Series({
                'predicted': y['proba'].sum(),
                'test_score': log_loss(y['actual'], y['proba'], labels=[False,True]) * 100,
            })).join(self.train_score)
            proj_rgstr = f'{self.proj_code}_current'
            proj_pred = f'{self.proj_code}_projection'
            proj_chg = f'{self.proj_code}_change_pct'
            qry = f"pred_code=={self.proj_code}"
            S = (
                S.query('not '+qry)
                .join(S.query(qry)['predicted'].droplevel('pred_code').rename(proj_pred))
                .join(self.mlt.query(qry)['regstr'].droplevel('pred_code').rename(proj_rgstr))
                .join(self.mlt)
                .sort_index()
            ).prep().fillna(0)
            S['predicted'] *= S['census'] > 0
            for k in ['predicted',proj_pred]:
                S[k] *= S['mlt']
            S.loc[S.eval('predicted==0'), 'predicted':proj_pred] = pd.NA
            S[proj_chg] = (S[proj_pred] / S.groupby(S.index.names.difference({'pred_code'})).transform('last')['census'] - 1) * 100
            alpha = 1
            S['overall_score'] = (S['train_score'] + alpha * S['test_score']) / (1 + alpha)
            S['error'] = S['predicted'] - S['census']
            S['error_pct'] = S['error'] / S['census'] * 100
            with warnings.catch_warnings(action='ignore'):
                S = S[[proj_rgstr,proj_pred,proj_chg,'regstr','admit','census','predicted','error','error_pct','overall_score','test_score','train_score','mlt','regstr_pct','admit_pct']].reset_index().prep()
                S['train_code'] = S['train_code'].astype('string').replace(str(self.proj_code), 'all')
                self.summary = S#.sort_values(['train_code','pred_code'], ascending=False)
                grp.remove('sim')
                self.rslt = {str(stat): self.summary.drop(columns='sim').groupby(grp).agg(stat).sort_index(ascending=False) for stat in self.stats}
            self.rslt[' 50%'].disp(100)
        return self.get(func, f"summary/{self.styp_code}/{self.crse_code}/{self.param['trf'][0]}/{self.param['imp'][0]}/{self.param['clf'][0]}.pkl",
                        pre=["Y","X"], drop=["terms","X","y","mlt","X_proc","clf","Y"])


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
        'distance': [pwrtrf],
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
    },
    'imp': {
        'random_state': 42,
        'datasets': 10,
        'iterations': 10,
        'tune': False,
        # 'tune': [False, True],
    },
    'clf': {
        'registration': True,
        'seed': 42,
        'metric': 'log_loss',
        'early_stop': True,
        # 'time_budget': 150,
        'time_budget': 5,
        'estimator_list': [['lgbm','xgboost','catboost','histgb','extra_tree','xgb_limitdepth','rf']],#'lrl1','lrl2','kneighbor'
        # 'ensemble': [False, True],
        'ensemble': False,
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


def run_amp(cycle_day, styp_codes=['n'], overwrite=None):
    for kwargs in cartesian({'cycle_day': cycle_day, 'styp_code': styp_codes, 'crse_code': crse_codes, 'param': cartesian(param_dcts), 'overwrite': [listify(overwrite)]}):
        self = AMP(**kwargs)
        self.get_X()
        self.get_X_proc()
        self.get_summary()
    def func():
        self.stack = dict()
        append = lambda k, v: self.stack.setdefault(k, []).append(v.copy())
        for fn in (self.root_path / 'summary').rglob('*.pkl'):
            print(fn)
            self.load(fn, force=True)
            self.load(str(fn).replace('summary','Y'))
            for k in ['Y', 'summary']:
                append(k, self[k])
            for k, v in self.rslt.items():
                append(k, v)
        self.stack = {k: pd.concat(v).prep() for k, v in self.stack.items()}
        self.report = (
            self.stack[' 50%']
            .reset_index()
            .drop(columns=['trf_hash','imp_hash','clf_hash'])
            .sort_values(['crse_code','levl_code','styp_code','train_code','pred_code'], ascending=[True,True,True,False,False])
            .round(2)
            .prep()
        )
        self.report.to_csv(self.root_path / f'AMP_{self.cycle_date.date()}.csv', index=False)
    return self.get(func, f"stack.pkl", drop=["terms","X","y","mlt","X_proc","clf","Y","summary","rslt"])


if __name__ == "__main__":
    print(pd.Timestamp.now())
    delattr(pd.Series, 'disp')
    delattr(pd.DataFrame, 'disp')
    @pd_ext
    def disp(df, max_rows=4, max_cols=200, **kwargs):
        print()
        print(df.reset_index().drop(columns='index', errors='ignore').head(max_rows).to_markdown(tablefmt='psql'))

    from IPython.utils.io import Tee
    with contextlib.closing(Tee(r'C:\\Users\gavin\\Documents\\AMP\\admitted_matriculation_predictor\\log.txt', "w", channel="stdout")) as outputstream:
        run_amp(119)




    # def get_Y(self):
    #     def func():
    #         clf_dct = self.param['clf'][2] | {'task':'classification', 'verbose':0}#, 'log_type': 'all'}
    #         max_iter = clf_dct.pop('max_iter')

    #         cols = uniquify(['_allcrse_cur', self.crse_code+'_cur', self.crse_code])
    #         y = self.y.query(f"crse_code.isin({cols})").unstack().droplevel(0,1)
    #         Z = (
    #             self.X_proc
    #             .join(y)
    #             .fillna({c:0 for c in cols})
    #             .addlevel({'crse_code':self.crse_code, 'train_code':self.train_code, 'clf_hash':self.param['clf'][0]})
    #             .rsindex(['index','crse_code','levl_code','styp_code','train_code','pred_code','trf_hash','imp_hash','clf_hash','sim'])
    #             .prep(bool=True, cat=True)
    #         )
    #         if self.crse_code not in Z:
    #             Z[self.crse_code] = False
    #         if Z.query(f"pred_code=={self.train_code} & sim==0")[self.crse_code].sum() < 10:
    #             self.Y = Z[self.crse_code].rename('actual').to_frame().assign(proba=0.0).copy()
    #             self.weight = 0
    #             self.train_score = 0
    #         else:
    #             X = Z.query(f"pred_code=={self.train_code}").copy()
    #             t = X.query(f"sim==0").groupby([self.crse_code,'__coll_code'], observed=True)
    #             y = X.pop(self.crse_code)
    #             # mask = X.reset_index()['index'].isin(t.sample(frac=0.75, random_state=clf_dct['seed']).reset_index()['index']).values
    #             def train(wgt, iter=''):
    #                 mask = X.reset_index()['index'].isin(t.sample(frac=0.75, random_state=clf_dct['seed']).reset_index()['index']).values
    #                 dct = clf_dct | {
    #                     'X_train':X[mask],
    #                     'y_train':y[mask],
    #                     'X_val':X[~mask],
    #                     'y_val':y[~mask],
    #                     'sample_weight':1+(2*y[mask]-1)*wgt,
    #                     'sample_weight_val':1+(2*y[~mask]-1)*wgt,
    #                     'log_file_name': self.path.with_stem(f"{self.path.stem}{iter}").with_suffix('.log'),
    #                 }
    #                 mkdir(dct['log_file_name'].parent)
    #                 clf = fl.AutoML(**dct)
    #                 clf.fit(**dct)
    #                 X_all = Z.copy()
    #                 clf.Y = X_all.pop(self.crse_code).rename('actual').to_frame().assign(proba=clf.predict_proba(X_all)[:,1]).prep(bool=True).copy()
    #                 return clf

    #             def score(clf):
    #                 # return log_loss(clf.Y['actual'], clf.Y['proba'])
    #                 S = clf.Y.groupby('pred_code').sum().query(f"pred_code!={self.proj_code}")
    #                 S['proba'] *= S['actual'] > 0
    #                 return S['proba'].sum() / S['actual'].sum() - 1
    #                 # S['err'] = S['proba'] - S['actual']
    #                 # return S['err'].sum() / S['actual'].sum()
                
    #             # def minimizer():
    #             #     res = sp.optimize.minimize_scalar(lambda wgt: abs(score(train(wgt))), bounds=(-1,1), options={'xatol':1e-8, 'maxiter':max_iter})
    #             #     print(res)
    #             #     return res.x

    #             def minimizer(anneal=1):
    #                 min_iter = 5
    #                 wgt = 0
    #                 err = 0
    #                 best_wgt = wgt
    #                 best_rmse = np.inf
    #                 self.hist = dict()
    #                 for i in range(max_iter):
    #                     wgt = np.clip(wgt - err*(anneal**i), -1, 1)
    #                     err = score(train(wgt))
    #                     self.hist[wgt] = err
    #                     W = np.array(list(self.hist.keys  ())[-min_iter:])
    #                     E = np.array(list(self.hist.values())[-min_iter:])
    #                     rmse = np.sqrt(np.mean(E**2))
    #                     if min_iter <= len(E) and rmse < best_rmse:
    #                         best_wgt = W.mean()
    #                         best_rmse = rmse
    #                     print(rjust(i,3), f'wgt={wgt: .12f}', f'best_wgt={best_wgt: .12f}', f'err={err: .12f}', f'rmse={rmse:.12f}', f'best_rmse={best_rmse:.12f}')
    #                     if best_rmse < 0.00001:
    #                         break
    #                 clf = train(best_wgt)
    #                 print(f"best_wgt={best_wgt: .5f} with error={score(clf): .5f}")
    #                 return best_wgt

    #             print('minimizing')
    #             with Timer():
    #                 self.weight = minimizer()

    #             clf_dct['time_budget'] *= 10
    #             clf = train(self.weight)
    #             self.Y = clf.Y
    #             self.train_score = clf.best_result['val_loss'] * 100
    #             self.clf = clf._trained_estimator
    #         for k in ['X','y','mlt','X_proc']:
    #             del self[k]
    #     return self.get(func, f"Y/{self.styp_code}/{self.crse_code}/{self.train_code}/{self.param['trf'][0]}/{self.param['imp'][0]}/{self.param['clf'][0]}.pkl", "X_proc")

    # def get_Y(self):
    #     def func():
    #         clf_dct = self.param['clf'][2] | {'task':'classification', 'verbose':0}#, 'log_type': 'all'}
    #         max_iter = clf_dct.pop('max_iter')
    #         # y = self.y.query(f"crse_code.isin([{self.crse_code}_cur', {self.crse_code}])").unstack().droplevel(0,1)
    #         # y = self.y_adm.query(f"crse_code=='{self.crse_code}'").unstack()
    #         g = lambda y: y.query(f"crse_code=='{self.crse_code}'").droplevel('crse_code')
    #         # y = {k:y.query(f"crse_code=='{self.crse_code}'").droplevel('crse_code') for k,y in self.y.items()}
    #         # y = self.y.query(f"crse_code=='{self.crse_code}'").droplevel('crse_code')
    #         Z = (
    #             self.X_proc
    #             # .join(  self.y['credit']) .fillna({'credit':0})
    #             # .join(g(self.y['regstr'])).fillna({'regstr':False})
    #             .join(g(self.y['actual'])).fillna({'actual':False})
    #             .addlevel({'crse_code':self.crse_code, 'train_code':self.train_code, 'clf_hash':self.param['clf'][0]})
    #             # .rsindex(['index','crse_code','levl_code','styp_code','train_code','pred_code','trf_hash','imp_hash','clf_hash','sim'])
    #             .prep(bool=True, cat=True)
    #         )
    #         # Z.disp(2)
    #         # assert 1==2
    #         self.weight = 0
    #         self.train_score = 0
    #         # if self.crse_code not in Z:
    #         #     Z[self.crse_code] = False
    #         # if Z.query(f"pred_code=={self.train_code} & sim==0")[self.crse_code].sum() < 10:
    #         if Z.query(f"pred_code=={self.train_code} & sim==0")['actual'].sum() < 10:
    #             self.Y = Z['actual'].to_frame().assign(proba=0.0).copy()
    #         else:
    #             # X = Z.query(f"pred_code=={self.train_code}").copy()
    #             # t = X.query(f"sim==0").groupby([self.crse_code,'__coll_code'], observed=True)
    #             # y = X.pop(self.crse_code)
    #             # mask = X.reset_index()['index'].isin(t.sample(frac=0.75, random_state=clf_dct['seed']).reset_index()['index']).values
    #             # X = Z.query(f"pred_code!={self.proj_code}").copy()
    #             # y = X.pop(self.crse_code)
    #             # mask = X.eval(f"pred_code=={self.train_code}")

    #             def train(Z, wgt=0, iter=''):
    #                 mask = Z.eval(f"pred_code=={self.train_code}")
    #                 X = Z.copy()
    #                 y = X.pop('actual')
    #                 dct = clf_dct | {
    #                     'X_train':X[mask],
    #                     'y_train':y[mask],
    #                     # 'eval_method':'cv',
    #                     # 'n_splits':5,
    #                     # 'split_type':'stratified',
    #                     'X_val':X,
    #                     'y_val':y,
    #                     'sample_weight':1+(2*y[mask]-1)*wgt,
    #                     'sample_weight_val':1+(2*y-1)*wgt,
    #                     'log_file_name': self.path.with_stem(f"{self.path.stem}{iter}").with_suffix('.log'),
    #                 }
    #                 mkdir(dct['log_file_name'].parent)
    #                 self.clf = fl.AutoML(**dct)
    #                 self.clf.fit(**dct)
    #                 self.Y = Z.filter(['regstr','actual']).assign(proba=self.clf.predict_proba(X)[:,1]).prep(bool=True).copy()
    #                 # self.Y = X[self.crse_code+'_cur'].rename('current').to_frame().assign(actual=y, proba=self.clf.predict_proba(X)[:,1]).prep(bool=True).copy()
                    
    #                 # y.rename('actual').to_frame().assign(proba=self.clf.predict_proba(X)[:,1]).prep(bool=True).copy()


    #             def score(wgt):
    #                 Z_trn = Z.query(f"pred_code!={self.proj_code}").copy()
    #                 train(Z_trn, wgt)
    #                 S = self.Y.groupby('pred_code').sum()
    #                 S['proba'] *= S['actual'] > 0
    #                 S['err'] = S['proba'] - S['actual']
    #                 S['err_pct'] = (S['err'] / S['actual']) * 100
    #                 S.disp(5)
    #                 # err = S['proba'].sum() / S['actual'].sum() - 1
    #                 # err = np.sqrt(np.mean(S['err']**2)) + S['err'].sum()**2
    #                 # err = np.mean(np.abs(S['err'])) + np.abs(np.sum(S['err']))
    #                 # err = np.mean(np.abs(S['err'])) + 10*np.abs(np.sum(S['err']))
    #                 # err = (S['err'].sum())**2
    #                 err = np.sum(S['err']**2)
    #                 print(wgt, err)
    #                 return err
    #                 # return clf.best_result['val_loss']
    #                 # return log_loss(clf.Y['actual'], clf.Y['proba'])
    #                 # S['err'] = S['proba'] - S['actual']
    #                 # return S['err'].sum() / S['actual'].sum()

    #             def minimizer():
    #                 res = sp.optimize.shgo(score, bounds=[(-1,1)], sampling_method='sobol')#, workers=-1)
    #                 print(res)
    #                 return res.x
                
    #             # def minimizer():
    #             #     res = sp.optimize.minimize_scalar(lambda wgt: abs(score(wgt)), bounds=(-1,1), options={'xatol':1e-8, 'maxiter':max_iter})
    #             #     print(res)
    #             #     return res.x

    #             # def minimizer(anneal=1):
    #             #     min_iter = 5
    #             #     wgt = 0
    #             #     err = 0
    #             #     best_wgt = wgt
    #             #     best_rmse = np.inf
    #             #     self.hist = dict()
    #             #     for i in range(max_iter):
    #             #         wgt = np.clip(wgt - err*(anneal**i), -1, 1)
    #             #         err = score(train(wgt))
    #             #         self.hist[wgt] = err
    #             #         W = np.array(list(self.hist.keys  ())[-min_iter:])
    #             #         E = np.array(list(self.hist.values())[-min_iter:])
    #             #         rmse = np.sqrt(np.mean(E**2))
    #             #         if min_iter <= len(E) and rmse < best_rmse:
    #             #             best_wgt = W.mean()
    #             #             best_rmse = rmse
    #             #         print(rjust(i,3), f'wgt={wgt: .12f}', f'best_wgt={best_wgt: .12f}', f'err={err: .12f}', f'rmse={rmse:.12f}', f'best_rmse={best_rmse:.12f}')
    #             #         if best_rmse < 0.00001:
    #             #             break
    #             #     clf = train(best_wgt)
    #             #     print(f"best_wgt={best_wgt: .5f} with error={score(clf): .5f}")
    #             #     return best_wgt

    #             # def minimizer():
    #             #     min_iter = 5
    #             #     max_iter = 20
    #             #     wgt = 0
    #             #     dct = dict()
    #             #     for i in range(max_iter):
    #             #         err = score(wgt)
    #             #         dct[wgt] = abs(err)
    #             #         print(i, wgt, err)
    #             #         wgt += (2*(err<0)-1) / 2**(i+1)
    #             #     best_wgt = min(dct, key=dct.get)
    #             #     print(best_wgt, dct[best_wgt])
    #             #     return best_wgt


    #             print('minimizing')
    #             with Timer():
    #                 self.weight = minimizer()
                
    #             clf_dct['time_budget'] *= 5
    #             train(Z, self.weight)
    #             # X = Z.copy()
    #             # y = X.pop(self.crse_code)
    #             # self.Y = y.rename('actual').to_frame().assign(proba=self.clf.predict_proba(X)[:,1]).prep(bool=True).copy()
    #             # self.Y = Z[self.crse_code].rename('actual').to_frame().assign(proba=clf.predict_proba(X)[:,1]).prep(bool=True).copy()
    #             # self.Y = y.rename('actual').to_frame().assign(proba=clf.predict(X)).prep(bool=True).copy()
    #             # self.Y = clf.Y
    #             self.train_score = self.clf.best_result['val_loss'] * 100
    #             self.clf = self.clf._trained_estimator
    #         for k in ['X','y','mlt','X_proc']:
    #             del self[k]
    #     return self.get(func, f"Y/{self.styp_code}/{self.crse_code}/{self.train_code}/{self.param['trf'][0]}/{self.param['imp'][0]}/{self.param['clf'][0]}.pkl", "X_proc")



            



            # Z.groupby(['pred_code',])
            # # T = Z.query(f"pred_code!={self.proj_code} & sim==0").groupby('pred_code')['actual'].sum().query('actual>=10').index.tolist()
            # if len(T) > 0:
            #     T.append(self.proj_code)
            #     Z = Z.query(f"pred_code.isin({T})").copy()
            #     X = Z.query(f"pred_code!={self.proj_code}").copy()
            #     idx = X.query(f"sim==0").groupby(['actual','__coll_code'], observed=True).sample(frac=0.75, random_state=clf_dct['seed']).reset_index()['index']
            #     msk = X.reset_index()['index'].isin(idx).values
            #     y = X.pop('actual')


                # X = Z.copy()
                # y = X.pop('actual')
                # X_mdl = Z.query(f"pred_code!={self.proj_code}").copy()
                # trn_idx = X_mdl.query(f"sim==0").groupby(['actual','__coll_code'], observed=True).sample(frac=0.75, random_state=clf_dct['seed']).reset_index()['index']
                # trn_msk = X_mdl.reset_index()['index'].isin(trn_idx).values
                # y_mdl = X_mdl.pop('actual')
                # dct = clf_dct | {
                #     'X_train':X[msk],
                #     'y_train':y[msk],
                #     # 'eval_method':'cv',
                #     # 'n_splits':5,
                #     # 'split_type':'stratified',
                #     'X_val':X[~msk],
                #     'y_val':y[~msk],
                #     # 'sample_weight':1+(2*y[mask]-1)*wgt,
                #     # 'sample_weight_val':1+(2*y-1)*wgt,
                #     # 'log_file_name': self.path.with_stem(f"{self.path.stem}{iter}").with_suffix('.log'),
                # }
                # # mkdir(dct['log_file_name'].parent)
                # self.clf = fl.AutoML(**dct)
                # self.clf.fit(**dct)
                # self.Y = Z[['actual']].assign(proba=self.clf.predict_proba(Z.drop(columns='actual'))[:,1]).prep(bool=True).copy()
                # self.train_score = self.clf.best_result['val_loss'] * 100
                # self.clf = self.clf._trained_estimator



# def get_stack(cycle_day, ext='', **kwargs):
#     self = AMP(cycle_day=cycle_day, **kwargs)
#     def func():
#         self.stack = dict()
#         append = lambda k, v: self.stack.setdefault(k, []).append(v)
#         for fn in (self.root_path / 'summary').rglob('*.pkl'):
#             print(fn)
#             self.load(fn, force=True)
#             self.load(str(fn).replace('summary','Y'))
#         #     # A.mlt = self.mlt.copy()
#         #     # A.summarize()
#         #     # A.dump(fn)
#             for k in ['Y', 'summary']:
#                 append(k, self[k])
#             for k, v in self.rslt.items():
#                 append(k, v)
#         self.stack = {k: pd.concat(v).prep() for k, v in self.stack.items()}
#         self.report = (
#             self.stack[' 50%']
#             .query("trf_hash=='fa15'")
#             .reset_index()
#             .drop(columns=['trf_hash','imp_hash','clf_hash'])
#             # .sort_values(['crse_code','levl_code','styp_code','train_code','pred_code'], ascending=[True,True,True,False,False])
#             .sort_values(['crse_code','levl_code','styp_code','pred_code'], ascending=[True,True,True,False])
#             .round(2)
#             .prep()
#         )
#         self.report.to_csv(self.root_path / f'AMP_{self.cycle_date.date()}.csv', index=False)
#     return self.get(func, f"stack.pkl", "X")


            # attr = [
            #     'idx',
            #     'id',
            #     'pidm',
            #     *code_desc('levl'),
            #     *code_desc('styp'),
            #     *code_desc('pred'),
            #     # *code_desc('apdc'),
            #     # *code_desc('admt'),
            #     # *code_desc('camp'),
            #     # *code_desc('coll'),
            #     # *code_desc('dept'),
            #     # *code_desc('majr'),
            #     # *code_desc('cnty'),
            #     # *code_desc('stat'),
            #     # *code_desc('natn'),
            #     # *code_desc('resd'),
            #     # *code_desc('lgcy'),
            #     # 'international',
            #     # 'gender',
            #     # *[f'race_{r}' for r in ['american_indian','asian','black','pacific','white','hispanic']],
            #     # 'waiver',
            #     # 'birth_day',
            #     # 'distance',
            #     # 'hs_qrtl',
            # ]