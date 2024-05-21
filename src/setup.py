import os, sys, warnings, time, datetime, contextlib, io, dataclasses, pathlib, shutil, dill, dotenv, itertools as it
import oracledb, numpy as np, scipy as sp, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, HTML, clear_output
from copy import deepcopy
from codetiming import Timer
warnings.filterwarnings("ignore", message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`")
dotenv.load_dotenv()
tab = '    '

##################### iterables helpers #####################
def listify(X, sort=False, **kwargs):
    """Turns X into a list"""
    if X is None or X is np.nan:
        X = []
    elif isinstance(X, (str,int,float,bool)) or callable(X):
        X = [X]
    elif isinstance(X, dict):
        X = list(X.values())
    else:
        X = list(X)
    return sorted(X, **kwargs) if sort else X

def tuplify(X, sort=False, **kwargs):
    return tuple(listify(X, sort, **kwargs))

def setify(X):
    return set(listify(X))

def mysort(X, **kwargs):
    if isinstance(X, dict):
        return dict(sorted(X.items(), **kwargs))
    else:
        return listify(X, True, **kwargs)

def cartesian(dct, sort=False, **kwargs):
    """Creates the Cartesian product of a dictionary with list-like values"""
    dct = mysort(dct, **kwargs) if sort else dct
    dct = {key: listify(val, sort, **kwargs) for key, val in dct.items()}
    dct = {key: val for key, val in dct.items() if len(val) > 0}
    return [dict(zip(dct.keys(), x)) for x in it.product(*dct.values())]

def uniquify(X, sort=False, **kwargs):
    if not isinstance(X, dict):
        X = listify(dict.fromkeys(listify(X)).keys())
    return mysort(X, **kwargs) if sort else X

def intersection(*args, sort=False, **kwargs):
    L = [listify(x) for x in listify(args)]
    y = [x for x in L[0] if x in set(L[0]).intersection(*L)]
    return mysort(y, **kwargs) if sort else y

def nest(path, dct=dict(), val=None):
    path = listify(path.values() if isinstance(path, dict) else path)
    k = path.pop(-1)
    a = dct
    for p in path:
        a.setdefault(p, dict())
        a = a[p]
    if val is None:
        return a[k] if k in a else None
    else:
        a[k] = val
        return dct

##################### string helpers #####################
def rjust(x, width, fillchar=' '):
    return str(x).rjust(width,str(fillchar))

def ljust(x, width, fillchar=' '):
    return str(x).ljust(width,str(fillchar))

def join(x, sep=', ', pre='', post=''):
    return f"{pre}{str(sep).join(map(str,listify(x)))}{post}"

def indent(qry, lev=1):
    return ('\n'+qry.strip()).replace('\n','\n'+tab*lev) if lev>=0 else qry

def isnum(x):
    return str(x).isnumeric()

def isqry(qry):
    return 'select' in qry.lower()

def mkqry(qry):
    return qry.strip() if isqry(qry) else f"select * from {qry}"

def subqry(qry, lev=1):
    return "(" + indent(qry.strip(), lev) + indent(")", lev-1)# if lev>0 else qry

def encrypt(plain):
    return plain ^ int(os.environ.get('ENCRYPT_KEY'))

def decrypt(crypt):
    return crypt ^ int(os.environ.get('ENCRYPT_KEY'))

##################### statistics helpers #####################
class pctl():
    def __init__(self, p):
        self.p = round(p if p > 1 else p*100)
        self.__name__ = f'{self.p}%'.rjust(4)
    def __str__(self):
        return self.__name__
    def __call__(self, x):
        return np.quantile(x, self.p/100)

# class IQR():
#     return pctl(75)(x)-pctl(25)(x)

# def ran(x):
#     return pctl(100)(x)-pctl(0)(x)

##################### pandas helpers #####################
def pd_ext(func):
    def wrapper(X, *args, **kwargs):
        try:
            Y = func(X, *args, **kwargs)
        except:
            Y = pd.DataFrame(X)
            try:
                Y = func(Y, *args, **kwargs)
            except:
                Y = Y.apply(func, *args, **kwargs)
        try:
            assert isinstance(X, pd.Series)
            return Y.iloc[:,0]
        except:
            return Y
    wrapper.__name__ = func.__name__
    for cls in [pd.DataFrame, pd.Series]:
        if not hasattr(cls, wrapper.__name__):
            setattr(cls, wrapper.__name__, wrapper)
    return wrapper

@pd_ext
def disp(df, max_rows=1, max_cols=200, **kwargs):
    display(HTML(df.to_html(max_rows=max_rows, max_cols=max_cols, **kwargs)))

@pd_ext
def query(df, *args, **kwargs):
    return df.query(*args, **kwargs)

@pd_ext
def eval(df, *args, **kwargs):
    return df.eval(*args, **kwargs)

@pd_ext
def convert(ser, bool=False, cat=False, dtype_backend='numpy_nullable'):
    assert isinstance(ser, pd.Series)
    if pd.api.types.is_string_dtype(ser) or pd.api.types.is_object_dtype(ser):
        ser = ser.astype('string')
        try:
            ser = pd.to_datetime(ser)
        except ValueError:
            try:
                ser = pd.to_numeric(ser, downcast='integer')
            except ValueError:
                ser = ser.str.lower().replace('', pd.NA)
    if pd.api.types.is_numeric_dtype(ser):
        ser = pd.to_numeric(ser, downcast='integer')
        if pd.api.types.is_integer_dtype(ser):
            ser = ser.astype('Int64')
    if bool:
        vals = set(ser.dropna().unique())
        for L in [['false','true'], [0,1], ['n','y']]:
            if vals.issubset(L):
                ser = (ser == L[1]).astype('boolean').fillna(False)
    if cat and pd.api.types.is_string_dtype(ser):
        ser = ser.astype('category')
    with warnings.catch_warnings(action='ignore'):
        return ser.convert_dtypes(dtype_backend)

@pd_ext
def prep(X, cap='casefold', bool=False, cat=False):
    if isinstance(X, str):
        return (getattr(X, cap)() if cap else X).strip()
    elif isinstance(X, (list, tuple, set, pd.Index)):
        return type(X)(prep(x, cap) for x in X)
    elif isinstance(X, dict):
        return {prep(k, cap): prep(v, cap) for k, v in X.items()}
    elif isinstance(X, pd.Series):
        assert 1==2
    elif isinstance(X, pd.DataFrame):
        g = lambda x: prep(x, cap).replace(' ','_').replace('-','_') if isinstance(x, str) else x
        return X.rename(columns=g).rename_axis(index=g).convert(bool=bool, cat=cat)
        # X = X.rename(columns=g).rename_axis(index=g)
        # idx = pd.MultiIndex.from_frame(X[[]].reset_index().convert(bool=bool, cat=cat))
        # return X.convert(bool=bool, cat=cat).set_index(idx).rename_axis(X.index.names)
    else:
        return X

@pd_ext
def addlevel(df, dct):
    return df.assign(**dct).prep().set_index(list(dct.keys()), append=True)

@pd_ext
def vc(df, by, dropna=False, digits=1, **kwargs):
    return df.groupby(by, dropna=dropna, observed=False, **kwargs).size().to_frame('ct').assign(pct=lambda x: (x['ct']/df.shape[0]*100).round(digits))

@pd_ext
def rindex(df, level=None, bare=False, **kwargs):
    level = level if level is None else intersection(level, df.index.names)
    df = df.reset_index(level, **kwargs)
    return df.reset_index(drop=True) if bare else df

@pd_ext
def sindex(df, level, **kwargs):
    return df.set_index(intersection(level, df.columns), **kwargs)

@pd_ext
def rsindex(df, level, **kwargs):
    return df.rindex(level, True).sindex(level, **kwargs)

@pd_ext
def grpby(df, by, **kwargs):
    return df.groupby(intersection(by, df.columns.union(df.index.names)), **kwargs)

@pd_ext
def missing(df, digits=1):
    return df.isnull().sum().sort_values(ascending=False).to_frame('ct').query('ct>0').assign(pct=lambda x: (x['ct']/df.shape[0]*100).round(digits))

@pd_ext
def impute(df, col, val=None, grp=None):
    val = val if val is not None else 'median' if pd.api.types.is_numeric_dtype(df[col]) else 'mode'
    match val:
        case 'median':
            func = lambda x: x.median()
        case 'mean' | 'ave' | 'avg' | 'average':
            func = lambda x: x.mean()
        case 'mode' | 'most_frequent':
            func = lambda x: x.mode()[0]
        case 'max':
            func = lambda x: x.max()
        case 'min':
            func = lambda x: x.min()
        case _:
            func = lambda x: val
    return (df if grp is None else df.groupby(grp))[col].transform(lambda x: x.fillna(func(x)))

##################### file helpers #####################
def getsizeof(x):
    if isinstance(x, dict):
        dct = {k: getsizeof(v) for k,v in x.items()}
        pd.Series(dct).rename('b').sort_values(ascending=False).disp(None)
    # elif hasattr(x, '__dict__'):
    #     getsizeof(x.__dict__)
    else:
        return sys.getsizeof(x)

def delete(path):
    path = pathlib.Path(path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.is_file():
        path.unlink()

def mkdir(path, overwrite=False):
    path = pathlib.Path(path)
    if overwrite:
        delete(path)
    path.mkdir(exist_ok=True, parents=True)

def write(path, obj, overwrite=False, **kwargs):
    path = pathlib.Path(path)
    if overwrite:
        delete(path)
    if not path.is_file():
        mkdir(path.parent)
        match path.suffix:
            case '.parq' | '.parquet':
                obj.to_parquet(path, **kwargs)
            case '.csv':
                obj.to_csv(path, **kwargs)
            case '.pkl':
                with open(path, 'wb') as f:
                    dill.dump(obj, f, **kwargs)
            case _:
                raise Exception("unknown sufffix", path.suffix)
    return obj

def read(path, overwrite=False, **kwargs):
    path = pathlib.Path(path)
    if overwrite:
        delete(path)
    try:
        match path.suffix:
            case '.parq' | '.parquet':
                return pd.read_parquet(path, **kwargs).prep()
            case '.csv':
                return pd.read_csv(path, **kwargs)
            case '.pkl':
                with open(path, 'rb') as f:
                    return dill.load(f, **kwargs)
            case _:
                raise Exception("unknown sufffix", path.suffix)
    except:
        return None

@dataclasses.dataclass
class MyBaseClass():
    overwrite: set = dataclasses.field(default_factory=set)
    dependence: dict = dataclasses.field(default_factory=dict)
    """Lets us access object attributes using self.attr or self['attr'] & easily save/read to file"""
    def __contains__(self, key):
        return key in self.__dict__
    def __getitem__(self, key):
        return self.__dict__[key]
    def __setitem__(self, key, val):
        self.__dict__[key] = val
    def __delitem__(self, key):
        if key in self:
            del self.__dict__[key]

    def __post_init__(self):
        if 'root_path' in self:
            self.root_path = pathlib.Path(self.root_path)
        self.overwrite = setify(self.overwrite)
        l = 0
        while l < len(self.overwrite):
            l = len(self.overwrite)
            self.overwrite |= {y for x in self.overwrite for y in setify(self.dependence[x] if x in self.dependence else {})}

    def load(self, path, overwrite=False, force=False):
        dct = read(path, overwrite)
        if dct is not None:
            if force:
                self.__dict__ = self.__dict__ | dct
            else:
                self.__dict__ = dct | self.__dict__
            return self

    def dump(self, path, overwrite=True):
        write(path, self.__dict__, overwrite)
        return self

    def get(self, func, fn, pre=[], drop=[], path=None):
        nm = fn.split('/')[0].split('.')[0]
        overwrite = nm in self.overwrite
        path = (self.root_path if path is None else path) / fn
        if nm in self:
            if overwrite:
                del self[nm]
        elif path.suffix == '.pkl':
            self.load(path, overwrite)
        else:
            self[nm] = read(path, overwrite)
        if nm not in self or self[nm] is None:
            for k in uniquify(pre):
                getattr(self, 'get_'+k)()
            with Timer():
                print('creating', fn, end=": ")
                self.path = path
                if func() != 'fail':
                    for k in uniquify(drop):
                        del self[k]
                    if path.suffix == '.pkl':
                        self.dump(path)
                    else:
                        write(path, self[nm])
        return self


@dataclasses.dataclass
class Oracle(MyBaseClass):
    database: str = 'WFOCUSP'
    timeout_default: int = 60000

    def execute(self, qry, show=False, timeout=None, **opts):
        qry = mkqry(qry)
        if show:
            print(qry)
        with warnings.catch_warnings(action='ignore'), oracledb.connect(user=os.environ.get(self.database+'_USER'), password=os.environ.get(self.database+'_PASS'), dsn=os.environ.get(self.database+'_DSN')) as connection:
            connection.call_timeout = self.timeout_default if timeout is None else timeout
            try:
                n_ses = pd.read_sql(f"select count(*) from sys.v_$session where username='{os.environ.get(self.database+'_USER')}'", connection).squeeze()
                print(f"{n_ses} active sessions")
            except:
                pass
            df = pd.read_sql(qry, connection).prep()
        return df
    
    def head(self, qry, rows=2, **opts):
        qry = mkqry(qry) + f"\nfetch first {rows} rows only"
        return self.execute(qry, **opts)

    def shape(self, qry, **opts):
        qry = f"select count(*) from {subqry(qry)}"
        m = self.execute(qry, **opts).squeeze()
        print(m)
        return m
    
    def value_counts(self, qry, val, **opts):
        sub = subqry(qry) if isqry(qry) else qry
        val = join(val)
        qry = f"select {val}, count(*) as ct from {subqry(qry)} group by {val} order by {val}"
        return self.execute(qry, **opts)

    def repeats(self, qry, val, **opts):
        sub = subqry(qry) if isqry(qry) else qry
        val = join(val)
        qry = f"select ct, count(*) as rep from (select count(*) as ct from {subqry(qry)} group by {val}) group by ct order by ct"
        return self.execute(qry, **opts)

    def dtypes(self, qry):
        dt = self.execute(SQL(qry).qry().head(10)).dtypes
        dt.disp()
        return dt