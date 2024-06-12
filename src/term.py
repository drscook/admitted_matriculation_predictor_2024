from flags import *

def dt(date, format='sql'):
    try:
        date = pd.to_datetime(date).date()
    except:
        return f"trunc({date})"
    format = format.lower()
    if format == 'sql':
        d = date.strftime('%d-%b-%y')
        return f"trunc(to_date('{d}'))"
    elif format == 'iso':
        return date.isoformat()
    else:
        return date.strftime(format)

def get_desc(nm, alias=None):
    tbl = 'stv'+nm if alias is None else 'stv'+alias
    return [f"A.{nm}_code", f"(select B.{tbl}_desc from {tbl} B where B.{tbl}_code = A.{nm}_code) as {nm}_desc"]

@dataclasses.dataclass
class Term(MyBaseClass):
    cycle_day: int = 0
    term_code: int = 202308
    show: set = dataclasses.field(default_factory=set)
    root_path: str = "/home/scook/institutional_data_analytics/admitted_matriculation_projection/resources"
    dependence: dict = dataclasses.field(default_factory=lambda: {'adm':'raw', 'flg':'raw', 'dst':'raw'})

    def get_trm(self):
        def func():
            qry = f"""
select
    A.stvterm_code as term_code,
    replace(A.stvterm_desc, ' ', '') as term_desc,
    A.stvterm_start_date as start_date,
    A.stvterm_end_date as end_date,
    A.stvterm_fa_proc_yr as fa_proc_yr,
    A.stvterm_housing_start_date as housing_start_date,
    A.stvterm_housing_end_date as housing_end_date,
    B.sobptrm_census_date as census_date
from stvterm A, sobptrm B
where A.stvterm_code = B.sobptrm_term_code and B.sobptrm_ptrm_code='1'"""
            self.trm = db.execute(qry, 'trm' in self.show)
        return self.get(func, "trm.parq")
    
    def get_dst(self):
        def func():
            raise Exception('go find distances parquet file')
        return self.get(func, "dst.parq")

    def __post_init__(self):
        super().__post_init__()
        self.get_trm()
        self.get_dst()
        self.year = self.term_code // 100
        self.appl_term_codes = [self.term_code, self.term_code-2] if self.term_code % 100 == 8 else [self.term_code]
        T = [self.trm.query("term_code==@t").squeeze() for t in self.appl_term_codes]
        self.appl_term_descs = [t['term_desc'] for t in T]
        self.term_desc = T[0]['term_desc']
        self.census_date = T[0]['census_date']
        delta = 7-(self.census_date.day_of_week-2)%7
        self.end_date = self.census_date + pd.Timedelta(days=delta) # Wednesday after census
        self.cycle_date = self.end_date - pd.Timedelta(days=self.cycle_day)
        # print(self.end_date.day_name(), self.end_date.date())
        # print(self.cycle_date.day_name(), self.cycle_date.date(), self.cycle_day)
        assert self.end_date.day_of_week == self.cycle_date.day_of_week == 2 and delta > 0 and delta <= 7
        self.stem = f"{rjust(self.cycle_day,3,0)}/{self.term_code}"
        self.flg_col = {
            'perm': [
                'id',
                'current_date',
                'styp_code',
                # 'transfer_hours',
                # 'inst_hours',
                # 'overall_hours',
                # 'inst_gpa',
                # 'overall_gpa',
                'fafsa_app',
                'finaid_offered',
                'finaid_accepted',
                # 'finaid_declined',
                'disb_req_complete',
                'schlship_app',
                # 'declined_schlrship',
                'math',
                'reading',
                'writing',
                'gap_score',
                ],
            'temp': [
                'app_date',
                'term_code',
                'ssb_last_accessed',
                'waiver_code',
                'ftic_gap_score',
                't_gap_score',
                'orien_sess',
                'orientation_hold_exists',
                'registered',
                'ver_complete',
                'selected_for_ver',
                'act_new_comp_score',
                'sat10_total_score',
                ],
        }

    def get_cycle_day(self, col='A.current_date'):
        return f"{dt(self.end_date)} - trunc({col})"

    def cutoff(self, col='A.current_date', criteria="= 0"):
        return f'{self.get_cycle_day(col)} {criteria}'


    def get_reg(self):
        def func():
            qry = f"""
select
    lower(B.ssbsect_subj_code) || B.ssbsect_crse_numb as crse_code,
    A.sfrstcr_term_code as term_code,
    A.sfrstcr_pidm as pidm,
    (select C.sgbstdn_levl_code from sgbstdn C where C.sgbstdn_pidm = A.sfrstcr_pidm and C.sgbstdn_term_code_eff <= A.sfrstcr_term_code order by C.sgbstdn_term_code_eff desc fetch first 1 rows only) as levl_code,
    (select C.sgbstdn_styp_code from sgbstdn C where C.sgbstdn_pidm = A.sfrstcr_pidm and C.sgbstdn_term_code_eff <= A.sfrstcr_term_code order by C.sgbstdn_term_code_eff desc fetch first 1 rows only) as styp_code,
    --sum(B.ssbsect_credit_hrs) as credit_hr
    sum(A.sfrstcr_credit_hr) as credit_hr
from sfrstcr A, ssbsect B
where
    A.sfrstcr_term_code = B.ssbsect_term_code
    and A.sfrstcr_crn = B.ssbsect_crn
    and A.sfrstcr_term_code = {self.term_code}
    and A.sfrstcr_ptrm_code not in ('28','R3')
    and  {self.get_cycle_day('A.sfrstcr_add_date')} >= {self.cycle_day}  -- added before cycle_day
    and ({self.get_cycle_day('A.sfrstcr_rsts_date')} < {self.cycle_day} or A.sfrstcr_rsts_code in ('DC','DL','RD','RE','RW','WD','WF')) -- dropped after cycle_day or still enrolled
    and B.ssbsect_subj_code <> 'INST'
group by B.ssbsect_subj_code, B.ssbsect_crse_numb, A.sfrstcr_term_code, A.sfrstcr_pidm"""

            qry = f"""
with A as {subqry(qry)}
select A.* from A
union all
select
    '_allcrse' as crse_code,
    A.term_code,
    A.pidm,
    A.levl_code,
    A.styp_code,
    sum(A.credit_hr) as credit_hr
from A
group by A.term_code, A.pidm, A.levl_code, A.styp_code
union all
select
    '_anycrse' as crse_code,
    A.term_code,
    A.pidm,
    A.levl_code,
    A.styp_code,
    case when sum(A.credit_hr) > 0 then 1 end as credit_hr
from A
group by A.term_code, A.pidm, A.levl_code, A.styp_code
"""
            self.reg = db.execute(qry, 'reg' in self.show)
        return self.get(func, fn=f"reg/{self.stem}.parq", subpath='data')


    def get_adm(self):
        def func():
            g = lambda s: indent(join(s,',\n'))+'\n'
            def f(term_desc):
                accept = "A.apst_code = 'D' and A.apdc_code in (select stvapdc_code from stvapdc where stvapdc_inst_acc_ind is not null)"
                sel = [
                    f"trunc(A.current_date) as cycle_date",
                    f"{self.get_cycle_day()} as cycle_day",
                    f"A.pidm",
                    f"A.id",
                    f"A.term_code as term_code_entry",
                    f"A.levl_code",
                    f"A.styp_code",
                    f"A.appl_no",
                    f"max({self.get_cycle_day()}) over (partition by A.pidm, A.appl_no) as appl_day",  # first date on snapshot table (saradap_appl_date has too many consistencies so this replaces it)
                    f"max(case when A.apst_code = 'D' and A.apdc_code in (select stvapdc_code from stvapdc where stvapdc_inst_acc_ind is not null) then {self.get_cycle_day()} end) over (partition by A.pidm, A.appl_no) as apdc_day",  # first date accepted
                    f"A.admt_code",
                    f"A.apst_code",
                    f"A.apdc_code",
                    f"A.camp_code",
                    f"A.saradap_resd_code as resd_code",
                    f"A.coll_code_1 as coll_code",
                    f"A.majr_code_1 as majr_code",
                    f"A.dept_code",
                    f"A.hs_percentile as hs_pctl",
                    # f"A.enrolled_ind",
                ]
                qry = f"select {g(sel)}from opeir.admissions_{term_desc} A"
                sel = [
                    f"A.*",
                    f"min(case when A.cycle_day >= {self.cycle_day} then A.cycle_day end) over (partition by A.pidm, A.appl_no) as before",
                    f"min(A.cycle_day) over (partition by A.pidm, A.appl_no) - min(A.cycle_day) over () as lag",
                ]
                qry = f"select {g(sel)}from {subqry(qry)} A where A.apdc_day >= A.cycle_day and {self.cycle_day} + 7 >= A.cycle_day and A.cycle_day >= 0"
                qry = f"select A.* from {subqry(qry)} A where A.cycle_day = A.before and A.lag <= 7"
                return qry
            qry = join([f(term_desc).strip() for term_desc in self.appl_term_descs], "\n\nunion all\n\n")
            
            stat_codes = join(['AL','AR','AZ','CA','CO','CT','DC','DE','FL','GA','IA','ID','IL','IN','KS','KY','LA','MA','MD','ME','MI','MN','MO','MS','MT','NC','ND','NE','NH','NJ','NM','NV','NY','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VA','VT','WA','WI','WV','WY'], "', '")
            def get_spraddr(nm):
                sel = "to_number(substr(B.spraddr_zip, 0, 5) default null on conversion error)" if nm == "zip" else "B.spraddr_"+nm
                return f"""
    (select B.{nm} from (
    select
        {sel} as {nm},
        B.spraddr_seqno as s,
        case
            when B.spraddr_atyp_code = 'PA' then 6
            when B.spraddr_atyp_code = 'PR' then 5
            when B.spraddr_atyp_code = 'MA' then 4
            when B.spraddr_atyp_code = 'BU' then 3
            when B.spraddr_atyp_code = 'BI' then 2
            --when B.spraddr_atyp_code = 'P1' then 1
            --when B.spraddr_atyp_code = 'P2' then 0
            end as r
    from spraddr B where B.spraddr_pidm = A.pidm and B.spraddr_stat_code in ('{stat_codes}')
    ) B where B.{nm} is not null and B.r is not null order by B.r desc, B.s desc fetch first 1 row only) as {nm}""".strip()

            sel = [
                f"A.*",
                f"row_number() over (partition by A.pidm order by A.appl_no desc) as r",
                f"{self.term_code} as term_code",
                get_spraddr("cnty_code"),
                get_spraddr("stat_code"),
                get_spraddr("zip"),
                f"(select B.gorvisa_natn_code_issue from gorvisa B where B.gorvisa_pidm = A.pidm order by gorvisa_seq_no desc fetch first 1 row only) as natn_code",
                f"(select B.spbpers_lgcy_code from spbpers B where B.spbpers_pidm = A.pidm) as lgcy_code",
                f"(select B.spbpers_birth_date from spbpers B where B.spbpers_pidm = A.pidm) as birth_date",
            ]
            qry = f"select {g(sel)}from {subqry(qry)} A"
            
            sel = [
                f"A.cycle_date",
                f"A.cycle_day",
                f"A.pidm",
                f"A.id",
                # f"A.appl_date",
                f"A.appl_day",
                # f"A.apdc_date",
                f"A.apdc_day",
                # f"A.birth_date",
                f"{self.get_cycle_day('birth_date')} as birth_day",
                # f"{dt(self.end_date)} as end_date",
                f"A.term_code_entry",
                *get_desc('term'),
                f"A.appl_no",
                *get_desc('levl'),
                *get_desc('styp'),
                *get_desc('admt'),
                *get_desc('apst'),
                *get_desc('apdc'),
                *get_desc('camp'),
                *get_desc('coll'),
                *get_desc('dept'),
                *get_desc('majr'),
                *get_desc('cnty'),
                *get_desc('stat'),
                f"A.zip",
                f"A.natn_code",
                f"(select B.stvnatn_nation from stvnatn B where B.stvnatn_code = A.natn_code) as natn_desc",
                f"coalesce((select distinct 1 from gorvisa B where B.gorvisa_pidm = A.pidm and B.gorvisa_vtyp_code is not null), 0) as international",
                f"coalesce((select distinct 1 from gorprac B where B.gorprac_pidm = A.pidm and B.gorprac_race_cde='IN'), 0) as race_american_indian",
                f"coalesce((select distinct 1 from gorprac B where B.gorprac_pidm = A.pidm and B.gorprac_race_cde='AS'), 0) as race_asian",
                f"coalesce((select distinct 1 from gorprac B where B.gorprac_pidm = A.pidm and B.gorprac_race_cde='BL'), 0) as race_black",
                f"coalesce((select distinct 1 from gorprac B where B.gorprac_pidm = A.pidm and B.gorprac_race_cde='HA'), 0) as race_pacific",
                f"coalesce((select distinct 1 from gorprac B where B.gorprac_pidm = A.pidm and B.gorprac_race_cde='WH'), 0) as race_white",
                f"coalesce((select distinct 1 from spbpers B where B.spbpers_pidm = A.pidm and B.spbpers_ethn_cde=2   ), 0) as race_hispanic",
                f"(select B.spbpers_sex from spbpers B where B.spbpers_pidm = A.pidm) as gender",
                *get_desc('lgcy'),
                *get_desc('resd'),
                f"A.hs_pctl",
                # f"A.enrolled_ind",
            ]
            qry = f"select {g(sel)}from {subqry(qry)} A where A.r = 1"
            self.adm = db.execute(qry, 'adm' in self.show)
        return self.get(func, fn=f"adm/{self.stem}.parq", subpath='data')


    def get_flg(self):
        def func():
            F = []
            for term_code in self.appl_term_codes:
                raw = Flags().path['parq'] / f"flg_{term_code}.parq"
                df = read(raw, columns=['current_date'])
                df['cycle_day'] = (self.end_date - df['current_date']).dt.days
                flg_day  = df.query(f'cycle_day>={self.cycle_day}')['cycle_day'].min()
                flg_date = df.query(f'cycle_day==@flg_day')['current_date'].min()
                filters = [('current_date','==',flg_date)]
                L = []
                missing = []
                if self.flg_col is None:
                    df = read(raw, filters=filters)
                else:
                    for c in sum(self.flg_col.values(),[]):
                        x = read(raw, filters=filters, columns=[c])
                        if x is None:
                            missing.append(c)
                            x = L[0][[]].assign(**{c:pd.NA})
                        L.append(x)
                    df = pd.concat(L, axis=1)
                print(f'{term_code} flags cycle day {flg_day} >= {self.cycle_day} on {flg_date}')
                # print(f'{term_code} flags cycle day {flg_day} >= {self.cycle_day} on {flg_date} missing columns: {missing}')
                F.append(df)
            with warnings.catch_warnings(action='ignore'):
                subset = ['id','term_code_entry','styp_code']
                df = (
                    pd.concat(F, ignore_index=True)
                    .prep()
                    .rename(columns={'current_date':'flg_date', 'term_code':'term_code_entry'})
                    .sort_values(by=[*subset,'app_date'])
                    .drop_duplicates(subset=subset, keep='last')
                    .copy()
                )
            df['gap_score'] = np.where(df['styp_code']=='n', df['ftic_gap_score'].combine_first(df['t_gap_score']).combine_first(df['gap_score']), df['t_gap_score'].combine_first(df['ftic_gap_score']).combine_first(df['gap_score']))
            df['ssb'] = df['ssb_last_accessed'].notnull()
            df['waiver'] = df['waiver_code'].notnull()
            df['oriented'] = np.where(df['orien_sess'].notnull() | df['registered'].notnull(), 'y', np.where(df['orientation_hold_exists'].notnull(), 'n', 'w'))
            df['verified'] = np.where(df['ver_complete'].notnull(), 'y', np.where(df['selected_for_ver'].notnull(), 'n', 'w'))
            df['sat10_total_score'] = (36-9) / (1600-590) * (df['sat10_total_score']-590) + 9
            df['act_equiv'] = df[['act_new_comp_score','sat10_total_score']].max(axis=1)
            for k in ['reading', 'writing', 'math']:
                df[k] = ~df[k].isin(['not college ready', 'retest required', pd.NA])
            self.flg = df.drop(columns=self.flg_col['temp']+['cycle_day'], errors='ignore').dropna(axis=1, how='all')
        return self.get(func, fn=f"flg/{self.stem}.parq", subpath='data')


    def get_raw(self):
        def func():
            df =  (
                self.adm
                .merge(self.flg, how='left', on=['id','term_code_entry','styp_code'])
                .merge(self.dst, how='left', on=['zip','camp_code'])
                .prep()
            )
            assert (df.groupby(['pidm','term_code']).size() == 1).all()
            self.raw = df
        return self.get(func, fn=f"raw/{self.stem}.parq", subpath='data', pre=["dst","adm","flg"])