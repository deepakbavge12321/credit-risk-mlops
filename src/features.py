# src/features.py
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils import reduce_mem


logger = logging.getLogger("ml_pipeline")

def application_features(df):
    df = df.copy()
    # Basic ratios
    df['CREDIT_INCOME_RATIO']      = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['ANNUITY_INCOME_RATIO']     = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['CREDIT_ANNUITY_RATIO']     = df['AMT_CREDIT'] / (df['AMT_ANNUITY'] + 1)
    df['CREDIT_GOODS_RATIO']       = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1)
    df['GOODS_INCOME_RATIO']       = df['AMT_GOODS_PRICE'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['INCOME_PER_PERSON']        = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
    df['INCOME_PER_CHILD']         = df['AMT_INCOME_TOTAL'] / (df['CNT_CHILDREN'] + 1)
    # Age/employment
    df['EMPLOYED_TO_BIRTH_RATIO']  = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + 1)
    df['DAYS_EMPLOYED_PERC']       = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + 1)
    df['CAR_TO_BIRTH_RATIO']       = df['OWN_CAR_AGE'] / ((-df['DAYS_BIRTH'] / 365) + 1)
    df['PHONE_TO_BIRTH_RATIO']     = df['DAYS_LAST_PHONE_CHANGE'] / (df['DAYS_BIRTH'] + 1)
    df['PHONE_TO_EMPLOY_RATIO']    = df['DAYS_LAST_PHONE_CHANGE'] / (df['DAYS_EMPLOYED'] + 1)
    # External sources (strongest predictors)
    df['EXT_SOURCE_MEAN']          = df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].mean(axis=1)
    df['EXT_SOURCE_STD']           = df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].std(axis=1)
    df['EXT_SOURCE_PROD']          = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_MIN']           = df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].min(axis=1)
    df['EXT_SOURCE_MAX']           = df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].max(axis=1)
    df['EXT12']                    = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    df['EXT13']                    = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']
    df['EXT23']                    = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    # Document flags: count how many docs submitted
    doc_cols = [c for c in df.columns if 'FLAG_DOCUMENT' in c]
    df['DOCS_COUNT']               = df[doc_cols].sum(axis=1)
    # Flag: anomalous employment (365243 = not employed)
    df['EMPLOYED_IS_ANOMALY']      = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['DAYS_EMPLOYED']            = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    # Credit term
    df['CREDIT_TERM']              = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1)
    # Social circle anomaly
    df['OBS_DEF_RATIO_30']         = df['DEF_30_CNT_SOCIAL_CIRCLE'] / (df['OBS_30_CNT_SOCIAL_CIRCLE'] + 1)
    df['OBS_DEF_RATIO_60']         = df['DEF_60_CNT_SOCIAL_CIRCLE'] / (df['OBS_60_CNT_SOCIAL_CIRCLE'] + 1)
    return df



def build_bureau_features(bureau_path):

    bureau = pd.read_csv(bureau_path)
    logger.info(f"bureau loaded: {bureau.shape}")

    # Feature engineering
    bureau['IS_ACTIVE'] = (bureau['CREDIT_ACTIVE'] == 'Active').astype(int)
    bureau['IS_CLOSED'] = (bureau['CREDIT_ACTIVE'] == 'Closed').astype(int)
    bureau['IS_BAD'] = (bureau['CREDIT_ACTIVE'] == 'Bad debt').astype(int)

    bureau['OVERDUE_RATIO'] = bureau['AMT_CREDIT_SUM_OVERDUE'] / (bureau['AMT_CREDIT_SUM'] + 1)
    bureau['DEBT_RATIO'] = bureau['AMT_CREDIT_SUM_DEBT'] / (bureau['AMT_CREDIT_SUM'] + 1)

    bureau['CREDIT_ENDDATE_DIFF'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_CREDIT']

    # Aggregations
    agg = bureau.groupby('SK_ID_CURR').agg({
        'SK_ID_BUREAU': 'count',
        'CREDIT_TYPE': 'nunique',

        'IS_ACTIVE': ['sum','mean'],
        'IS_CLOSED': ['sum','mean'],
        'IS_BAD': ['sum','mean'],

        'AMT_CREDIT_SUM_DEBT': ['sum','mean','max'],
        'AMT_CREDIT_SUM': ['sum','mean','max'],
        'AMT_CREDIT_SUM_OVERDUE': ['sum','max'],
        'AMT_CREDIT_SUM_LIMIT': ['mean','max'],

        'DAYS_CREDIT': ['mean','min','max','std'],
        'DAYS_CREDIT_ENDDATE': ['mean','max'],
        'DAYS_CREDIT_UPDATE': 'mean',

        'AMT_ANNUITY': ['mean','sum','max'],
        'CNT_CREDIT_PROLONG': 'sum',

        'OVERDUE_RATIO': 'mean',
        'DEBT_RATIO': 'mean',
        'CREDIT_ENDDATE_DIFF': 'mean',
    })

    # Flatten columns
    agg.columns = ['BUREAU_' + '_'.join(col).upper() for col in agg.columns]
    agg = agg.reset_index()

    # Additional ratios
    agg['BUREAU_DEBT_CREDIT_RATIO'] = (
        agg['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] /
        (agg['BUREAU_AMT_CREDIT_SUM_SUM'] + 1)
    )

    agg['BUREAU_ACTIVE_RATIO'] = (
        agg['BUREAU_IS_ACTIVE_SUM'] /
        (agg['BUREAU_SK_ID_BUREAU_COUNT'] + 1)
    )

    agg['BUREAU_OVERDUE_CREDIT_RATIO'] = (
        agg['BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM'] /
        (agg['BUREAU_AMT_CREDIT_SUM_SUM'] + 1)
    )

    return agg


def build_bureau_balance_features(bb_path, bureau_path):

    bb = pd.read_csv(bb_path)
    logger.info(f"bureau_balance loaded: {bb.shape}")

    # One-hot encoding
    bb_dummies = pd.get_dummies(bb['STATUS'], prefix='BB_STATUS')

    bb = pd.concat(
        [bb[['SK_ID_BUREAU', 'MONTHS_BALANCE']], bb_dummies],
        axis=1
    )

    status_cols = [c for c in bb.columns if c.startswith('BB_STATUS_')]

    # Aggregation per bureau ID
    agg_dict = {'MONTHS_BALANCE': ['count', 'min', 'max']}
    for c in status_cols:
        agg_dict[c] = 'mean'

    bb_bureau = bb.groupby('SK_ID_BUREAU').agg(agg_dict)

    bb_bureau.columns = (
        ['BB_MONTHS_COUNT', 'BB_MONTHS_MIN', 'BB_MONTHS_MAX'] +
        [c + '_MEAN' for c in status_cols]
    )

    bb_bureau = bb_bureau.reset_index()

    # Map to SK_ID_CURR
    bureau_key = pd.read_csv(
        bureau_path,
        usecols=['SK_ID_BUREAU', 'SK_ID_CURR']
    )

    bb_bureau = bb_bureau.merge(
        bureau_key,
        on='SK_ID_BUREAU',
        how='left'
    )

    # Aggregate to client level
    mean_cols = [c + '_MEAN' for c in status_cols]

    client_dict = {
        'BB_MONTHS_COUNT': ['mean', 'sum'],
        'BB_MONTHS_MIN': 'min',
        'BB_MONTHS_MAX': 'max'
    }

    for c in mean_cols:
        client_dict[c] = ['mean', 'max']

    bb_client = (
        bb_bureau
        .drop('SK_ID_BUREAU', axis=1)
        .groupby('SK_ID_CURR')
        .agg(client_dict)
    )

    bb_client.columns = [
        'BB_CLIENT_' + '_'.join(col).upper()
        for col in bb_client.columns
    ]

    bb_client = bb_client.reset_index()

    return bb_client


def main():
    import yaml

    logger.info("Running feature engineering stage …")

    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Load raw application data
    df = pd.read_csv(config["data"]["main"])

    # Application features
    df = application_features(df)

    # Bureau features
    bureau = build_bureau_features(config["data"]["bureau"])
    bb = build_bureau_balance_features(
        config["data"]["bb"],
        config["data"]["bureau"]
    )

    # Merge
    df = df.merge(bureau, on="SK_ID_CURR", how="left")
    df = df.merge(bb, on="SK_ID_CURR", how="left")

    logger.info(f"Shape after merges: {df.shape}")

    # Encode categoricals
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("Unknown")
        le.fit(df[col])
        df[col] = le.transform(df[col])

    # Fill missing
    df.fillna(0, inplace=True)

    # Reduce memory
    df = reduce_mem(df)

    # Save output for next stage
    df.to_csv("data/features.csv", index=False)

    logger.info("Saved features.csv")


if __name__ == "__main__":
    main()