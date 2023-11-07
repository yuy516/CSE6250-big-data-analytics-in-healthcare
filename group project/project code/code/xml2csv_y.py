import numpy as np
import pandas as pd
from xml.etree import ElementTree

# convert xml file to csv file
def convert_csv(inpath, outpath):
    # parse
    xml = ElementTree.parse(inpath)
    root = xml.getroot()

    # create dataframe for each disease
    for disease in root.find('diseases').findall('disease'):
        if disease.get('name') == 'Asthma':
            df_Asthma = get_disease_df(disease)
        if disease.get('name') == 'CAD':
            df_CAD = get_disease_df(disease)
        if disease.get('name') == 'CHF':
            df_CHF = get_disease_df(disease)
        if disease.get('name') == 'Depression':
            df_Depression = get_disease_df(disease)
        if disease.get('name') == 'Diabetes':
            df_Diabetes = get_disease_df(disease)
        if disease.get('name') == 'Gallstones':
            df_Gallstones = get_disease_df(disease)
        if disease.get('name') == 'GERD':
            df_GERD = get_disease_df(disease)
        if disease.get('name') == 'Gout':
            df_Gout = get_disease_df(disease)
        if disease.get('name') == 'Hypercholesterolemia':
            df_HC = get_disease_df(disease)
        if disease.get('name') == 'Hypertension':
            df_Hypertension = get_disease_df(disease)
        if disease.get('name') == 'Hypertriglyceridemia':
            df_HT = get_disease_df(disease)
        if disease.get('name') == 'OA':
            df_OA = get_disease_df(disease)
        if disease.get('name') == 'Obesity':
            df_Obesity = get_disease_df(disease)
        if disease.get('name') == 'OSA':
            df_OSA = get_disease_df(disease)
        if disease.get('name') == 'PVD':
            df_PVD = get_disease_df(disease)
        if disease.get('name') == 'Venous Insufficiency':
            df_VI = get_disease_df(disease)

    # create csv for output label
    df = df_Asthma.merge(df_CAD, on='id', how='outer').rename(columns = {'judgment_x':'Asthma', 'judgment_y':'CAD'})
    df = df.merge(df_CHF, on='id', how='outer').rename(columns = {'judgment':'CHF'})
    df = df.merge(df_Depression, on='id', how='outer').rename(columns = {'judgment':'Depression'})
    df = df.merge(df_Diabetes, on='id', how='outer').rename(columns={'judgment': 'Diabetes'})
    df = df.merge(df_Gallstones, on='id', how='outer').rename(columns={'judgment': 'Gallstones'})
    df = df.merge(df_GERD, on='id', how='outer').rename(columns={'judgment': 'GERD'})
    df = df.merge(df_Gout, on='id', how='outer').rename(columns={'judgment': 'Gout'})
    df = df.merge(df_HC, on='id', how='outer').rename(columns={'judgment': 'HC'})
    df = df.merge(df_Hypertension, on='id', how='outer').rename(columns={'judgment': 'Hypertension'})
    df = df.merge(df_HT, on='id', how='outer').rename(columns={'judgment': 'HT'})
    df = df.merge(df_OA, on='id', how='outer').rename(columns={'judgment': 'OA'})
    df = df.merge(df_Obesity, on='id', how='outer').rename(columns={'judgment': 'Obesity'})
    df = df.merge(df_OSA, on='id', how='outer').rename(columns={'judgment': 'OSA'})
    df = df.merge(df_PVD, on='id', how='outer').rename(columns={'judgment': 'PVD'})
    df = df.merge(df_VI, on='id', how='outer').rename(columns={'judgment': 'VI'})

    df.to_csv(outpath)

def get_disease_df(disease):
    rows = []
    for doc in disease:
        id = doc.get('id')
        judgment = doc.get('judgment')
        rows.append({'id': id, 'judgment': judgment})
    df = pd.DataFrame(rows, columns=["id", 'judgment'])
    return df

def save_output_label(inpath, outpath):
    df_train_texture = pd.read_csv(inpath + 'Training_Textual_Judgment.csv')
    df_train_texture = df_train_texture.drop(columns=df_train_texture.columns[0])

    df_train_intuitive = pd.read_csv(inpath + 'Training_Intuitive_Judgment.csv')
    df_train_intuitive = df_train_intuitive.drop(columns=df_train_intuitive.columns[0])

    df_test_texture = pd.read_csv(inpath + 'Test_Textual_Judgment.csv')
    df_test_texture = df_test_texture.drop(columns=df_test_texture.columns[0])

    df_test_intuitive = pd.read_csv(inpath + 'Test_Intuitive_Judgment.csv')
    df_test_intuitive = df_test_intuitive.drop(columns=df_test_intuitive.columns[0])

    df_train = df_train_texture.merge(df_train_intuitive, on='id', how='outer')
    df_train['Asthma'] = df_train.apply(lambda x: 1 if x.Asthma_x == 'Y' or x.Asthma_y == 'Y' else 0 if x.Asthma_x == 'N' or x.Asthma_y == 'N' else np.nan, axis=1)
    df_train['CAD'] = df_train.apply(lambda x: 1 if x.CAD_x == 'Y' or x.CAD_y == 'Y' else 0 if x.CAD_x == 'N' or x.CAD_y == 'N' else np.nan, axis=1)
    df_train['CHF'] = df_train.apply(lambda x: 1 if x.CHF_x == 'Y' or x.CHF_y == 'Y' else 0 if x.CHF_x == 'N' or x.CHF_y == 'N' else np.nan, axis=1)
    df_train['Depression'] = df_train.apply(lambda x: 1 if x.Depression_x == 'Y' or x.Depression_y == 'Y' else 0 if x.Depression_x == 'N' or x.Depression_y == 'N' else np.nan, axis=1)
    df_train['Diabetes'] = df_train.apply(lambda x: 1 if x.Diabetes_x == 'Y' or x.Diabetes_y == 'Y' else 0 if x.Diabetes_x == 'N' or x.Diabetes_y == 'N' else np.nan, axis=1)
    df_train['Gallstones'] = df_train.apply(lambda x: 1 if x.Gallstones_x == 'Y' or x.Gallstones_y == 'Y' else 0 if x.Gallstones_x == 'N' or x.Gallstones_y == 'N' else np.nan, axis=1)
    df_train['GERD'] = df_train.apply(lambda x: 1 if x.GERD_x == 'Y' or x.GERD_y == 'Y' else 0 if x.GERD_x == 'N' or x.GERD_y == 'N' else np.nan, axis=1)
    df_train['Gout'] = df_train.apply(lambda x: 1 if x.Gout_x == 'Y' or x.Gout_y == 'Y' else 0 if x.Gout_x == 'N' or x.Gout_y == 'N' else np.nan, axis=1)
    df_train['HC'] = df_train.apply(lambda x: 1 if x.HC_x == 'Y' or x.HC_y == 'Y' else 0 if x.HC_x == 'N' or x.HC_y == 'N' else np.nan, axis=1)
    df_train['Hypertension'] = df_train.apply(lambda x: 1 if x.Hypertension_x == 'Y' or x.Hypertension_y == 'Y' else 0 if x.Hypertension_x == 'N' or x.Hypertension_y == 'N' else np.nan, axis=1)
    df_train['HT'] = df_train.apply(lambda x: 1 if x.HT_x == 'Y' or x.HT_y == 'Y' else 0 if x.HT_x == 'N' or x.HT_y == 'N' else np.nan, axis=1)
    df_train['OA'] = df_train.apply(lambda x: 1 if x.OA_x == 'Y' or x.OA_y == 'Y' else 0 if x.OA_x == 'N' or x.OA_y == 'N' else np.nan, axis=1)
    df_train['Obesity'] = df_train.apply(lambda x: 1 if x.Obesity_x == 'Y' or x.Obesity_y == 'Y' else 0 if x.Obesity_x == 'N' or x.Obesity_y == 'N' else np.nan, axis=1)
    df_train['OSA'] = df_train.apply(lambda x: 1 if x.OSA_x == 'Y' or x.OSA_y == 'Y' else 0 if x.OSA_x == 'N' or x.OSA_y == 'N' else np.nan, axis=1)
    df_train['PVD'] = df_train.apply(lambda x: 1 if x.PVD_x == 'Y' or x.PVD_y == 'Y' else 0 if x.PVD_x == 'N' or x.PVD_y == 'N' else np.nan, axis=1)
    df_train['VI'] = df_train.apply(lambda x: 1 if x.VI_x == 'Y' or x.VI_y == 'Y' else 0 if x.VI_x == 'N' or x.VI_y == 'N' else np.nan, axis=1)

    train_Asthma = df_train[['id','Asthma']].dropna()
    train_Asthma.to_csv(outpath+'Asthma/y_train.csv')
    train_CAD = df_train[['id', 'CAD']].dropna()
    train_CAD.to_csv(outpath + 'CAD/y_train.csv')
    train_CHF = df_train[['id', 'CHF']].dropna()
    train_CHF.to_csv(outpath + 'CHF/y_train.csv')
    train_Depression = df_train[['id', 'Depression']].dropna()
    train_Depression.to_csv(outpath + 'Depression/y_train.csv')
    train_Diabetes = df_train[['id', 'Diabetes']].dropna()
    train_Diabetes.to_csv(outpath + 'Diabetes/y_train.csv')
    train_Gallstones = df_train[['id', 'Gallstones']].dropna()
    train_Gallstones.to_csv(outpath + 'Gallstones/y_train.csv')
    train_GERD = df_train[['id', 'GERD']].dropna()
    train_GERD.to_csv(outpath + 'GERD/y_train.csv')
    train_Gout = df_train[['id', 'Gout']].dropna()
    train_Gout.to_csv(outpath + 'Gout/y_train.csv')
    train_HC = df_train[['id', 'HC']].dropna()
    train_HC.to_csv(outpath + 'HC/y_train.csv')
    train_Hypertension = df_train[['id', 'Hypertension']].dropna()
    train_Hypertension.to_csv(outpath + 'Hypertension/y_train.csv')
    train_HT = df_train[['id', 'HT']].dropna()
    train_HT.to_csv(outpath + 'HT/y_train.csv')
    train_OA = df_train[['id', 'OA']].dropna()
    train_OA.to_csv(outpath + 'OA/y_train.csv')
    train_Obesity = df_train[['id', 'Obesity']].dropna()
    train_Obesity.to_csv(outpath + 'Obesity/y_train.csv')
    train_OSA = df_train[['id', 'OSA']].dropna()
    train_OSA.to_csv(outpath + 'OSA/y_train.csv')
    train_PVD = df_train[['id', 'PVD']].dropna()
    train_PVD.to_csv(outpath + 'PVD/y_train.csv')
    train_VI = df_train[['id', 'VI']].dropna()
    train_VI.to_csv(outpath + 'VI/y_train.csv')

    df_test = df_test_texture.merge(df_test_intuitive, on='id', how='outer')
    df_test['Asthma'] = df_train.apply(lambda x: 1 if x.Asthma_x == 'Y' or x.Asthma_y == 'Y' else 0 if x.Asthma_x == 'N' or x.Asthma_y == 'N' else np.nan, axis=1)
    df_test['CAD'] = df_train.apply(lambda x: 1 if x.CAD_x == 'Y' or x.CAD_y == 'Y' else 0 if x.CAD_x == 'N' or x.CAD_y == 'N' else np.nan, axis=1)
    df_test['CHF'] = df_train.apply(lambda x: 1 if x.CHF_x == 'Y' or x.CHF_y == 'Y' else 0 if x.CHF_x == 'N' or x.CHF_y == 'N' else np.nan, axis=1)
    df_test['Depression'] = df_train.apply(lambda x: 1 if x.Depression_x == 'Y' or x.Depression_y == 'Y' else 0 if x.Depression_x == 'N' or x.Depression_y == 'N' else np.nan,axis=1)
    df_test['Diabetes'] = df_train.apply(lambda x: 1 if x.Diabetes_x == 'Y' or x.Diabetes_y == 'Y' else 0 if x.Diabetes_x == 'N' or x.Diabetes_y == 'N' else np.nan, axis=1)
    df_test['Gallstones'] = df_train.apply(lambda x: 1 if x.Gallstones_x == 'Y' or x.Gallstones_y == 'Y' else 0 if x.Gallstones_x == 'N' or x.Gallstones_y == 'N' else np.nan, axis=1)
    df_test['GERD'] = df_train.apply(lambda x: 1 if x.GERD_x == 'Y' or x.GERD_y == 'Y' else 0 if x.GERD_x == 'N' or x.GERD_y == 'N' else np.nan, axis=1)
    df_test['Gout'] = df_train.apply(lambda x: 1 if x.Gout_x == 'Y' or x.Gout_y == 'Y' else 0 if x.Gout_x == 'N' or x.Gout_y == 'N' else np.nan, axis=1)
    df_test['HC'] = df_train.apply(lambda x: 1 if x.HC_x == 'Y' or x.HC_y == 'Y' else 0 if x.HC_x == 'N' or x.HC_y == 'N' else np.nan, axis=1)
    df_test['Hypertension'] = df_train.apply(lambda x: 1 if x.Hypertension_x == 'Y' or x.Hypertension_y == 'Y' else 0 if x.Hypertension_x == 'N' or x.Hypertension_y == 'N' else np.nan, axis=1)
    df_test['HT'] = df_train.apply(lambda x: 1 if x.HT_x == 'Y' or x.HT_y == 'Y' else 0 if x.HT_x == 'N' or x.HT_y == 'N' else np.nan, axis=1)
    df_test['OA'] = df_train.apply(lambda x: 1 if x.OA_x == 'Y' or x.OA_y == 'Y' else 0 if x.OA_x == 'N' or x.OA_y == 'N' else np.nan, axis=1)
    df_test['Obesity'] = df_train.apply(lambda x: 1 if x.Obesity_x == 'Y' or x.Obesity_y == 'Y' else 0 if x.Obesity_x == 'N' or x.Obesity_y == 'N' else np.nan, axis=1)
    df_test['OSA'] = df_train.apply(lambda x: 1 if x.OSA_x == 'Y' or x.OSA_y == 'Y' else 0 if x.OSA_x == 'N' or x.OSA_y == 'N' else np.nan, axis=1)
    df_test['PVD'] = df_train.apply(lambda x: 1 if x.PVD_x == 'Y' or x.PVD_y == 'Y' else 0 if x.PVD_x == 'N' or x.PVD_y == 'N' else np.nan, axis=1)
    df_test['VI'] = df_train.apply(lambda x: 1 if x.VI_x == 'Y' or x.VI_y == 'Y' else 0 if x.VI_x == 'N' or x.VI_y == 'N' else np.nan, axis=1)

    test_Asthma = df_test[['id', 'Asthma']].dropna()
    test_Asthma.to_csv(outpath + 'Asthma/y_test.csv')
    test_CAD = df_test[['id', 'CAD']].dropna()
    test_CAD.to_csv(outpath + 'CAD/y_test.csv')
    test_CHF = df_test[['id', 'CHF']].dropna()
    test_CHF.to_csv(outpath + 'CHF/y_test.csv')
    test_Depression = df_test[['id', 'Depression']].dropna()
    test_Depression.to_csv(outpath + 'Depression/y_test.csv')
    test_Diabetes = df_test[['id', 'Diabetes']].dropna()
    test_Diabetes.to_csv(outpath + 'Diabetes/y_test.csv')
    test_Gallstones = df_test[['id', 'Gallstones']].dropna()
    test_Gallstones.to_csv(outpath + 'Gallstones/y_test.csv')
    test_GERD = df_test[['id', 'GERD']].dropna()
    test_GERD.to_csv(outpath + 'GERD/y_test.csv')
    test_Gout = df_test[['id', 'Gout']].dropna()
    test_Gout.to_csv(outpath + 'Gout/y_test.csv')
    test_HC = df_test[['id', 'HC']].dropna()
    test_HC.to_csv(outpath + 'HC/y_test.csv')
    test_Hypertension = df_test[['id', 'Hypertension']].dropna()
    test_Hypertension.to_csv(outpath + 'Hypertension/y_test.csv')
    test_HT = df_test[['id', 'HT']].dropna()
    test_HT.to_csv(outpath + 'HT/y_test.csv')
    test_OA = df_test[['id', 'OA']].dropna()
    test_OA.to_csv(outpath + 'OA/y_test.csv')
    test_Obesity = df_test[['id', 'Obesity']].dropna()
    test_Obesity.to_csv(outpath + 'Obesity/y_test.csv')
    test_OSA = df_test[['id', 'OSA']].dropna()
    test_OSA.to_csv(outpath + 'OSA/y_test.csv')
    test_PVD = df_test[['id', 'PVD']].dropna()
    test_PVD.to_csv(outpath + 'PVD/y_test.csv')
    test_VI = df_test[['id', 'VI']].dropna()
    test_VI.to_csv(outpath + 'VI/y_test.csv')

def main():
    inpath = '../data/source/Training_Textual_Judgment.xml'
    outpath = '../data/processed/Training_Textual_Judgment.csv'
    convert_csv(inpath, outpath)

    inpath = '../data/source/Training_Intuitive_Judgment.xml'
    outpath = '../data/processed/Training_Intuitive_Judgment.csv'
    convert_csv(inpath, outpath)

    inpath = '../data/source/Test_Textual_Judgment.xml'
    outpath = '../data/processed/Test_Textual_Judgment.csv'
    convert_csv(inpath, outpath)

    inpath = '../data/source/Test_Intuitive_Judgment.xml'
    outpath = '../data/processed/Test_Intuitive_Judgment.csv'
    convert_csv(inpath, outpath)

    filepath = '../data/processed/'
    outpath = '../data/processed/'
    save_output_label(filepath, outpath)

if __name__ == "__main__":
    main()