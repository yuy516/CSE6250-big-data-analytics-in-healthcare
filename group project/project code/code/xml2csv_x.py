import numpy as np
import pandas as pd
from xml.etree import ElementTree

def convert_csv(inpath, outpath):
    # parse
    xml = ElementTree.parse(inpath)
    root = xml.getroot()

    # create dataframe
    rows = []
    columns = ['id', 'text']

    for doc in root.find('docs'):
        id = doc.get('id')
        text = doc.find('text').text
        rows.append({'id': id, 'text': text})

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(outpath)

def save_input_text(inpath, outpath, category):
    df_train_records = pd.read_csv(inpath + 'Training_Obesity_Records.csv')
    df_train_records = df_train_records.drop(columns=df_train_records.columns[0])
    y_train = pd.read_csv(inpath + category + '/y_train.csv')
    y_train = y_train.drop(columns=y_train.columns[0])
    df_train = df_train_records.merge(y_train, on='id', how='left').rename(columns={category: 'label'}).dropna()
    print('Size of train data for' + category + ":", len(df_train))
    df_train.to_csv(outpath + category + '/' + category + '_train.csv')

    df_test_records = pd.read_csv(inpath + 'Test_Obesity_Records.csv')
    df_test_records = df_test_records.drop(columns=df_test_records.columns[0])
    y_test = pd.read_csv(inpath + category + '/y_test.csv')
    y_test = y_test.drop(columns=y_test.columns[0])
    df_test = df_test_records.merge(y_test, on='id', how='left').rename(columns={category: 'label'}).dropna()
    print('Size of test data for' + category + ":", len(df_test))
    df_test.to_csv(outpath + category + '/' + category + '_test.csv')

    df_all = pd.concat([df_train, df_test], ignore_index=True)
    df_all.to_csv(outpath + category + '/' + category + '.csv')

def main():
    inpath = '../data/source/Training_Obesity_Records.xml'
    outpath = '../data/processed/Training_Obesity_Records.csv'
    convert_csv(inpath, outpath)

    inpath = '../data/source/Test_Obesity_Records.xml'
    outpath = '../data/processed/Test_Obesity_Records.csv'
    convert_csv(inpath, outpath)

    inpath = '../data/processed/'
    outpath = '../data/processed/'
    save_input_text(inpath, outpath, 'Asthma')
    save_input_text(inpath, outpath, 'CAD')
    save_input_text(inpath, outpath, 'CHF')
    save_input_text(inpath, outpath, 'Depression')
    save_input_text(inpath, outpath, 'Diabetes')
    save_input_text(inpath, outpath, 'Gallstones')
    save_input_text(inpath, outpath, 'GERD')
    save_input_text(inpath, outpath, 'Gout')
    save_input_text(inpath, outpath, 'HC')
    save_input_text(inpath, outpath, 'HT')
    save_input_text(inpath, outpath, 'Hypertension')
    save_input_text(inpath, outpath, 'OA')
    save_input_text(inpath, outpath, 'Obesity')
    save_input_text(inpath, outpath, 'OSA')
    save_input_text(inpath, outpath, 'PVD')
    save_input_text(inpath, outpath, 'VI')

if __name__ == "__main__":
    main()