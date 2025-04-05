import os
import pandas as pd

def collect_all(folder_paths, feature_types, type_kurtosis=1, type_skewness=1):

    datas = []
    for index, folder_path in enumerate(folder_paths):
        if index == 0:
            datas.append(pd.read_csv(f'{folder_path}/General.csv'))
        else:
            if feature_types[index] == 'kurtosis':
                feature_data = pd.read_csv(f'{folder_path}/{"Fisher (Normal)" if type_kurtosis == 1 else "Article"}/General.csv')
                feature_data = feature_data.loc[:, ~feature_data.columns.isin(datas[0].columns)]
                datas.append(feature_data)
            elif feature_types[index] == 'skewness':
                feature_data = pd.read_csv(f'{folder_path}/{"Default" if type_skewness == 1 else "Article"}/General.csv')
                feature_data = feature_data.loc[:, ~feature_data.columns.isin(datas[0].columns)]
                datas.append(feature_data)
            else:
                feature_data = pd.read_csv(f'{folder_path}/General.csv')
                feature_data = feature_data.loc[:, ~feature_data.columns.isin(datas[0].columns)]
                datas.append(feature_data)

    data = pd.concat(datas, axis=1)
    data.to_csv(f'../Dataset_Test_Eren/Graphs/Datasets/Dataset_Vector.csv', index=False)

'''    hue_data, kurtosis_data, skewness_data = pd.read_csv(f'{folder_paths[0]}/General_HueStd.csv'), [], []
    if type_kurtosis == 1:
        kurtosis_data = pd.read_csv(f'{folder_paths[1]}/Fisher (Normal)/General.csv')
    else:
        kurtosis_data = pd.read_csv(f'{folder_paths[1]}/Article/General_a.csv')
    if type_skewness == 1:
        skewness_data = pd.read_csv(f'{folder_paths[2]}/Default/General.csv')
    else:
        skewness_data = pd.read_csv(f'{folder_paths[2]}/Article/General_a.csv')

    kurtosis_data = kurtosis_data.loc[:, ~kurtosis_data.columns.isin(hue_data.columns)]
    skewness_data = skewness_data.loc[:, ~skewness_data.columns.isin(hue_data.columns)]

    data = pd.concat([hue_data, kurtosis_data, skewness_data], axis=1)
    data.to_csv(f'../Dataset_Test_Eren/Graphs/Dataset_Vector_a_HueStd.csv', index=False) '''

def collect_degrees(folder_paths, type_kurtosis, type_skewness):
    for index,folder_path in enumerate(folder_paths):
        data = []
        if index == 0 or index == 3 or index == 4 or index == 5: # hue, hsv_hue, saturation, value
            for files in os.listdir(folder_path):
                if files.endswith('.csv') and not files.startswith('General'):
                    datas = pd.read_csv(f'{folder_path}/{files}')
                    data.append(datas)
            combined_data = pd.concat(data, ignore_index=True)
            combined_data.to_csv(f'{folder_path}/General.csv', index=False)
        elif index == 1: # kurtosis
            if type_kurtosis == 1:
                for files in os.listdir(f'{folder_path}/Fisher (Normal)'):
                    if files.endswith('.csv') and not files.startswith('General'):
                        kurtosis_data = pd.read_csv(f'{folder_path}/Fisher (Normal)/{files}')
                        data.append(kurtosis_data)
            else:
                for files in os.listdir(f'{folder_path}/Article'):
                    if files.endswith('.csv') and not files.startswith('General'):
                        kurtosis_data = pd.read_csv(f'{folder_path}/Article/{files}')
                        data.append(kurtosis_data)

            combined_data = pd.concat(data, ignore_index=True)
            combined_data.to_csv(f'{folder_path}/{"Fisher (Normal)" if type_kurtosis == 1 else "Article"}/General.csv', index=False)
        elif index == 2:  # skewness
            if type_skewness == 1:
                for files in os.listdir(f'{folder_path}/Default'):
                    if files.endswith('.csv') and not files.startswith('General'):
                        skewness_data = pd.read_csv(f'{folder_path}/Default/{files}')
                        data.append(skewness_data)
            else:
                for files in os.listdir(f'{folder_path}/Article'):
                    if files.endswith('.csv') and not files.startswith('General'):
                        skewness_data = pd.read_csv(f'{folder_path}/Article/{files}')
                        data.append(skewness_data)
            combined_data = pd.concat(data, ignore_index=True)
            combined_data.to_csv(f'{folder_path}/{"Default" if type_kurtosis == 1 else "Article"}/General.csv', index=False)


def convert_to_csv(path, file_name, feature_type, type_number):
    labels = {'hue': ('Hue Mean', 'Hue Std'), 'kurtosis': ('Kurtosis A', 'Kurtosis B'), 'skewness': ('Skewness A', 'Skewness B'), 'hsv_hue': ('HSV_Hue Mean', 'HSV_Hue Std'), 'saturation': ('Saturation Mean', 'Saturation Std'), 'value': ('Value Mean', 'Value Std')}
    data = []
    with open(path, 'r') as file:
        for line in file:
            parts = line.split(',')
            degree = parts[0].split(':')[1].strip()
            feature_one = parts[1].split(':')[1].strip()
            feature_two = parts[2].split(':')[1].strip()
            data.append([degree, feature_one, feature_two])

    df = pd.DataFrame(data, columns=['Degree', labels[feature_type][0], labels[feature_type][1]])
    if feature_type == 'hue':
        df.to_csv(f'../Dataset_Test_Eren/Graphs/{feature_type.capitalize()}/{file_name.split(".")[0]}.csv', index=False)
    elif feature_type == 'kurtosis':
        if type_number == 1:
            df.to_csv(f'../Dataset_Test_Eren/Graphs/{feature_type.capitalize()}/Fisher (Normal)/{file_name.split(".")[0]}.csv', index=False)
        else:
            df.to_csv(f'../Dataset_Test_Eren/Graphs/{feature_type.capitalize()}/Article/{file_name.split(".")[0]}.csv', index=False)
    elif feature_type == 'skewness':
        if type_number == 1:
            df.to_csv(f'../Dataset_Test_Eren/Graphs/{feature_type.capitalize()}/Default/{file_name.split(".")[0]}.csv', index=False)
        else:
            df.to_csv(f'../Dataset_Test_Eren/Graphs/{feature_type.capitalize()}/Article/{file_name.split(".")[0]}.csv', index=False)
    elif feature_type == 'hsv_hue':
        df.to_csv(f'../Dataset_Test_Eren/Graphs/HSV/Hue/{file_name.split(".")[0]}.csv', index=False)
    elif feature_type == 'saturation':
        df.to_csv(f'../Dataset_Test_Eren/Graphs/HSV/{feature_type.capitalize()}/{file_name.split(".")[0]}.csv', index=False)
    elif feature_type == 'value':
        df.to_csv(f'../Dataset_Test_Eren/Graphs/HSV/{feature_type.capitalize()}/{file_name.split(".")[0]}.csv', index=False)

def arrange_folders():
    folder_paths = ['../Dataset_Test_Eren/Graphs/Hue', '../Dataset_Test_Eren/Graphs/Kurtosis', '../Dataset_Test_Eren/Graphs/Skewness', '../Dataset_Test_Eren/Graphs/HSV/Hue', '../Dataset_Test_Eren/Graphs/HSV/Saturation', '../Dataset_Test_Eren/Graphs/HSV/Value']
    feature_types = ['hue', 'kurtosis', 'skewness', 'hsv_hue', 'saturation', 'value']
    for index, folder_path in enumerate(folder_paths):
        if index == 0 or index == 3 or index == 4 or index == 5:
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.txt'):
                    convert_to_csv(f'{folder_path}/{file_name}', file_name, feature_types[index], 0)
        elif index == 1:
            inner_folder = ['Fisher (Normal)', 'Article']
            for inner_index, folder in enumerate(inner_folder):
                for file_name in os.listdir(f'{folder_path}/{folder}'):
                    if file_name.endswith('.txt'):
                        convert_to_csv(f'{folder_path}/{folder}/{file_name}', file_name, feature_types[index], 1 if inner_index == 0 else 2)
        elif index == 2:
            inner_folder = ['Default', 'Article']
            for inner_index, folder in enumerate(inner_folder):
                for file_name in os.listdir(f'{folder_path}/{folder}'):
                    if file_name.endswith('.txt'):
                        convert_to_csv(f'{folder_path}/{folder}/{file_name}', file_name, feature_types[index], 1 if inner_index == 0 else 2)


if __name__ == '__main__':
    folders = ['../Dataset_Test_Eren/Graphs/Hue', '../Dataset_Test_Eren/Graphs/Kurtosis', '../Dataset_Test_Eren/Graphs/Skewness', '../Dataset_Test_Eren/Graphs/HSV/Hue', '../Dataset_Test_Eren/Graphs/HSV/Saturation', '../Dataset_Test_Eren/Graphs/HSV/Value']
    features_types = ['hue', 'kurtosis', 'skewness', 'hsv_hue', 'hsv_saturation', 'hsv_value']
#    collect_degrees(folders, 2, 2)
#    collect_all(folders, features_types, 2, 2)
#    arrange_folders()
