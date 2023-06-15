mainPath = '../../'
rawPath = mainPath + 'raw/'
cleanPath = mainPath + 'clean/'

kimorePath = mainPath + '/KIMORE/KiMoRe.zip'

with zipfile.ZipFile(kimorePath, 'r') as zip_ref:
    zip_ref.extractall('./')

# Define directory structure
base_dir = './KiMoRe'
class_dir = ['CG', 'GPP']
sub_class_dir = [['Expert', 'NotExpert'], ['BackPain', 'Parkinson', 'Stroke']]
exercises = ['Es1', 'Es2', 'Es3', 'Es4', 'Es5']

# Define class mapping
class_mapping = {'CG-Expert': 0, 'CG-NotExpert': 1, 'GPP-Stroke': 2, 'GPP-Parkinson': 3, 'GPP-BackPain': 4}

# Define lists to store data
data_list = []

# Loop through all directories and subdirectories
for class_idx, class_name in enumerate(class_dir):
    for sub_class_name in sub_class_dir[class_idx]:
        individuals_dir = os.path.join(base_dir, class_name, sub_class_name)
        individuals = os.listdir(individuals_dir)

        for individual in individuals:
            individual_dir = os.path.join(individuals_dir, individual)
            for exercise in exercises:
                exercise_dir = os.path.join(individual_dir, exercise)

                # Load labels
                labels_path = os.path.join(exercise_dir, 'Label', f'ClinicalAssessment_{individual}.xlsx')
                label_df = pd.read_excel(labels_path)
                labels = label_df.iloc[0, 1:6].values
                label = labels[int(exercise.replace('Es', '')) - 1]

                # Load data
                raw_dir = os.path.join(exercise_dir, 'Raw')
                files = os.listdir(raw_dir)

                data_dict = {'JointPosition': np.array([]), 'JointOrientation': np.array([])}  # Initialize data dict

                for file in files:
                    for key in data_dict.keys():
                        if key in file:
                            data_path = os.path.join(raw_dir, file)
                            try:
                                data_df = pd.read_csv(data_path, header=None)
                                
                                # If the DataFrame is not empty, convert it to a numpy array
                                if not data_df.empty:
                                    data_dict[key] = data_df.values
                            except Exception as e:
                                print(f"Error reading file {file}: {str(e)}")
                # Associate data with class
                if data_dict['JointPosition'].size != 0 and data_dict['JointOrientation'].size != 0 and not np.isnan(label):
                  data_class = class_mapping[f'{class_name}-{sub_class_name}']
                  data_list.append((data_dict['JointPosition'], data_dict['JointOrientation'], data_class, label, exercise))

# Convert lists to numpy arrays
kimoreDataset = np.array(data_list, dtype=object)

# Now, 'data' is a numpy array where each element is a tuple.
# The first element of the tuple is the sensor data for one exercise of one individual, and the second element is the class of the data.
# 'labels' is a numpy array where each element is the evaluation scores for the five exercises of one individual.

