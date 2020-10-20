import os
import pandas as pd
from sklearn import model_selection

# ['image_name', 'patient_id', 'sex', 'age_approx','anatom_site_general_challenge', 'diagnosis', 'benign_malignant','target'],
if __name__ == '__main__':
    input_path = '/kaggle/input/siim-isic-melanoma-classification/'
    df = pd.read_csv(os.path.join(input_path, 'train.csv'))
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values
    kf = model_selection.StratifiedKFold(n_splits=10)

    for fold_, (_,_) in enumerate(kf.split(X=df, y=y)):
        df.loc[:, 'kfold'] = fold_
    df.to_csv('train_folds.csv', index=False)