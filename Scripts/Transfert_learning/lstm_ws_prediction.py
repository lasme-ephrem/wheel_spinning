import numpy as np
import pandas as pd
import onnxruntime as rt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences 

def preprocessing_for_nn_inference(df):

    #preprocessing for matrix design
    X = df[["heure", "duration", "score"]].values  
    liste_eleve = list(df["student_id"].unique())
    X_sequence = []
    for eleve in liste_eleve:
        d = df[(df["student_id"] == eleve)]
        liste_activite = list(d["activity_id"].unique())
        for activite in liste_activite:
            d2 = d[d["activity_id"] == activite]
            une_sequence_X = d2[["heure", "duration", "score"]].values
            X_sequence.append(une_sequence_X)
    X_seq = pad_sequences(X_sequence, maxlen=20, dtype="float32", padding="post", value = -10.)
    
    #normalization of testing dataset
    test_reshape = X_seq.reshape(-1, X_seq.shape[-1])
    colonnes = ["col" + str(i) for i in range(1, X_seq.shape[2] + 1)]
    test_df = pd.DataFrame(test_reshape, columns = colonnes)
    test_df_no_mask = test_df[test_df["col1"] != -10]
    
    #chargement onnx de la normalisation sur les données d'entraînement
    scaler = rt.InferenceSession("scaler.onnx")
    test_df_no_mask_norm = scaler.run(None, {'X': test_df_no_mask.values})[0]
    test_df.iloc[test_df_no_mask.index,] = test_df_no_mask_norm
    X_test = test_df.values.reshape(X_seq.shape)
    
    return X_test


def prediction_nn(X_test):
    #X_test = preprocessing_for_nn_inference(df)
    model = load_model('ws_model.h5')
    print(model.summary())
    y_pred = model.predict(X_test)
    prediction_class = np.where(y_pred > 0.5, 1, 0)
    return y_pred, prediction_class