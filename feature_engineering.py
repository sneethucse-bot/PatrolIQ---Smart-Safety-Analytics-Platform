import numpy as np
from sklearn.preprocessing import MinMaxScaler

CRIME_SEVERITY = {
    "HOMICIDE": 10,
    "CRIM SEXUAL ASSAULT": 9,
    "ROBBERY": 8,
    "ASSAULT": 7,
    "BATTERY": 6,
    "NARCOTICS": 5,
    "BURGLARY": 5,
    "THEFT": 4,
    "CRIMINAL DAMAGE": 3
}

def engineer_features(df):
    df["Crime_Severity_Score"] = df["Primary Type"].map(
        lambda x: CRIME_SEVERITY.get(x, 2)
    )

    scaler = MinMaxScaler()
    df[["Lat_norm", "Lon_norm"]] = scaler.fit_transform(
        df[["Latitude", "Longitude"]]
    )

    return df
