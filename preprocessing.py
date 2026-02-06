import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Parse date safely
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    df = df.dropna(subset=["Date"])

    # Temporal features
    df["Hour"] = df["Date"].dt.hour
    df["Day_of_Week"] = df["Date"].dt.day_name()
    df["Month"] = df["Date"].dt.month
    df["Is_Weekend"] = df["Day_of_Week"].isin(["Saturday", "Sunday"]).astype(int)
    
    # Season
    def get_season(month):
        if month in [12, 1, 2]: return "Winter"
        elif month in [3, 4, 5]: return "Spring"
        elif month in [6, 7, 8]: return "Summer"
        else: return "Fall"
    df["Season"] = df["Month"].apply(get_season)

    # Crime severity score
    CRIME_SEVERITY = {"HOMICIDE":10,"ROBBY":8,"ASSAULT":7,"BATTERY":6,"NARCOTICS":5,"THEFT":4}
    df["Crime_Severity_Score"] = df["Primary Type"].map(lambda x: CRIME_SEVERITY.get(x, 2))

    # Drop missing coordinates
    df = df.dropna(subset=["Latitude","Longitude"])

    # Normalize lat/lon
    scaler = MinMaxScaler()
    df[["Lat_norm","Lon_norm"]] = scaler.fit_transform(df[["Latitude","Longitude"]])

    return df
