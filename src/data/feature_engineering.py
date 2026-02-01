import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("FeatureEngineering")

def add_age_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Add AgeGroup categorical feature based on age bins."""
    logger.info("FeatureEngineering: Adding AgeGroup feature")
    bins = [20, 30, 40, 50, 60, 70, 100]
    labels = ['20-29','30-39','40-49','50-59','60-69','70+']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    return df

def add_bmi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add BMI_Category and BMI_Risk features."""
    logger.info("FeatureEngineering: Adding BMI features")

    def bmi_category(bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif bmi < 25:
            return 'Normal'
        elif bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'
    df['BMI_Category'] = df['BMI'].apply(bmi_category)

    def bmi_risk(bmi):
        if bmi < 18.5 or bmi >= 25:
            return 'HighRisk'
        else:
            return 'LowRisk'
    df['BMI_Risk'] = df['BMI'].apply(bmi_risk)
    return df
