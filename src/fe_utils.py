"""
Domain Feature Engineering Utilities (Revised based on Empirical Auditing)
- Removed 'retention_active' since retcall is already strictly identical to (retcalls > 0).
"""
import pandas as pd

def engineer_telecom_features(data: pd.DataFrame) -> pd.DataFrame:
    df_out = data.copy()
    
    # 1. Handset Obsolescence Ratio (eqpdays / months in days)
    df_out['handset_age_ratio'] = df_out['eqpdays'] / (df_out['months'] * 30.4375 + 1.0)
    
    # 2. Cost per Minute (Unit billing pressure)
    df_out['cost_per_mou'] = df_out['revenue'] / (df_out['mou'] + 1.0)
    
    # 3. Overage Burden Ratio (Overage relative to base recurring fee)
    df_out['overage_burden'] = df_out['overage'] / (df_out['recchrge'] + 1.0)
    
    # 4. Severe Usage Drop Indicator (changem < -20% and mou > 50)
    df_out['usage_drop_severe'] = ((df_out['changem'] < -20) & (df_out['mou'] > 50)).astype(int)
    
    # 5. Network Quality Friction Rate (dropblk / mou)
    df_out['network_friction_rate'] = df_out['dropblk'] / (df_out['mou'] + 1.0)
    
    return df_out
