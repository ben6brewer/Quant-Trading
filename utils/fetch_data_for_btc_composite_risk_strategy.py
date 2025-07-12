from utils.fetch_fear_and_greed_index_data import *
from utils.fetch_mvrv_historical import *
from utils.fetch_btc_historical import *
from utils.fetch_pi_cycle_historical import *
from utils.fetch_200w_sma_vs_prev_top import *
from utils.pretty_print_df import *

def fetch_data_for_btc_composite_risk_strategy(period="max", interval="1d"):
    btc_df = fetch_btc_historical_data()
    mvrv_df = fetch_mvrv_historical_data()
    fng_df = fetch_fear_and_greed_index_data()
    pi_cycle_df = fetch_pi_cycle_historical_data()
    sma_df = fetch_200w_sma_vs_prev_top()

    # Select relevant columns
    btc_df = btc_df[['date', 'high', 'open', 'low', 'close']]
    mvrv_df = mvrv_df[['date', 'mvrv_risk']]
    fng_df = fng_df[['date', 'F&G_risk']]
    pi_cycle_df = pi_cycle_df[['date', 'pi_cycle_risk']]
    sma_df = sma_df[['date', 'sma_cycle_risk']]

    # Merge with btc_df as left join to keep all BTC dates
    merged_df = btc_df.merge(mvrv_df, on='date', how='left') \
                    .merge(fng_df, on='date', how='left') \
                    .merge(pi_cycle_df, on='date', how='left')


    # Calculate mean risk from all risk columns, ignoring NaNs
    risk_columns = [col for col in merged_df.columns if col.endswith('_risk')]
    merged_df['mean_risk'] = merged_df[risk_columns].mean(axis=1, skipna=True)

    # pretty_print_df(merged_df.tail(10), title="BTC Composite Risk Strategy Input Data")

    return merged_df
