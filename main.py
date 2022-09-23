import pandas as pd


def main():
    # dataset_name = 'YouGov-Imperial COVID-19 Behavior Tracker'
    dataset_name = 'owid-co2-data'
    df = pd.read_csv(f"datasets\{dataset_name}.csv")

    target_cols = ["country", "year", "iso_code", "population", "gdp", "co2", "cement_co2", "coal_co2",
                   "gas_co2", "methane", "nitrous_oxide", "oil_co2", "other_industry_co2", "share_global_cement_co2",
                   "share_global_co2", "share_global_coal_co2"]

    # Removing years without n2o or methane data

    df = df[target_cols]

    df = df.loc[(df['year'] > 1991) & (df['year'] < 2019)]  # TODO - add GPD data from external dataset for 2019-2020

    # Normalizing based on number of nan entries
    df_ = df.notnull().astype('int')
    df_['country'] = df['country']
    df_ = df_.groupby(['country'], as_index=False).mean()

    # Thresholding

    # GDP
    df_ = df_.loc[df_['gdp'] > 0.9]

    # Emission breakdowns
    df_ = df_.loc[((df_['cement_co2'] > 0.75) |
                   (df_['coal_co2'] > 0.75) |
                   (df_['gas_co2'] > 0.75))]

    # Generic density check
    df_ = df_[df_.iloc[:, 1:].mean(axis=1) > 0.85]

    # Un-squeezing country and replacing nans with 0
    df_final = df.loc[df["country"].isin(df_['country'])].fillna(0)

    output_col_ordered = ['year', 'country', 'iso_code', 'gdp', 'population', 'methane', 'nitrous_oxide', 'cement_co2',
                          'coal_co2', 'gas_co2', 'oil_co2', 'flaring_co2', 'other_co2', 'co2', 'share_methane',
                          'share_nitrous_oxide', 'share_cement_co2', 'share_coal_co2', 'share_gas_co2', 'share_oil_co2',
                          'share_flaring_co2', 'share_other_co2', 'share_co2']

    print("")


if __name__ == '__main__':
    main()
