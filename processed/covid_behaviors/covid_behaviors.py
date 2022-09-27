import pandas as pd


def main():
    # dataset_name = 'YouGov-Imperial COVID-19 Behavior Tracker'
    dataset_name = 'COVID-19_Behavior_Tracker'
    df = pd.read_csv(f"{dataset_name}.csv")

    target_cols = ["Entity", "Year",
                   "household_members_contact", "people_contact_outside_household", "times_left_home_yesterday", "handwashing_yesterday",
                   "willingness_isolate_if_symptoms", "difficult_to_isolate", "willingness_isolate_if_advised",
                   "mask_outside_home", "mask_at_home", "mask_grocery_store", "mask_clothing_store", "mask_at_work", "mask_public_transport",
                   "avoided_people_with_symptoms", "avoided_going_out", "avoided_healthcare_settings", "avoided_public_transport",
                   "avoided_working_outside_home", "children_avoided_school", "avoided_guests_at_home", "avoided_small_gatherings",
                   "avoided_medium_gatherings", "avoided_large_gatherings", "avoided_crowded_areas", "avoided_shops",
                   "cleaned_surfaces_home", "covered_mouth_sneeze", "hand_sanitiser", "handwashing_yesterday",
                    "scared_contracting_covid", "happier_two_weeks_ago",
                   "covid_dangerous_to_me", "likely_get_covid_future", "mask_protect_me", "mask_protect_others", "life_greatly_affected"]

    # Removing years without n2o or methane data

    df = df[target_cols]
    df_ = df.loc[df['Year'] < 500]

    # Normalizing based on number of nan entries
    df_norm = df_.notnull().astype('int')
    df_norm['Entity'] = df['Entity']
    df_norm = df_norm.groupby(['Entity'], as_index=False).mean()



    thresholds = df_norm.loc[:, df_norm.columns != "Entity"].mean(axis=0)
    thresholded_cols = thresholds.loc[thresholds > 0.75]
    thresholded_headers = list(thresholded_cols.axes[0])

    threshold_entity = df_norm[thresholded_headers].mean(axis=1)
    thresholded_cols = threshold_entity.loc[threshold_entity > 0.9]

    # Thresholding

    # GDP
    thresholded_headers.insert(0, "Entity")
    df_ = df_[thresholded_headers]

    df_final = df_.fillna(0)

    df_final.to_csv("covid-19_behaviors.csv", index=False)


if __name__ == '__main__':
    main()
