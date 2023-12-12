# project.py


import numpy as np
import pandas as pd
import pathlib
from pathlib import Path

# If this import errors, run `pip install plotly` in your Terminal with your conda environment activated.
import plotly.express as px



# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def count_monotonic(arr):
    diff= np.diff(arr)
    num_less=np.sum(diff<0)
    return num_less

def monotonic_violations_by_country(vacs):    
    output = vacs.groupby('Country_Region').agg({'Doses_admin': count_monotonic,'People_at_least_one_dose': count_monotonic})
    output_1=output.rename(columns = {'Doses_admin':'Doses_admin_monotonic', 'People_at_least_one_dose':'People_at_least_one_dose_monotonic'})
    return output_1


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def robust_totals(vacs):
    output = vacs.groupby('Country_Region').agg({'Doses_admin': lambda x: x.quantile(q=0.97),'People_at_least_one_dose':  lambda x: x.quantile(q=0.97)})

    return output


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def fix_dtypes(pops_raw):
    new_df=pops_raw.copy()
    new_df['World Percentage']= new_df['World Percentage'].str.rstrip('%').astype('float')/100
    new_df['Population in 2023']=( new_df['Population in 2023']*1000).astype('int64')
    new_df['Area (Km²)']= new_df['Area (Km²)'].str.replace(r'\D', '', regex=True).astype('int64')
    new_df['Density (P/Km²)']= new_df['Density (P/Km²)'].str.replace('/Km²', '', regex=True).astype('float')
    return new_df


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def missing_in_pops(tots, pops):
    return set(tots.index)-set(pops['Country (or dependency)'])

    
def fix_names(pops):
    change_entry={'Myanmar':'Burma',
                  'Cape Verde':'Cabo Verde',
                  'Republic of the Congo':'Congo (Brazzaville)',
                  'DR Congo':'Congo (Kinshasa)',
                 'Ivory Coast':"Cote d'Ivoire",
                 'Czech Republic':'Czechia',
                 'South Korea':'Korea, South',
                 'United States':'US',
                 'Palestine':'West Bank and Gaza'
                 }
    new_df=pops.copy()
    new_df['Country (or dependency)'] = new_df['Country (or dependency)'].replace(change_entry)
    
    return new_df


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def draw_choropleth(tots, pops_fixed):
    new_df=tots.copy()
    new_df_1=pd.merge(new_df, pops_fixed, left_on=new_df.index, right_on=pops_fixed['Country (or dependency)'])
    new_df_1['Doses Per Person']=new_df_1['Doses_admin'] / new_df_1['Population in 2023']
    fig = px.choropleth(
        new_df_1,
        locations='ISO',
        color='Doses Per Person',
        color_continuous_scale='Blues',
        hover_name='Country (or dependency)',
        title='COVID Vaccine Doses Per Person',
        labels={'Doses Per Person': 'Doses Per Person'},
        )
    return fig


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def clean_israel_data(df):
    cleaned = df.copy()
    cleaned['Age'] = pd.to_numeric(cleaned['Age'], errors='coerce')
    cleaned['Vaccinated'] = cleaned['Vaccinated'].astype(bool)
    cleaned['Severe Sickness'] = cleaned['Severe Sickness'].astype(bool)
    return cleaned


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def mcar_permutation_tests(df, n_permutations=100):
    test_vaccinated=[]
    test_Severe_Sickness=[]
    for i in range(n_permutations):
        new_df=df.copy()
        shuffled_vaccinated = df['Vaccinated'].sample(frac=1).reset_index(drop=True)
        new_df['Vaccinated']=shuffled_vaccinated
        diff_Vaccinated=abs(new_df.loc[new_df['Age'].notna()]['Vaccinated'].mean() - new_df.loc[new_df['Age'].isna()]['Vaccinated'].mean())
        test_vaccinated.append(diff_Vaccinated)
        shuffled_Severe_Sickness = df['Severe Sickness'].sample(frac=1).reset_index(drop=True)
        new_df['Severe Sickness']=shuffled_Severe_Sickness
        diff_Severe_Sickness=abs(new_df.loc[new_df['Age'].notna()]['Severe Sickness'].mean() - new_df.loc[new_df['Age'].isna()]['Severe Sickness'].mean())
        test_Severe_Sickness.append(diff_Severe_Sickness)
    
    return (np.array(test_vaccinated),np.array(test_Severe_Sickness))

    
    
def missingness_type():
    return 2


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def effectiveness(df):
    vaccinated = len(df[(df['Vaccinated'] == True)])
    unvaccinated = len(df[(df['Vaccinated'] == False)])
    vaccinated_severe = len(df[(df['Vaccinated'] == True) & (df['Severe Sickness'] == True)])
    unvaccinated_severe = len(df[(df['Vaccinated'] == False) & (df['Severe Sickness'] == True)])
    pv_prop = vaccinated_severe / vaccinated
    pu_prop = unvaccinated_severe / unvaccinated
    return 1 - pv_prop / pu_prop

# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


AGE_GROUPS = [
    '12-15',
    '16-19',
    '20-29',
    '30-39',
    '40-49',
    '50-59',
    '60-69',
    '70-79',
    '80-89',
    '90-'
]

def stratified_effectiveness(df):
    age_ranges = [(12, 15), (16, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 79), (80, 89), (90, float('inf'))]

    df_copy = df.copy()
    df_copy['Age Group'] = pd.cut(df_copy['Age'], bins=[age[0] - 1 for age in age_ranges] + [age_ranges[-1][1]], labels=AGE_GROUPS)
    grouped_probabilities = df_copy.groupby(['Age Group', 'Vaccinated'])['Severe Sickness'].mean().unstack(fill_value=0)
    effectiveness_by_age_group = 1 - (grouped_probabilities[True] / grouped_probabilities[False])

    return effectiveness_by_age_group


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def effectiveness_calculator(
    *,
    young_vaccinated_prop,
    old_vaccinated_prop,
    young_risk_vaccinated,
    young_risk_unvaccinated,
    old_risk_vaccinated,
    old_risk_unvaccinated
):

    young_effectiveness = 1 - (young_risk_vaccinated / young_risk_unvaccinated)
    old_effectiveness = 1 - (old_risk_vaccinated / old_risk_unvaccinated)
    
    overall_risk_vaccinated = (young_vaccinated_prop * young_risk_vaccinated + old_vaccinated_prop * old_risk_vaccinated) / (young_vaccinated_prop + old_vaccinated_prop)
    overall_risk_unvaccinated = ((1 - young_vaccinated_prop) * young_risk_unvaccinated + (1 - old_vaccinated_prop) * old_risk_unvaccinated) / ((1 - young_vaccinated_prop) + (1 - old_vaccinated_prop))
    overall_effectiveness = 1 - (overall_risk_vaccinated/overall_risk_unvaccinated)

    effectiveness_dict = {
        'Overall': overall_effectiveness,
        'Young': young_effectiveness,
        'Old': old_effectiveness
    }

    return effectiveness_dict


# ---------------------------------------------------------------------
# QUESTION 11
# ---------------------------------------------------------------------


def extreme_example():
    return {
        'young_vaccinated_prop': 0.01, 
        'old_vaccinated_prop': 0.99, 
        'young_risk_vaccinated': 0.01,
        'young_risk_unvaccinated':0.07,
        'old_risk_vaccinated':0.09,
        'old_risk_unvaccinated':0.53}