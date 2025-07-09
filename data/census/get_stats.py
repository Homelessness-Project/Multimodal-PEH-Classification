# Import modules
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from census import Census
from us import states
import os
import argparse
import folium
from folium import Choropleth
from branca.colormap import linear


c = Census("b0b390980b9b5b6f78dfd88feabec8d2da0f060b")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process entity type and state inputs.")

    # Optional argument for entity_type with default value "county"
    parser.add_argument(
        "--entity_type",
        type=str,
        default="county",
        help="Specify the entity type (default: 'county') (options: 'county', 'trace', 'place)."
    )

    # Optional argument for year with default value "2023"
    parser.add_argument(
        "--year",
        type=str,
        default="2023",
        help="Specify the year"
    )

    # Optional argument that takes a list with a default value
    parser.add_argument(
        "--states",              # The argument name
        type=str,               # The type of each item in the list
        nargs="*",              # Allow zero or more arguments
        default=["AK", "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "HI", "GA", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"],  # Default value if not provided
        help="A list of states"
    )

    # Required argument for path to where to save figure/map 
    parser.add_argument(
        "--save_name",
        type=str,
        required=True,
        help="Specify stem for saved files"
    )

    # Optional argument for crime data
    parser.add_argument(
        "--crime",
        type=str,
        help="Path to crime data file"
    )

    return parser.parse_args()

def get_fips(state_abbr, entity_type):
    """
    Fetch a list of county FIPS codes for a given state.
    
    Parameters:
    state_abbr (str): State abbreviation (e.g., 'CA' for California).
    
    Returns:
    list: List of county FIPS codes for the state.
    """
    state_fips = states.lookup(state_abbr).fips
    if entity_type == "county": 
        counties = c.acs5.state_county(fields=["NAME"], state_fips=state_fips, county_fips="*")
        return [{"fips": county["county"], "name": county["NAME"], "state": county["state"], "entity": county["county"]} for county in counties]
    elif entity_type == "place": 
        places = c.acs5.state_place(fields=["NAME"], state_fips=state_fips, place="*")
        print(f"places: {places}")
        return [{"fips": place["place"], "name": place["NAME"], "state": place["state"], "entity": place["place"]} for place in places]

def fetch_population_data(state_abbr, entity_type, census_keys):
    """
    Fetch racial/ethnic group population data for all counties in a state.
    
    Parameters:
    state_abbr (str): State abbreviation (e.g., 'CA' for California).
    
    Returns:
    list: Population data for each county.
    """
    state_fips = states.lookup(state_abbr).fips
    entity_list = get_fips(state_abbr, entity_type)

    results = []
    
    if entity_type == "county": 
        for entity in entity_list:
            data = c.acs5.state_county(
                fields=list(census_keys.values()),
                state_fips=state_fips,
                county_fips=entity["fips"]
            )
            # Add entity name and data to results
            if data:
                results.append({"name": entity["name"], "data": data[0], "state": entity["state"], "entity": entity["entity"]})
    elif entity_type == "place": 
        for entity in entity_list:
            data = c.acs5.state_place(
                fields=list(census_keys.values()),
                state_fips=state_fips,
                place=entity["fips"]
            )
            # Add entity name and data to results
            if data:
                results.append({"name": entity["name"], "data": data[0], "state": entity["state"], "entity": entity["entity"]})
    return results

def calculate_racial_fragmentation(population_data, census_keys):
    """
    Calculate racial fragmentation index for each county.
    
    Parameters:
    population_data (list): List of population data for racial/ethnic groups.
    
    Returns:
    list: List of dictionaries with county names and fragmentation indices.
    """
    results = []
    for entry in population_data:
        entity_name = entry["name"]
        data = entry["data"]
        
        # Total population
        total_population = sum([
            data.get("B02001_002E", 0),  # White
            data.get("B02001_003E", 0),  # Black
            data.get("B02001_004E", 0),  # American Indian
            data.get("B02001_005E", 0),  # Asian
            data.get("B02001_006E", 0),  # Pacific Islander
            data.get("B02001_007E", 0),  # Other race
            data.get("B02001_008E", 0),  # Two or more races
            data.get("B03003_003E", 0),  # Hispanic or Latino
        ])
        
        if total_population == 0:
            continue  # Skip counties with no population
        
        # Calculate proportions
        proportions = [
            data.get("B02001_002E", 0) / total_population,  # White
            data.get("B02001_003E", 0) / total_population,  # Black
            data.get("B02001_004E", 0) / total_population,  # American Indian
            data.get("B02001_005E", 0) / total_population,  # Asian
            data.get("B02001_006E", 0) / total_population,  # Pacific Islander
            data.get("B02001_007E", 0) / total_population,  # Other race
            data.get("B02001_008E", 0) / total_population,  # Two or more races
            data.get("B03003_003E", 0) / total_population,  # Hispanic or Latino
        ]
        
        # Calculate Racial Fragmentation Index
        rfi = 1 - sum(p ** 2 for p in proportions)
        geoid = entry["state"] + entry["entity"]
        print(f"entity name: {entity_name}")

        # Get Other Stats 
        stats = {}
        for stat in census_keys: 
            #public_assistance = data.get("B19058_001E", 0) # PUBLIC ASSISTANCE INCOME OR FOOD STAMPS/SNAP IN THE PAST 12 MONTHS FOR HOUSEHOLDS
            stats[stat] = data.get(census_keys[stat], 0)
            print(f"{stat}: {data.get(census_keys[stat], 0)}")
        base = {"entity_name": entity_name, "fragmentation_index": rfi, "state": entry["state"], "entity": entry["entity"], "GEOID": geoid}
        base.update(stats)
        results.append(base)
    
    return results

def create_interactive_rfi_map(rfi_results, state_ids, entity_type, save_name, shapefile_path):
    # Load the county shapefile
    if entity_type == "county": 
        entities = gpd.read_file(shapefile_path)
        print(f"columns: {entities.columns.values}")
    elif entity_type == "place": 
        all_entities = [] 
        for state_id in state_ids: 
            shapefile_path = f"{shapefile_path}_{state_id}_place.zip"
            try: 
                entities = gpd.read_file(shapefile_path)
                all_entities.append(entities)
            except Exception as e: 
                print(f"could not open shapefile: {shapefile_path}")
                continue 
        entities = pd.concat(all_entities, ignore_index=True)
 

    # TODO: remove this filtering? 
    entities = entities[entities["STATEFP"].isin(state_ids)]
    entities.rename(columns={"geoid": "STATEFP"}, inplace=True)

    entities_merged = entities.merge(rfi_results, on = "GEOID")
    
    # Convert to JSON for mapping
    entities_json = entities_merged.to_json()
    #print(f"json: {entities_json}")

    # Create color scale
    colormap = linear.YlGnBu_03.scale(entities_merged["fragmentation_index"].min(), entities_merged["fragmentation_index"].max())

    # Create Folium map
    m = folium.Map(location=[37.8, -96], zoom_start=4)
    
   # Add Choropleth layer
    folium.Choropleth(
        geo_data=entities_json,
        name="Racial Fragmentation",
        data=entities_merged,
        columns=["GEOID", "fragmentation_index"],
        key_on="feature.properties.GEOID",
        fill_color="YlGnBu",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Racial Fragmentation Index",
    ).add_to(m)

    # Add GeoJson layer for tooltips
    folium.GeoJson(
        entities_json,
        name="Entities",
        style_function=lambda feature: {
            "fillColor": colormap(feature["properties"].get("fragmentation_index", 0)
) if feature["properties"].get("fragmentation_index", 0)
 is not None else "gray",
            "color": "black",
            "weight": 0.5,
            "fillOpacity": 0.7
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["GEOID", "fragmentation_index", "entity_name"], 
            aliases=["Entity GEOID:", "Fragmentation Index:", "Entity:"], 
            localize=True
        )
    ).add_to(m)

    # Add color bar
    #colormap.caption = "Racial Fragmentation Index"
    #m.add_child(colormap)

    folium.LayerControl().add_to(m)
    
    # Save the map to an HTML file or display
    m.save(f"{save_name}.html")
    return m


def create_rfi_map(rfi_results, state_ids, entity_type, save_name, shapefile_path):
    """
    Create a color-coded map of racial fragmentation index (RFI) by county.
    
    Parameters:
    rfi_results (list): List of dictionaries with county names and RFI values.
    shapefile_path (str): Path to the county shapefile.
    """
    # Load the county shapefile
    if entity_type == "county": 
        entities = gpd.read_file(shapefile_path)
        print(f"columns: {entities.columns.values}")
    elif entity_type == "place": 
        all_entities = [] 
        for state_id in state_ids: 
            shapefile_path = f"{shapefile_path}_{state_id}_place.zip"
            try: 
                entities = gpd.read_file(shapefile_path)
                all_entities.append(entities)
            except Exception as e: 
                print(f"could not open shapefile: {shapefile_path}")
                continue 
        entities = pd.concat(all_entities, ignore_index=True)
 
    entities.rename(columns={"geoid": "STATEFP"}, inplace=True)
    # TODO: remove this filtering? 
    entities = entities[entities["STATEFP"].isin(state_ids)]
    
    # TODO: should I reproject? 
    #counties = counties.to_crs(epsg = 32617)

    # Print GeoDataFrame of shapefile
    print(f"Entities in RFI map function")
    print(entities)
    print('Shape: ', entities.shape)

    # Check shapefile projection
    print("\nThe shapefile projection is: {}".format(entities.crs))
    print(f"columns: {entities.columns.values}")
    print(entities["GEOID"])
    print("Column data types for shapefile data:\n{}".format(entities.dtypes))
    
    entities_merged = entities.merge(rfi_results, on = "GEOID")
    #print(f"Entities Merged: {entities_merged}")
    print(f"columns: {entities_merged.columns.values}")
    for_csv = entities_merged.drop('geometry', axis=1)
    for_csv.to_csv(f"{save_name}.csv")

    # Load the state boundary shapefile
    state_shapefile = f"./us-state-boundaries.zip"
    states = gpd.read_file(state_shapefile)
    states.rename(columns={"geoid": "STATEFP"}, inplace=True)
    states = states[states["STATEFP"].isin(state_ids)]  # Filter by selected states

     # Plot the map
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    entities_merged.plot(column="fragmentation_index", 
                cmap="coolwarm", 
                linewidth=0.8, 
                ax=ax, 
                edgecolor="0.8", 
                legend=True)
    
    # Overlay state boundaries
    states.boundary.plot(ax=ax, color="black", linewidth=1.5, label="State Boundaries")
    
    ax.set_title(f"Racial Fragmentation Index (RFI) by {entity_type} ", fontsize=16)
    ax.axis("off")
    plt.savefig(f"{save_name}.png")
    plt.show()
   
def main():
    args = parse_arguments()
    entity_type = args.entity_type 
    year = args.year
    print(f"Entity Type: {entity_type}")
    print(f"Year: {year}")
    shapefile="./tl_2024_us_county.zip"
    
    if entity_type == "county": 
        pass 
        shapefile = f"https://www2.census.gov/geo/tiger/TIGER{year}/COUNTY/tl_{year}_us_county.zip"
    elif entity_type == "place": 
        print(f"Place option not working")
        exit(1)
        shapefile = f"https://www2.census.gov/geo/tiger/TIGER{year}/PLACE/tl_{year}" # this is for 1 state 
    shapefile="./tl_2024_us_county.zip"
    print(f"Shapefile: {shapefile}")
    print(f"States: {args.states}")
    state_abbreviations = args.states
    
    # Example usage: Fetch and calculate RFI for all counties in California
    #state_abbreviations = ["IN", "DE", "MI", "MO", "OH", "IL", "CA"]
    #state_abbreviations = ["DE"]
    # this does not include Washington DC 
    #state_abbreviations= ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
    census_keys={
                    "race_total": "B02001_001E",  # Race Total
                    "white": "B02001_002E",  # White alone
                    "black": "B02001_003E",  # Black or African American alone
                    "native": "B02001_004E",  # American Indian and Alaska Native alone
                    "asian": "B02001_005E",  # Asian alone
                    "hawaiin": "B02001_006E",  # Native Hawaiian and Other Pacific Islander alone
                    "other": "B02001_007E",  # Some other race alone
                    "multi": "B02001_008E",  # Two or more races
                    "hispanic": "B03003_003E",  # HISPANIC OR LATINO ORIGIN TODO: check this 
                    "public_assistance_or_food_stamps": "B19058_001E",   # PUBLIC ASSISTANCE INCOME OR FOOD STAMPS/SNAP IN THE PAST 12 MONTHS FOR HOUSEHOLDS
                    "public_assistance_or_food_stamps_with_cash": "B19058_002E",   # PUBLIC ASSISTANCE INCOME OR FOOD STAMPS/SNAP IN THE PAST 12 MONTHS FOR HOUSEHOLDS
                    "public_assistance_or_food_stamps_no_cash": "B19058_003E",   # PUBLIC ASSISTANCE INCOME OR FOOD STAMPS/SNAP IN THE PAST 12 MONTHS FOR HOUSEHOLDS
                    "social_security": "B19055_001E", # SOCIAL SECURITY INCOME IN THE PAST 12 MONTHS FOR HOUSEHOLDS
                    "social_security_with": "B19055_002E", # SOCIAL SECURITY INCOME IN THE PAST 12 MONTHS FOR HOUSEHOLDS
                    "social_security_without": "B19055_003E", # SOCIAL SECURITY INCOME IN THE PAST 12 MONTHS FOR HOUSEHOLDS
                    "supplemental_security": "B19056_001E", # SUPPLEMENTAL SECURITY INCOME (SSI) IN THE PAST 12 MONTHS FOR HOUSEHOLDS
                    "supplemental_security_with": "B19056_002E", # SUPPLEMENTAL SECURITY INCOME (SSI) IN THE PAST 12 MONTHS FOR HOUSEHOLDS
                    "supplemental_security_without": "B19056_003E", # SUPPLEMENTAL SECURITY INCOME (SSI) IN THE PAST 12 MONTHS FOR HOUSEHOLDS
                    "public_assistance": "B19057_001E", # PUBLIC ASSISTANCE INCOME IN THE PAST 12 MONTHS FOR HOUSEHOLDS
                    "public_assistance_with": "B19057_002E", # PUBLIC ASSISTANCE INCOME IN THE PAST 12 MONTHS FOR HOUSEHOLDS
                    "public_assistance_without": "B19057_003E", # PUBLIC ASSISTANCE INCOME IN THE PAST 12 MONTHS FOR HOUSEHOLDS
                    "total_population": "B01003_001E",	# TOTAL POPULATION
                    "poverty_level":"B17001_001E", # POVERTY STATUS IN THE PAST 12 MONTHS BY SEX BY AGE
                    "below_poverty_level":"B17001_002E", # POVERTY STATUS IN THE PAST 12 MONTHS BY SEX BY AGE
                    "above_poverty_level":"B17001_031E", # POVERTY STATUS IN THE PAST 12 MONTHS BY SEX BY AGE
                    "GINI":"B19083_001E" # GINI INDEX OF INCOME INEQUALITY
    }
    
    state_ids = []
    all_population_data = []
    all_rfi_results = [] 
    for state_abbr in state_abbreviations: 
        print(f"Getting data for state: {state_abbr}...")
        state_fips = states.lookup(state_abbr).fips
        print(f"fips for state: {state_abbr} is {state_fips}")
        state_ids.append(str(state_fips))
        population_data = fetch_population_data(state_abbr, entity_type, census_keys)
        
        print("Population Data")
        print(population_data)

        print(f"Adding population data for state: {state_abbr} to overall list")
        all_population_data.extend(population_data)


        rfi_results = calculate_racial_fragmentation(population_data, census_keys)
        print("RFI Results")
        print(rfi_results)

        print(f"Adding rfi results for {state_abbr} to all rfi results")
        all_rfi_results.extend(rfi_results)

    
    rfi_df = pd.DataFrame(all_rfi_results)

    '''
    print(f"Crime Data file: {args.crime}")
    crime_df = pd.read_csv(args.crime)
    print(f"crime data: {crime_df.columns.values}")
    print(f"dataframe columns: {rfi_df.columns.values}\n\n")
    # need to remove any words after "County"
    # need to add state check 
    for county in crime_df["County"]: 
        
        #print(f"County: {county}")
        rfi_df.loc[rfi_df['entity_name'].str.contains(f"{county}", case=False, na=False), 'new_col'] = 'found'
        #if (rfi_df['entity_name'].str.contains(county).any()): 
        #    print(f"found match for {county}: {rfi_df['entity_name'].str.contains(county).any()}") 
    #return 
    '''
    print(f"dataframe columns: {rfi_df.columns.values}\n\n")
    create_rfi_map(rfi_df, state_ids, entity_type, args.save_name, shapefile_path=shapefile)
    #create_interactive_rfi_map(rfi_df, state_ids, entity_type, args.save_name, shapefile_path=shapefile)

if __name__ == "__main__":
    main()

