from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import argparse
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description="Find similar counties")

    # Required argument for path to census data 
    parser.add_argument(
        "--census_data_csv",
        type=str,
        required=True,
        help="Census Data file"
    )
     # Required argument for path to crime data 
    parser.add_argument(
        "--crime_data_csv",
        type=str,
        required=True,
        help="Crime Data file"
    )
     # Required argument for path to homelessness data  
    parser.add_argument(
        "--homeless_data_csv",
        type=str,
        required=True,
        help="Homeless Data file"
    )
    # Required argument for path to where to save figure/map/csv 
    parser.add_argument(
        "--save_name",
        type=str,
        required=True,
        help="Save Name"
    )
    

    # Required argument for the number of k means clusters  
    parser.add_argument(
        "--n_clusters",
        type=int,
        required=True,
        help="Number of Clusters"
    )
    # Required argument for number of knn neighbors 
    parser.add_argument(
        "--n_neighbors",
        type=int,
        required=True,
        help="Number of KNN Neighbors"
    )

    parser.add_argument('--weight', nargs='?', const='default_value', help="Set the weight value optionally")

    return parser.parse_args()

def apply_weights(df, weights):
    """
    Apply weights to the already normalized DataFrame.
    """
    for col, weight in weights.items():
        if col in df.columns:
            df[col] *= weight
    return df

def normalize_data(df, cols_to_normalize): 
    # Select relevant columns
    # cols_to_normalize = ["poverty_level", "public_assistance_with", "total_population", "fragmentation_index"]

    # Choose a scaler (either one works)
    scaler = StandardScaler()  # Z-score standardization
    # scaler = MinMaxScaler()  # Min-max scaling (0 to 1)

    normalized_data = scaler.fit_transform(df[cols_to_normalize])
    df_normalized = pd.DataFrame(normalized_data, columns=cols_to_normalize, index=df.index)
    return df_normalized

def cluster(df_normalized, df, save_name, n_clusters, confounding_cols): 
    # Clustering without racial fragmentation
    X = df_normalized[confounding_cols]

    # Choose the number of clusters (adjust based on your data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X)

    # Compute racial fragmentation statistics within each cluster
    df_grouped = df.groupby("cluster")["fragmentation_index"].agg(["min", "max", "std", "mean"])

    # Sort counties in each cluster by racial fragmentation
    for cluster_id in df["cluster"].unique():
        print(f"Cluster {cluster_id}:")
        subset = df[df["cluster"] == cluster_id].sort_values("fragmentation_index")
        #print(subset[["county_name", "fragmentation_index"]].head(3))  # Least fragmented
        #print(subset[["county_name", "fragmentation_index"]].tail(3))  # Most fragmented
        #print()
    print(df)
    df.to_csv(f"{save_name}_clusters.csv")

def knn(df_normalized, df, save_name, nneighbors, confounding_cols): 

    # Fit KNN model
    knn = NearestNeighbors(n_neighbors=nneighbors, metric="euclidean")
    knn.fit(df_normalized[confounding_cols])

    # Find nearest neighbors for each county
    distances, indices = knn.kneighbors(df_normalized[confounding_cols])
    for i, index in enumerate(indices): 
        j = indices[i]
        #print(f"j: {j}")
        #print(f"County:{df.iloc[j[0]]["entity_name"]}")
        for n,sims in enumerate(j): 
            if n == 0: 
                fixed_county_index = sims 
                continue
            #print(f"sims: {sims}, fixed: {df.iloc[fixed_county_index]["entity_name"]} county: {df.iloc[sims]["entity_name"]}")
            df.at[fixed_county_index, f"neighbor{n}"] = f"{df.iloc[sims]["entity_name"]}"
            df.at[fixed_county_index, f"neighbor{n}_distance"] = f"{distances[i][n]}"
            df.at[fixed_county_index, f"neighbor{n}_rfi"] = df.at[sims, "fragmentation_index"]
    df.to_csv(f"{save_name}_knn.csv")
            
    return 
    # Find county pairs with large racial fragmentation differences
    county_pairs = []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # Skip itself (first entry)
            diff = abs(df.iloc[i]["racial_fragmentation"] - df.iloc[j]["racial_fragmentation"])
            county_pairs.append((df.iloc[i]["county_name"], df.iloc[j]["county_name"], diff))

    # Sort by largest racial fragmentation difference
    county_pairs = sorted(county_pairs, key=lambda x: -x[2])

    # Print top county pairs
    for pair in county_pairs[:10]:
        print(pair)



def main(): 
    args = parse_arguments()
    if args.weight is None:
        print('Weight was not set')
    elif args.weight == 'default_value':
        args.weight = 2.0 
        print(f'Weight was set without a value, using default: {args.weight}')
    else:
        args.weight = float(args.weight)
        print(f'Argument was set with value: {args.weight}')

    print(f"census data file: {args.census_data_csv}")
    data = pd.read_csv(args.census_data_csv)

    # open homelessness data file and merge with census data file 
    #homelessness_data = pd.read_csv(args.homeless_data_csv)
    #data = census_data.merge(homelessness_data, on = "GEOID")

    # get census data per 10k 
    data["below_poverty_level_per_10k"] = (data["below_poverty_level"]/data['total_population'])*10000
    data["public_assistance_with_per_10k"] = (data["public_assistance_with"]/data['total_population'])*10000

    
    # open crime data file and merge with census and homelessness data 
    '''
    crime_data = pd.read_csv(args.crime_data_csv)
    data = data.merge(crime_data, on = "GEOID")

    # sum violent crime and property crime numbers 
    data['Violent crime'] = data['Violent crime'].str.replace(',', '').astype(int)
    data['Property crime'] = data['Property crime'].str.replace(',', '').astype(int)
    cols = ['Violent crime', 'Property crime']
    data['sum_crime_per_10k'] = (data[cols].sum(axis=1)/data['total_population'])*10000
    '''
    
    
    # cluster based on confounding cols 
    # normalize all cols we care about 
    #confounding_cols = ["below_poverty_level", "public_assistance_with", "total_population", "GINI", "Homelessness_Rate_per_10k", "sum_crime_per_10k"]
    #cols_to_normalize = ["below_poverty_level", "public_assistance_with", "total_population", "GINI", "Homelessness_Rate_per_10k", "sum_crime_per_10k", "fragmentation_index"]
    #confounding_cols = ["below_poverty_level_per_10k", "public_assistance_with_per_10k", "total_population", "GINI", "Homelessness Rate per 10k"]
    #cols_to_normalize = ["below_poverty_level_per_10k", "public_assistance_with_per_10k", "total_population", "GINI", "Homelessness Rate per 10k", "fragmentation_index"]
    #confounding_cols = ["below_poverty_level_per_10k", "public_assistance_with_per_10k", "total_population", "GINI", "sum_crime_per_10k", "Median_rent_percent_income"]
    #cols_to_normalize = ["below_poverty_level_per_10k", "public_assistance_with_per_10k", "total_population", "GINI", "sum_crime_per_10k", "Median_rent_percent_income", "fragmentation_index"]
    confounding_cols = ["below_poverty_level_per_10k", "public_assistance_with_per_10k", "total_population", "GINI",  "Median_rent_percent_income"]
    cols_to_normalize = ["below_poverty_level_per_10k", "public_assistance_with_per_10k", "total_population", "GINI", "Median_rent_percent_income", "fragmentation_index"]
    #confounding_cols = ["below_poverty_level_per_10k", "public_assistance_with_per_10k", "total_population", "GINI"]
    #cols_to_normalize = ["below_poverty_level_per_10k", "public_assistance_with_per_10k", "total_population", "GINI",  "fragmentation_index"]
    
    save_name = f"{args.save_name}_{args.n_clusters}clusters_{args.n_neighbors}neighbors"
    normalized_data = normalize_data(data, cols_to_normalize)
    # apply weights 
    if args.weight: 
        print(f"Applying weight: {args.weight}")
        weights = {
        "total_population": args.weight,
        }

        data = apply_weights(data, weights)
        save_name = f"{save_name}_{args.weight}weight"

    
    # output data 
    data.to_csv(f"{save_name}.csv")

    # normalize data 
    #normalized_data = normalize_data(data, cols_to_normalize)

    # do two types of similarity scores: k means clustering and knn (k nearest neighbors)
    cluster(normalized_data, data, save_name, int(args.n_clusters), confounding_cols)
    knn(normalized_data, data, save_name, int(args.n_neighbors), confounding_cols)


if __name__ == "__main__":
    main()

