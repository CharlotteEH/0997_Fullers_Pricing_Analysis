import time

import pandas as pd
import numpy as np
import itertools
from itertools import combinations

from a_connection import engine

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


def time_taken(func):
    """
    Wrapper for timing functions.
    :param func: Function to be timed.
    :return: Results of the function and prints the time taken for function to run.
    """

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()

        print(f'{func.__name__}: {end - start:.2f}')
        return result

    return wrapper


@time_taken
def read_in_data():
    data = pd.read_sql('''
 SELECT  
    D_datekey AS date,
    rel_qtr AS qtr,
    d_period AS period,
    d_yr AS year,
    OT_CGAIdent AS outlet,
	pm_uomdescription as size,
	pt_at_format as format,
    PT_ProductDescription AS product,
	case when
		OT_TL4_CGA = 'London & South East' then 'London & South East'
		else 'n'
	end as region1,
	case when
		OT_M25_TL1 = 'Inside M25' then 'Inside M25'
		else 'n'
	end as region2,
	CASE 
		WHEN OT_County = 'Hampshire' then 'Hampshire'
		WHEN OT_DistrictDescription = 'City And County of the City of London' THEN 'City of London'
		WHEN OT_DistrictDescription = 'City of Westminster London Borough' THEN 'City of Westminster'
		ELSE 'n'
	END AS region3,
	case when
		OT_DistrictDescription in ('Brent London Boro', 'Haringey London Boro', 'Camden London Boro', 'Islington London Boro', 
		'City And County of the City of London', 'Kensington and Chelsea London Boro', 'City of Westminster London Borough', 
		'Lambeth London Boro', 'Ealing London Boro', 'Lewisham London Boro', 'Enfield London Boro', 'Newham London Boro', 
		'Greenwich London Boro', 'Southwark London Boro', 'Hackney London Boro', 'Hammersmith And Fulham London Boro', 
		'Tower Hamlets London Boro', 'Wandsworth London Boro') then 'Central London'
		when OT_DistrictDescription in ('Barking and Dagenham London Boro',	'Barnet London Boro',	'Bexley London Boro',	'Brentwood District (B)',	'Bromley London Boro',	
		'Broxbourne District (B)',	'Croydon London Boro',	'Dacorum District (B)',	'Dartford District (B)',	'Elmbridge District (B)',	
		'Epping Forest District',	'Epsom and Ewell District (B)',	'Harrow London Boro',	'Havering London Boro',	'Hertsmere District (B)',	
		'Hillingdon London Boro',	'Hounslow London Boro',	'Kingston upon Thames London Boro',	'Merton London Boro',	'Mole Valley District',	
		'Redbridge London Boro',	'Reigate and Banstead District (B)',	'Richmond upon Thames London Boro',	'Runnymede District (B)',	
		'Sevenoaks District',	'South Bucks District',	'Spelthorne District (B)',	'St. Albans District (B)',	'Sutton London Boro',	'Tandridge District',	
		'Three Rivers District',	'Thurrock (B)',	'Waltham Forest London Boro',	'Watford District (B)',	'Woking District (B)') then 'Suburban London'
		else 'n'
	end as region4,
    OT_SL2_Novellus AS segment,
    SUM([F_SalesValue_£]) AS value,
    SUM(F_SalesVolume_MLS) AS volume,
    SUM(F_SalesQuantity) AS quantity,
    PI_ClientDescription AS data_partner

FROM [WS_LIVE].[dbo].[vw_Epos_Weekly]

WHERE
    D_DateKey >= '2021-12-25 00:00:00.000'  -- 3 years ago
    AND OT_SL5_Novellus = 'Pub'
	AND OT_TL4_CGA = 'London & South East'
    AND (
        (PT_ProductDescription = 'Fosters' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Carling' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Carlsberg Danish Pilsner' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Amstel' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Cruzcampo' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Pravha' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Heineken Original' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'San Miguel' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Madri Excepcional' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Peroni Nastro Azzurro' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Asahi Super Dry' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Birra Moretti' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Estrella Damm' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Camden Hells Lager' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Camden Town Pale Ale' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Beavertown Neck Oil Session IPA' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Tiny Rebel Easy Livin' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Brewdog Punk IPA' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Guinness' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Cornish Orchards Cornish Gold Cider' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Aspall Cyder' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Fullers London Pride (Cask)' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Sharps Doom Bar (Cask)' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'St Austell Tribute Ale (Cask)' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Timothy Taylors Landlord (Cask)' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR

		(PT_ProductDescription = 'Peroni Nastro Azzurro' AND PM_UomDescription = '330ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Heineken Original' AND PM_UomDescription = '330ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Sol' AND PM_UomDescription = '330ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Peroni Nastro Azzurro 0.0' AND PM_UomDescription = '330ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Corona Extra' AND PM_UomDescription = '330ml' AND PT_AT_Format = 'packaged') OR

		(PT_ProductDescription = 'Tanqueray' AND PM_UomDescription = '25ml') OR
		(PT_ProductDescription = 'Beefeater' AND PM_UomDescription = '25ml') OR
		(PT_ProductDescription = 'Smirnoff Black' AND PM_UomDescription = '25ml') OR
		(PT_ProductDescription = 'Smirnoff Red' AND PM_UomDescription = '25ml') OR
		(PT_ProductDescription = 'Absolut Blue' AND PM_UomDescription = '25ml') OR
		(PT_ProductDescription = 'Bacardi Carta Negra' AND PM_UomDescription = '25ml') OR
		(PT_ProductDescription = 'Bacardi Carta Blanca' AND PM_UomDescription = '25ml') OR
		(PT_ProductDescription = 'Kraken Black Spiced Rum' AND PM_UomDescription = '25ml') OR
		(PT_ProductDescription = 'Havana Club 3 Year Old' AND PM_UomDescription = '25ml') OR
		(PT_ProductDescription = 'Jack Daniels' AND PM_UomDescription = '25ml') OR
		(PT_ProductDescription = 'Johnnie Walker Black Label 12 Year Old' AND PM_UomDescription = '25ml') OR
		(PT_ProductDescription = 'Johnnie Walker Red Label' AND PM_UomDescription = '25ml') OR
		(PT_ProductDescription = 'Jameson' AND PM_UomDescription = '25ml') OR

		(PT_ProductDescription = 'Diet Pepsi' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Pepsi' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Pepsi Max' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Diet Coke' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Coke Zero' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR
		(PT_ProductDescription = 'Coca-Cola' AND PM_UomDescription = 'pint' AND PT_AT_Format = 'draught') OR


		(PT_ProductDescription = 'Fever Tree Indian Tonic Water'AND PM_UomDescription = '200ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Fever Tree Refreshingly Light Indian Tonic'AND PM_UomDescription = '200ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Fever Tree Mediterranean Tonic Water'AND PM_UomDescription = '200ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Fever Tree Elderflower Tonic Water'AND PM_UomDescription = '200ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Fever Tree Lemon Tonic'AND PM_UomDescription = '200ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Fever Tree Rhubarb & Raspberry Tonic'AND PM_UomDescription = '200ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Fever Tree Citrus Tonic'AND PM_UomDescription = '200ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Fever Tree Refreshingly Light Sweet Rhubarb & Raspberry Tonic'AND PM_UomDescription = '200ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Fever Tree Refreshingly Light Cucumber Tonic Water'AND PM_UomDescription = '200ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Fever Tree Refreshingly Light Aromatic Tonic'AND PM_UomDescription = '200ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Fever Tree Refreshingly Light Mediterranean Tonic Water'AND PM_UomDescription = '200ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Fever Tree Refreshingly Light Lemon Tonic Water'AND PM_UomDescription = '200ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Fever Tree Refreshingly Light Elderflower Tonic Water'AND PM_UomDescription = '200ml' AND PT_AT_Format = 'packaged') OR

		(PT_ProductDescription = 'Old Mout Cider Pineapple & Raspberry'AND PM_UomDescription = '500ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Old Mout Cider Kiwi & Lime'AND PM_UomDescription = '500ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Old Mout Cider Berries & Cherries'AND PM_UomDescription = '500ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Old Mout Strawberry & Apple'AND PM_UomDescription = '500ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Old Mout Cider Pomegranate & Strawberry'AND PM_UomDescription = '500ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Old Mout Cider Watermelon and Lime'AND PM_UomDescription = '500ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Any Other Old Mout'AND PM_UomDescription = '500ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Old Mout Cider Passionfruit & Apple'AND PM_UomDescription = '500ml' AND PT_AT_Format = 'packaged') OR

		(PT_ProductDescription = 'Rekorderlig Strawberry & Lime'AND PM_UomDescription = '500ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Rekorderlig Wild Berries'AND PM_UomDescription = '500ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Rekorderlig Blood Orange'AND PM_UomDescription = '500ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Rekorderlig Passion Fruit'AND PM_UomDescription = '500ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Rekorderlig Watermelon'AND PM_UomDescription = '500ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Rekorderlig Mango and Raspberry'AND PM_UomDescription = '500ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Rekorderlig Peach & Raspberry'AND PM_UomDescription = '500ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Rekorderlig Spiced Plum'AND PM_UomDescription = '500ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Rekorderlig Blackberry And Blackcurrant'AND PM_UomDescription = '500ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Rekorderlig Pear'AND PM_UomDescription = '500ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Rekorderlig Apple Cider'AND PM_UomDescription = '500ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Rekorderlig Pink Lemon Cider'AND PM_UomDescription = '500ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Any Other Rekorderlig'AND PM_UomDescription = '500ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Rekorderlig Peach & Apricot'AND PM_UomDescription = '500ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Rekorderlig Winter'AND PM_UomDescription = '500ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Rekorderlig Apple and Blackcurrant'AND PM_UomDescription = '500ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Rekorderlig Orange and Ginger'AND PM_UomDescription = '500ml' AND PT_AT_Format = 'packaged') OR
		(PT_ProductDescription = 'Rekorderlig Apple & Guava'AND PM_UomDescription = '500ml' AND PT_AT_Format = 'packaged')
    )

    AND OT_CGAIdent > 0
    AND PT_ProductId > 0
    AND F_SalesVolume_MLS > 0
    AND [F_SalesValue_£] > 0
    AND F_SalesQuantity > 0
    AND OT_TL5_ISBA = 'GB'
GROUP BY 
    D_datekey,
    d_period,
    rel_qtr,
    d_yr,
    OT_CGAIdent,
	pm_uomdescription,
	pt_at_format,
    PT_ProductDescription,
	OT_SL2_Novellus,
    OT_M25_TL1, 
	OT_TL4_CGA, 
	OT_DistrictDescription, 
	OT_County,
    OT_Quality_CGA,
    PI_ClientDescription;


            ''', engine)


    data.loc[data["product"].str.contains("Fever Tree", case=False, na=False), "product"] = "Any Fever Tree Tonic"
    data.loc[data["product"].str.contains("Old Mout", case=False, na=False), "product"] = "Any Full Alcohol Old Mout"
    data.loc[
        data["product"].str.contains("Rekorderlig", case=False, na=False), "product"] = "Any Full Alcohol Rekorderlig"

    data["product"] = np.where(
        data["format"].isin(["Draught", "Packaged"]),
        data["format"] + " " + data["product"] + " " + data["size"],
        data["product"] + " " + data["size"]
    )

    data = data.drop(columns=["size", "format"])

    data['date'] = pd.to_datetime(data['date'])
    data = data.astype({
        'period': 'uint32',
        'qtr': 'uint32',
        'year': 'uint32',
        'outlet': 'category',
        'product': 'category',
        'region1': 'category',
        'region2': 'category',
        'region3': 'category',
        'region4': 'category',
        'segment': 'category',
        'volume': 'float',
        'value': 'float',
        'quantity': 'float'
    })

    return data


@time_taken
def pull_data(file_name: str, pull):
    try:
        data = pd.read_parquet(file_name)
        print(f'{file_name} already exists, skipping data pull.')

    except FileNotFoundError as e:
        print(f'{e} - performing data pull')
        data = pull()
        data.to_parquet(file_name)

    return data


@time_taken
def remove_outliers(
        data: pd.DataFrame,
        group_columns: list[str],
        pint_products: list[str],
        products_330: list[str],
        products_25: list[str],
        products_200: list[str],
        products_500: list[str],
        metrics: list[str] = None,
        st_devs: int = 3,
        units: int = None
):
    data = data.dropna()
    print("shape is", data.shape)

    if metrics is None:
        metrics = ['value', 'volume']

    #     print(
    # "units is", units
    #     )
    #     print(data.head())
    #     if units is not None:
    #         data["volume"] = data["volume"] / units
    #     print(data.head())

    # decide units for different products, for the food one change the volume units to quantity
    data["volume"] = data["volume"].where(
        ~data["product"].isin(pint_products),
        data["volume"] / 568
    )

    # data["volume"] = data["volume"].where(
    #     ~data["product"].isin(other_products),
    #     data["quantity"]
    # )

    data["volume"] = data["volume"].where(
        ~data["product"].isin(products_330),
        data["volume"] / 330
    )
    data["volume"] = data["volume"].where(
        ~data["product"].isin(products_25),
        data["volume"] / 25
    )
    data["volume"] = data["volume"].where(
        ~data["product"].isin(products_200),
        data["volume"] / 200
    )
    data["volume"] = data["volume"].where(
        ~data["product"].isin(products_500),
        data["volume"] / 500
    )

    size_1 = data.shape[0]

    for m in metrics:
        data[f'mean_{m}'] = data.groupby(group_columns, observed=True)[m].transform('mean')
        data[f'stdev_{m}'] = data.groupby(group_columns, observed=True)[m].transform('std')

        data = data[abs(data[f'mean_{m}'] - data[m]) < (st_devs * data[f'stdev_{m}'])]

        data.drop(columns=[f'mean_{m}', f'stdev_{m}'], inplace=True)

    print(f'Outliers removed from set: {size_1 - data.shape[0]:,}')
    return data


@time_taken
def get_continuous_items(
        data: pd.DataFrame,
        date_count_threshold: int = 2,
        item_column: str = 'brand',
        date_column: str = 'date',
        format_column: str = 'format',
        outlet_column: str = 'outlet',
        period_column: str = 'period'
):
    g_c = [x for x in data.columns if x in [outlet_column, period_column, format_column, item_column]]

    outlets = data[[date_column] + g_c].drop_duplicates()

    outlets = outlets.groupby(
        g_c, observed=True
    ).agg(
        {date_column: 'nunique'}
    ).reset_index(drop=False)

    return pd.merge(
        left=data,
        right=outlets[outlets[date_column] > date_count_threshold][g_c],
        how='inner',
        on=g_c
    )


@time_taken
def price_banding_function(products: list[str],
                           brands: list[str],
                           quantiles: list[float],
                           metrics: list[str],
                           region_columns: list[str],
                           segment_columns: list[str],
                           data: pd.DataFrame,
                           period_filter: list[int],
                           date_column: str = 'date',
                           period_column: str = 'period',
                           outlet_column: str = 'outlet',
                           data_partner_column: str = 'data_partner'):
    data["price"] = data["value"] / data["quantity"]
    print("data is", "\n", data.head())
    data = data[data[period_column].isin(period_filter)]
    print("data is", "\n", data.head())

    reg_seg = [col for col in region_columns + segment_columns if col in data.columns]

    results = []
    counts = []
    for product in products:
        for brand in brands:
            filtered_df = data[data[product] == brand]
            # print("product is", brand, "filtered data is", "\n", filtered_df.head())

            group_columns = [date_column, period_column, product]
            if reg_seg:
                for region_column in region_columns:
                    for segment_column in segment_columns:
                        cols_to_add_global = [region_column, segment_column]

                        result_df = pd.concat([
                            filtered_df.groupby(list(group_columns) + list(combo), observed=True)["price"].quantile(
                                quantiles).unstack().reset_index()
                            for i in range(0, 3)
                            for combo in combinations(cols_to_add_global, 2 - i)
                            # do widest df first to make sure columns are created
                        ])  # 2-i means we start with 2, then 1 then 0.

                        result_df = result_df.rename(columns={product: "product"})

                        # print("result df is", "\n", result_df.head())

                        result_df["product_type"] = product
                        result_df[region_column] = result_df[region_column].cat.add_categories('GB')
                        result_df[region_column] = result_df[region_column].fillna('GB')
                        result_df[segment_column] = result_df[segment_column].cat.add_categories('All Pubs')
                        result_df[segment_column] = result_df[segment_column].fillna('All Pubs')

                        result_df = result_df.rename(columns={region_column: "region",
                                                              segment_column: "segment"})

                        results.append(result_df)

            else:
                result_df = filtered_df.groupby(list(group_columns), observed=True)["price"].quantile(
                    quantiles).unstack().reset_index()

                result_df = result_df.rename(columns={product: "product"})

                # print("result df is", "\n", result_df.head())

                result_df["product_type"] = product

                results.append(result_df)

            group_columns = [period_column, product]

            if reg_seg:
                for region_column in region_columns:
                    for segment_column in segment_columns:
                        cols_to_add_global = [region_column, segment_column]

                        count_df = pd.concat([
                            filtered_df.groupby(
                                list(group_columns) + list(combo), observed=True
                            ).agg({outlet_column: "nunique"}).reset_index()
                            # ,data_partner_column: "nunique"}).reset_index()
                            for i in range(0, 3)
                            for combo in combinations(cols_to_add_global, 2 - i)
                            # do widest df first to make sure columns are created
                        ])  # 2-i means we start with 2, then 1 then 0.

                        count_df = count_df.rename(columns={product: "product",
                                                            outlet_column: "total_outlet_count"})

                        # print("result df is", "\n", count_df.head())

                        count_df["product_type"] = product
                        count_df[region_column] = count_df[region_column].cat.add_categories('GB')
                        count_df[region_column] = count_df[region_column].fillna('GB')
                        count_df[segment_column] = count_df[segment_column].cat.add_categories('All Pubs')
                        count_df[segment_column] = count_df[segment_column].fillna('All Pubs')

                        count_df = count_df.rename(columns={region_column: "region",
                                                            segment_column: "segment"})

                        counts.append(count_df)

            else:
                count_df = filtered_df.groupby(
                    list(group_columns), observed=True
                ).agg({outlet_column: "nunique"}).reset_index()

                count_df = count_df.rename(columns={product: "product",
                                                    outlet_column: "total_outlet_count"})

                # print("result df is", "\n", count_df.head())

                count_df["product_type"] = product

                counts.append(count_df)

    final_result_df = pd.concat(results, ignore_index=True)
    final_result_df = final_result_df.drop_duplicates()
    from pathlib import Path

    path = Path("H:/0997_Fullers_Pricing_Analysis/Results")
    final_result_df.to_csv(path / "test1.csv", index=False)

    final_count_df = pd.concat(counts, ignore_index=True)
    final_count_df = final_count_df.drop_duplicates()
    print("final result df is", "\n", final_result_df.head())
    print("final count df is", "\n", final_count_df.head())
    print(final_count_df["region"].unique())
    print(final_count_df["product"].unique())
    final_count_df.to_csv(path / "test2.csv", index=False)

    # ^^ finding the outlet count for the whole period.  can't do it for the percentile groups because of the way im doing it
    # doing weekly percentile allocations and ros, then average over period.  outlets will be in one or more percentile groups
    # so when i do the outlet count it will be more outlets than there actually is

    # find the average quantiles over the period and pivot to merge with the results later

    agg_dict = {quantile: 'mean' for quantile in quantiles}
    reg_seg = [col for col in ("region", "segment") if col in final_result_df.columns]
    quantile_df = final_result_df.groupby(
        [
            period_column,
            "product",
            "product_type"
        ] + reg_seg, observed=True
    ).agg(agg_dict).reset_index()

    print("quantile df", "\n", quantile_df.head())
    print(quantile_df["product"].unique())

    quantile_df = pd.melt(quantile_df,
                          id_vars=[period_column, "product", "product_type"] + reg_seg,
                          value_vars=quantiles,
                          var_name='percentile',
                          value_name='percentile_price')
    print("quantile df", "\n", quantile_df.head())
    print(quantile_df["product"].unique())
    # merge the percentiles each week with the dataframe.  the all categories were created in the percentile creation, they dont
    # exist in the original data so they have to be recreated here.
    # every week, every outlet is assigned the percentile

    agg_dict = {metric: 'sum' for metric in metrics}
    agg_dict[outlet_column] = 'nunique'
    print(agg_dict)

    agg_dict2 = {f'{metric}_ros': 'mean' for metric in metrics}
    agg_dict2.update({f'sum_{metric}': 'mean' for metric in metrics})
    print(agg_dict2)

    data1 = data.copy()
    for region_column in region_columns:
        data1[region_column] = "GB"

    data2 = data.copy()
    for region_column in region_columns:
        for segment_column in segment_columns:
            data2[region_column] = "GB"
            data2[segment_column] = "All Pubs"

    data3 = data.copy()
    for segment_column in segment_columns:
        data3[segment_column] = "All Pubs"

    data_sets = [data, data1, data2, data3]
    print(data.head())
    print(final_result_df.head())
    print(products)

    quantile_results = []
    for product in products:
        for data in data_sets:
            for region_column in region_columns:
                for segment_column in segment_columns:
                    print("prod, region, seg are", "\n", product, region_column, segment_column)
                    quantile_scenarios = pd.merge(
                        final_result_df, data,
                        left_on=[date_column, period_column, "product", "region", "segment"],
                        right_on=[date_column, period_column, product, region_column, segment_column],
                        how="inner"
                    )
                    print("quantile scenarios before assign is", "\n", quantile_scenarios.head())

                    def assign_percentile(row):
                        value = row['price']
                        if value <= row[0.2]:
                            return 0.2
                        elif value <= row[0.4]:
                            return 0.4
                        elif value <= row[0.6]:
                            return 0.6
                        elif value <= row[0.8]:
                            return 0.8
                        elif value <= row[1.0]:
                            return 1
                        else:
                            return ''

                    quantile_scenarios['percentile'] = quantile_scenarios.apply(assign_percentile, axis=1)
                    print("quantile scenarios after assign is", "\n", quantile_scenarios.head())
                    quantile_results.append(quantile_scenarios)

    quantile_results = pd.concat(quantile_results, ignore_index=True)
    print("quantile results is", "\n", quantile_results.head())
    print(quantile_results["product"].unique())

    # from pathlib import Path
    #
    # path = Path("H:/0997_Fullers_Pricing_Analysis/Results")
    # quantile_results.to_csv(path / "test2.csv", index=False)

    # find ros for every percentile, every week, then average over period
    ros_df = quantile_results.groupby(
        [
            date_column,
            period_column,
            "product",
            "product_type",
            "percentile"
        ] + reg_seg, observed=True
    ).agg(agg_dict).reset_index()

    for metric in metrics:
        ros_df = ros_df.rename(columns={metric: f'sum_{metric}'})
        ros_df[f'{metric}_ros'] = ros_df[f'sum_{metric}'] / ros_df[outlet_column]
    print("ros df is", "\n", ros_df.head())
    print(ros_df["product"].unique())

    ros_df = ros_df.groupby(
        [
            period_column,
            "product",
            "product_type",
            "percentile"
        ] + reg_seg, observed=True
    ).agg(agg_dict2).reset_index()
    print("ros df is", "\n", ros_df.head())
    print(ros_df["product"].unique())

    # merge in the overall outlet counts -  the number of outlets that sell that product in that period

    ros_df = pd.merge(
        ros_df, final_count_df,
        on=[period_column,
            "product",
            "product_type"] + reg_seg,
        how="inner"
    )
    print("ros df is", "\n", ros_df.head())
    print(ros_df["product"].unique())

    # find the outlet counts for the period that are used in ros, using these for ndas
    count_df = quantile_results.groupby(
        [
            period_column,
            "product",
            "product_type",
            "percentile"
        ] + reg_seg, observed=True
    ).agg(
        {
            outlet_column: "nunique",
            data_partner_column: "nunique"
        }
    ).rename(columns={outlet_column: "outlet_count",
                      data_partner_column: "data_partner_count"}
             ).reset_index()
    print("count df is", "\n", count_df.head())
    print(count_df["product"].unique())

    # merge these counts for ndas into the results
    result_df = pd.merge(
        count_df, ros_df,
        on=[period_column,
            "product",
            "product_type",
            "percentile"] + reg_seg,
        how="inner"
    )
    print("result df is", "\n", result_df.head())
    print(result_df["product"].unique())

    dp_df = quantile_results.groupby(
        [
            period_column,
            "product",
            "product_type",
            "percentile",
            data_partner_column
        ] + reg_seg, observed=True
    ).agg(
        {
            "volume": "sum"
        }
    ).rename(columns={"volume": "dp_volume"}
             ).reset_index()
    print("dp df is", "\n", dp_df.head())
    print(dp_df["product"].unique())

    tv_df = dp_df.groupby(
        [
            period_column,
            "product",
            "product_type",
            "percentile"
        ] + reg_seg, observed=True
    ).agg(
        {
            "dp_volume": "sum"
        }
    ).rename(columns={"dp_volume": "total_dp_volume"}
             ).reset_index()
    print("tv df is", "\n", tv_df.head())
    print(tv_df["product"].unique())

    nda = pd.merge(
        dp_df, tv_df,
        on=[period_column,
            "product",
            "product_type",
            "percentile"] + reg_seg,
        how="inner"
    )
    nda["dp_volume_share"] = nda["dp_volume"] / nda["total_dp_volume"]
    print("nda is", "\n", nda.head())
    print(nda["product"].unique())

    nda["rank"] = nda.groupby(
        [period_column,
         "product",
         "product_type",
         "percentile"
         ] + reg_seg, observed=True)["dp_volume_share"].rank(ascending=False)
    nda = nda.loc[nda["rank"] == 1]
    nda = nda.rename(columns={data_partner_column: "nda_data_partner_with_max_share",
                              "dp_volume_share": "nda_max_data_partner_share"})
    print("nda is", "\n", nda.head())
    print(nda["product"].unique())

    final_result_df = pd.merge(
        result_df, nda,
        on=[period_column,
            "product",
            "product_type",
            "percentile"] + reg_seg,
        how="inner"
    )
    print("final_result_df is", "\n", final_result_df.head())
    print(final_result_df["product"].unique())

    # merge in the percentile prices
    final_result_df = pd.merge(
        quantile_df, final_result_df,
        on=[period_column,
            "product",
            "product_type",
            "percentile"] + reg_seg,
        how="left"
    )
    print("final_result_df is", "\n", final_result_df.head())
    print(final_result_df["product"].unique())

    metric_columns = ([f'{metric}_ros' for metric in metrics])

    final_result_df = final_result_df[
        [period_column, "product", "product_type", *reg_seg, "percentile", "percentile_price",
         *metric_columns, "outlet_count", "total_outlet_count", "data_partner_count",
         "nda_data_partner_with_max_share", "nda_max_data_partner_share"]]

    final_result_df = final_result_df[final_result_df["region"] != "n"]
    final_result_df = final_result_df[final_result_df["region"] != "GB"]

    final_result_df = final_result_df.loc[
        (final_result_df["outlet_count"] >= 50) &
        (final_result_df["data_partner_count"] >= 3) &
        (final_result_df["nda_max_data_partner_share"] <= 0.5)
        ]

    final_result_df = final_result_df[
        [period_column, "product", "product_type", *reg_seg, "percentile", "percentile_price",
         *metric_columns, "outlet_count", "total_outlet_count"]]
    print(final_result_df["product"].unique())

    outlet_totals = final_result_df[
        [period_column, "product", "product_type", *reg_seg, "total_outlet_count"]].drop_duplicates()

    return final_result_df, outlet_totals


@time_taken
def avg_price_function(metrics: list[str],
                       region_columns: list[str],
                       segment_columns: list[str],
                       data: pd.DataFrame,
                       date_column: str = 'date',
                       outlet_column: str = 'outlet',
                       data_partner_column: str = 'data_partner'):
    data["price"] = data["value"] / data["quantity"]

    print("data is", "\n", data.head())

    reg_seg = [col for col in region_columns + segment_columns if col in data.columns]

    agg_dict = {metric: 'sum' for metric in metrics}
    agg_dict[outlet_column] = 'nunique'
    agg_dict["price"] = "mean"
    agg_dict[data_partner_column] = "nunique"

    # ros_df = data.groupby(
    #     [
    #         date_column,
    #         "product",
    #     ] + reg_seg, observed=True
    # ).agg(agg_dict).reset_index()
    # ros_df = ros_df.rename(columns={"price": "avg_price"})
    ros = []
    if reg_seg:
        for region_column in region_columns:
            for segment_column in segment_columns:
                cols_to_add_global = [region_column, segment_column]

                ros_df = pd.concat([
                    data.groupby([
                                     date_column,
                                     "product"] + list(combo), observed=True
                                 ).agg(agg_dict).reset_index()
                    for i in range(0, 3)
                    for combo in combinations(cols_to_add_global, 2 - i)
                ])
                if region_column in reg_seg:
                    ros_df[region_column] = ros_df[region_column].cat.add_categories('GB')
                    ros_df[region_column] = ros_df[region_column].fillna("GB")
                    ros_df = ros_df.rename(columns={region_column: "region"})
                if segment_column in reg_seg:
                    ros_df[segment_column] = ros_df[segment_column].cat.add_categories('All Pubs')
                    ros_df[segment_column] = ros_df[segment_column].fillna("All Pubs")
                    ros_df = ros_df.rename(columns={segment_column: "segment"})
                ros.append(ros_df)
        ros_df = pd.concat(ros, ignore_index=True)

    else:
        ros_df = data.groupby(
            [
                date_column,
                "product",
            ], observed=True
        ).agg(agg_dict).reset_index()

    ros_df = ros_df.rename(columns={"price": "avg_price"})
    print(ros_df.head())
    print(ros_df.tail())
    print(ros_df["product"].unique())

    ros_df = ros_df.rename(columns={outlet_column: "outlet_count",
                                    data_partner_column: "data_partner_count"})
    ros_df = ros_df.drop_duplicates()

    for metric in metrics:
        ros_df = ros_df.rename(columns={metric: f'sum_{metric}'})
        ros_df[f'{metric}_ros'] = ros_df[f'sum_{metric}'] / ros_df["outlet_count"]
    print("ros df is", "\n", ros_df.head())
    print(ros_df.tail())
    print(ros_df["product"].unique())

    # dp_df = data.groupby(
    #     [
    #         date_column,
    #         "product",
    #         data_partner_column
    #     ] + reg_seg, observed=True
    # ).agg(
    #     {
    #         "volume": "sum"
    #     }
    # ).reset_index()
    # dp_df = dp_df.rename(columns={"volume": "dp_volume"})
    # print("dp_df is", "\n", dp_df.head())

    dp = []
    if reg_seg:
        for region_column in region_columns:
            for segment_column in segment_columns:
                cols_to_add_global = [region_column, segment_column]

                dp_df = pd.concat([
                    data.groupby(
                        [date_column,
                         "product",
                         data_partner_column
                         ] + list(combo), observed=True
                    ).agg({"volume": "sum"}
                          ).reset_index()
                    for i in range(0, 3)
                    for combo in combinations(cols_to_add_global, 2 - i)
                ])
                if region_column in reg_seg:
                    dp_df[region_column] = dp_df[region_column].cat.add_categories('GB')
                    dp_df[region_column] = dp_df[region_column].fillna("GB")
                    dp_df = dp_df.rename(columns={region_column: "region"})
                if segment_column in reg_seg:
                    dp_df[segment_column] = dp_df[segment_column].cat.add_categories('All Pubs')
                    dp_df[segment_column] = dp_df[segment_column].fillna("All Pubs")
                    dp_df = dp_df.rename(columns={segment_column: "segment"})
                dp.append(dp_df)
        dp_df = pd.concat(dp, ignore_index=True)

    else:
        dp_df = data.groupby(
            [date_column,
             "product",
             data_partner_column], observed=True
        ).agg({"volume": "sum"}
              ).reset_index()

    dp_df = dp_df.rename(columns={"volume": "dp_volume"})
    print("dp_df is", "\n", dp_df.head())
    print(dp_df.tail())
    print(dp_df["product"].unique())
    dp_df = dp_df.drop_duplicates()

    #
    # if reg_seg:
    #     total_vol = pd.concat([dp_df.groupby(
    #         [
    #             date_column,
    #             "product"
    #         ] + list(combo), observed=True
    #     ).agg(
    #         {
    #             "dp_volume": "sum"
    #         }
    #     ).reset_index()
    #     for i in range(0, 3)
    #     for combo in combinations(cols_to_add_global, 2 - i)
    #     ])
    #     if region_column in reg_seg:
    #         total_vol[region_column] = total_vol[region_column].cat.add_categories('GB')
    #         total_vol[region_column] = total_vol[region_column].fillna("GB")
    #     if segment_column in reg_seg:
    #         total_vol[segment_column] = total_vol[segment_column].cat.add_categories('All Pubs')
    #         total_vol[segment_column] = total_vol[segment_column].fillna("All Pubs")
    # else:
    #     total_vol = dp_df.groupby(
    #         [date_column,
    #             "product"
    #         ], observed=True
    #     ).agg(
    #         {"dp_volume": "sum"}
    #     ).reset_index()
    #
    # total_vol = total_vol.rename(columns={"dp_volume": "total_dp_volume"})
    # print("total volume is", "\n", total_vol.head())
    # print(total_vol.tail())

    reg_seg = [col for col in ("region", "segment") if col in dp_df.columns]
    total_vol = dp_df.groupby(
        [
            date_column,
            "product"
        ] + reg_seg, observed=True
    ).agg(
        {
            "dp_volume": "sum"
        }
    ).reset_index()
    total_vol = total_vol.rename(columns={"dp_volume": "total_dp_volume"})
    print("total volume is", "\n", total_vol.head())
    print(total_vol.tail())
    print(total_vol["product"].unique())

    nda = pd.merge(
        dp_df, total_vol,
        on=[date_column,
            "product"] + reg_seg,
        how="inner"
    )
    nda["dp_volume_share"] = nda["dp_volume"] / nda["total_dp_volume"]
    print("nda is", "\n", nda.head())
    print(nda.tail())
    print(nda["product"].unique())

    nda["rank"] = nda.groupby(
        [date_column,
         "product"] + reg_seg, observed=True)["dp_volume_share"].rank(ascending=False)
    nda = nda.loc[nda["rank"] == 1]
    nda = nda.rename(columns={data_partner_column: "nda_data_partner_with_max_share",
                              "dp_volume_share": "nda_max_data_partner_share"})
    print("nda is", "\n", nda.head())
    print(nda["product"].unique())

    results = pd.merge(
        ros_df, nda,
        on=[date_column,
            "product"] + reg_seg,
        how="inner"
    )
    print("results is", "\n", results.head())
    print(results["product"].unique())

    metric_columns = ([f'{metric}_ros' for metric in metrics])

    results = results[
        [date_column, "product", *reg_seg, "avg_price", *metric_columns, "outlet_count", "data_partner_count",
         "nda_data_partner_with_max_share", "nda_max_data_partner_share"]]

    results = results[results["region"] != "n"]
    results = results[results["region"] != "GB"]

    results = results.loc[
        (results["outlet_count"] >= 50) &
        (results["data_partner_count"] >= 3) &
        (results["nda_max_data_partner_share"] <= 0.5)
        ]

    results = results[[date_column, "product", *reg_seg, "avg_price", *metric_columns, "outlet_count"]]
    print(results["product"].unique())

    return results


def main():
    group_columns = [
        'date',
        'region1',
        'region2',
        'region3',
        'region4',
        'segment'
    ]

    metrics = [
        'volume',
        'value'
    ]

    data = pull_data(file_name='data_0997.parquet', pull=read_in_data)
    print(data.head())

    # from pathlib import Path
    #
    # path = Path("H:/0997_Fullers_Pricing_Analysis/Code")
    # data.to_csv(path / "data.csv", index=False)

    products_pint = ['Draught Fosters Pint', 'Draught Carling Pint', 'Draught Carlsberg Danish Pilsner Pint', 'Draught Amstel Pint',
                     'Draught Cruzcampo Pint', 'Draught Pravha Pint', 'Draught Heineken Original Pint', 'Draught San Miguel Pint',
                     'Draught Madri Excepcional Pint', 'Draught Peroni Nastro Azzurro Pint', 'Draught Asahi Super Dry Pint',
                     'Draught Birra Moretti Pint', 'Draught Estrella Damm Pint',  'Draught Camden Hells Lager Pint',
                    'Draught Camden Town Pale Ale Pint',
                     'Draught Beavertown Neck Oil Session IPA Pint',  'Draught Tiny Rebel Easy Livin Pint',
                     'Draught Brewdog Punk IPA Pint', 'Draught Guinness Pint', 'Draught Cornish Orchards Cornish Gold Cider Pint',
                     'Draught Aspall Cyder Pint', 'Draught Fullers London Pride (Cask) Pint', 'Draught Sharps Doom Bar (Cask) Pint',
                     'Draught St Austell Tribute Ale (Cask) Pint', 'Draught Timothy Taylors Landlord (Cask) Pint',
                     'Draught Diet Pepsi Pint', 'Draught Pepsi Pint', 'Draught Pepsi Max Pint', 'Draught Diet Coke Pint',
                     'Draught Coke Zero Pint',  'Draught Coca-Cola Pint']
    products_330 = ['Packaged Peroni Nastro Azzurro 330ml',  'Packaged Heineken Original 330ml', 'Packaged Sol 330ml',
                    'Packaged Peroni Nastro Azzurro 0.0 330ml',  'Packaged Corona Extra 330ml'] #20
    products_25 = ['Tanqueray 25ml', 'Beefeater 25ml', 'Smirnoff Black 25ml', 'Smirnoff Red 25ml', 'Absolut Blue 25ml',
                   'Bacardi Carta Negra 25ml', 'Bacardi Carta Blanca 25ml', 'Kraken Black Spiced Rum 25ml',
                   'Havana Club 3 Year Old 25ml','Jack Daniels 25ml', 'Johnnie Walker Black Label 12 Year Old 25ml',
                   'Johnnie Walker Red Label 25ml', 'Jameson 25ml']
    products_200 = ["Packaged Any Fever Tree Tonic 200ml"]
    products_500 = ["Packaged Any Full Alcohol Old Mout 500ml", "Packaged Any Full Alcohol Rekorderlig 500ml"]

    print("before removing outliers", data.head(), data.shape)
    data = remove_outliers(data=data, group_columns=group_columns + ['product'], metrics=metrics,
                           pint_products=products_pint,
                           products_330=products_330,
                           products_25=products_25,
                           products_200=products_200,
                           products_500=products_500)
    print("after removing outliers", data.head(), data.shape)

    # print("before cont", data.shape)
    # data = get_continuous_items(
    #     data=data,
    #     date_count_threshold=3,
    #     item_column="product"
    # )
    # print("after cont", data.shape)

    products = data["product"].unique()
    print(products)
    print(data["segment"].unique())

    prices, outlet_totals = price_banding_function(products=["product"],
                                                   brands=products,
                                                   metrics=metrics,
                                                   quantiles=[0.2, 0.4, 0.6, 0.8, 1.0],
                                                   data=data,
                                                   period_column="qtr",
                                                   period_filter=[1, 5],
                                                   region_columns=["region1", "region2", "region3", "region4"],
                                                   segment_columns=["segment"])

    regions = prices[["region"]].drop_duplicates().reset_index(drop=True)
    segments = prices[["segment"]].drop_duplicates().reset_index(drop=True)
    products = prices[["product"]].drop_duplicates().reset_index(drop=True)
    lists = pd.concat([regions, segments, products], axis=1)

    file_path = "H:/0997_Fullers_Pricing_Analysis/Results/price_results.xlsx"

    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        prices.to_excel(writer, sheet_name='PC_Data', index=False)
        lists.to_excel(writer, sheet_name='lists', index=False)
        outlet_totals.to_excel(writer, sheet_name='OT_Data', index=False)
    print(prices.head())

    avg = avg_price_function(metrics=metrics,
                             data=data,
                             region_columns=["region1", "region2", "region3", "region4"],
                             segment_columns=["segment"]
                             )
    print(avg.head())

    regions = avg[["region"]].drop_duplicates().reset_index(drop=True)
    segments = avg[["segment"]].drop_duplicates().reset_index(drop=True)
    products = avg[["product"]].drop_duplicates().reset_index(drop=True)
    dates = avg[["date"]].drop_duplicates().reset_index(drop=True)

    lists = pd.concat([regions, segments, products, dates], axis=1)

    file_path = "H:/0997_Fullers_Pricing_Analysis/Results/avg_results.xlsx"

    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        avg.to_excel(writer, sheet_name='avg', index=False)
        lists.to_excel(writer, sheet_name='lists', index=False)


if __name__ == '__main__':
    main()


     # NEED TO HAVE MULTIPLE REGION COLUMNS
    # dotn know how the final gb step is going to work out now./  maybe do it every time adn drop dupes
#outliers used to work and now since ive messed with ergion it doesnt