import time

import pandas as pd
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
         select 
      D_datekey as date,
      rel_qtr as qtr,
	  d_period as period,
      d_yr as year,
      OT_CGAIdent as outlet,
	  PT_ProductDescription as product,
      OT_TL2_Novellus as region,
      OT_SL2_Novellus as segment,
	  OT_Quality_CGA as quality,
      sum([F_SalesValue_£]) as value,
      sum(F_SalesVolume_MLS) as volume
	  ,sum(F_SalesQuantity) as quantity,
	  PI_ClientDescription as data_partner

FROM [WS_LIVE].[dbo].[vw_Epos_Weekly]

    where
	D_DateKey >= '2021-12-04 00:00:00.000'  -- 3 years ago
	AND OT_SL2_Novellus IN ('Food Pub', 'High Street Pub', 'Community Pub')
	AND PT_ProductDescription in (
'Sol',
'Peroni Nastro Azzurro',
'Amstel',
'Baileys',
'Guinness',
'Brown Bag Crisps',
'Chocolate Ice Cream',
'Raspberry Ripple',
'Drambuie',
'La Palma Chardonnay',
'Brown Bag Salt & Vinegar',
'Brown Bag Cheese & Onion',
'Tea')
 AND OT_CGAIdent > 0
 AND PT_ProductId > 0
 AND F_SalesVolume_MLS > 0
 AND [F_SalesValue_£] > 0
 AND F_SalesQuantity > 0
 AND OT_TL5_ISBA = 'GB'


    group by D_datekey,
	d_period,
	rel_qtr,
      d_yr,
      OT_CGAIdent,
	  PT_ProductDescription,
      OT_TL2_Novellus,
      OT_SL2_Novellus,
	  OT_Quality_CGA,
	   PI_ClientDescription
            ''', engine)

    data['date'] = pd.to_datetime(data['date'])
    data = data.astype({
        'period': 'uint32',
        'qtr':'uint32',
        'year': 'uint32',
        'outlet': 'category',
        'product': 'category',
        'region': 'category',
        'segment': 'category',
        'quality': 'category',
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
        metrics: list[str] = None,
        st_devs: int = 3,
        units: int = None
):
    data = data.dropna()
    print("shape is", data.shape)

    if metrics is None:
        metrics = ['value', 'volume']

    print(
"units is", units
    )
    print(data.head())
    if units is not None:
        data["volume"] = data["volume"] / units
    print(data.head())

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
                           data: pd.DataFrame,
                           period_filter: list[int],
                           date_column: str = 'date',
                           period_column: str = 'period',
                           outlet_column: str = 'outlet',
                           region_column: str = 'region',
                           segment_column: str = 'segment',
                           data_partner_column: str = 'data_partner'):

    data["price"] = data["value"] / data["quantity"]
    print("data is", "\n", data.head())
    data = data[data[period_column].isin(period_filter)]
    print("data is", "\n", data.head())

    reg_seg = [col for col in (region_column, segment_column) if col in data.columns]

    results = []
    counts = []
    for product in products:
        for brand in brands:
            filtered_df = data[data[product] == brand]
            #print("product is", brand, "filtered data is", "\n", filtered_df.head())

            group_columns = [date_column, period_column, product]
            if reg_seg:
                cols_to_add_global = [region_column, segment_column]

                result_df = pd.concat([
                    filtered_df.groupby(list(group_columns) + list(combo), observed=True)["price"].quantile(
                        quantiles).unstack().reset_index()
                    for i in range(0, 3)
                    for combo in combinations(cols_to_add_global, 2 - i)
                    # do widest df first to make sure columns are created
                ])  # 2-i means we start with 2, then 1 then 0.

                result_df = result_df.rename(columns={product: "product"})

                #print("result df is", "\n", result_df.head())

                result_df["product_type"] = product
                result_df[region_column] = result_df[region_column].cat.add_categories('GB')
                result_df[region_column] = result_df[region_column].fillna('GB')
                result_df[segment_column] = result_df[segment_column].cat.add_categories('All Pubs')
                result_df[segment_column] = result_df[segment_column].fillna('All Pubs')

            else:
                cols_to_add_global = [region_column, segment_column]

                result_df = filtered_df.groupby(list(group_columns), observed=True)["price"].quantile(
                        quantiles).unstack().reset_index()

                result_df = result_df.rename(columns={product: "product"})

                #print("result df is", "\n", result_df.head())

                result_df["product_type"] = product

            results.append(result_df)

            group_columns = [period_column, product]

            if reg_seg:
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

                #print("result df is", "\n", count_df.head())

                count_df["product_type"] = product
                count_df[region_column] = count_df[region_column].cat.add_categories('GB')
                count_df[region_column] = count_df[region_column].fillna('GB')
                count_df[segment_column] = count_df[segment_column].cat.add_categories('All Pubs')
                count_df[segment_column] = count_df[segment_column].fillna('All Pubs')
            else:
                count_df = filtered_df.groupby(
                        list(group_columns), observed=True
                    ).agg({outlet_column: "nunique"}).reset_index()

                count_df = count_df.rename(columns={product: "product",
                                                    outlet_column: "total_outlet_count"})

                #print("result df is", "\n", count_df.head())

                count_df["product_type"] = product

            counts.append(count_df)

    final_result_df = pd.concat(results, ignore_index=True)
    from pathlib import Path

    path = Path("H:/Fullers_Pricing_Analysis/Results")
    final_result_df.to_csv(path / "test1.csv", index=False)

    final_count_df = pd.concat(counts, ignore_index=True)
    print("final result df is", "\n", final_result_df.head())
    print("final count df is", "\n", final_count_df.head())
    print(final_count_df["product"].unique())

    # ^^ finding the outlet count for the whole period.  can't do it for the percentile groups because of the way im doing it
    # doing weekly percentile allocations and ros, then average over period.  outlets will be in one or more percentile groups
    # so when i do the outlet count it will be more outlets than there actually is

    # find the average quantiles over the period and pivot to merge with the results later

    agg_dict = {quantile: 'mean' for quantile in quantiles}
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
    data1[region_column] = "GB"

    data2 = data.copy()
    data2[region_column] = "GB"
    data2[segment_column] = "All Pubs"

    data3 = data.copy()
    data3[segment_column] = "All Pubs"

    data_sets = [data, data1, data2, data3]

    quantile_results = []
    for product in products:
        for data in data_sets:
            quantile_scenarios = pd.merge(
                final_result_df, data,
                left_on=[date_column, period_column, "product"] + reg_seg,
                right_on=[date_column, period_column, product] + reg_seg,
                how="inner"
            )

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
            quantile_results.append(quantile_scenarios)

    quantile_results = pd.concat(quantile_results, ignore_index=True)
    print("quantile results is", "\n", quantile_results.head())
    print(quantile_results["product"].unique())

    from pathlib import Path

    path = Path("H:/Fullers_Pricing_Analysis/Results")
    quantile_results.to_csv(path / "test2.csv", index=False)

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

    final_result_df = final_result_df[[period_column, "product", "product_type", *reg_seg, "percentile", "percentile_price",
                                       *metric_columns, "outlet_count", "total_outlet_count", "data_partner_count",
                                       "nda_data_partner_with_max_share","nda_max_data_partner_share"]]

    # final_result_df = final_result_df.loc[
    #     (final_result_df["outlet_count"] >= 50) &
    #     (final_result_df["data_partner_count"] >= 3) &
    #     (final_result_df["nda_max_data_partner_share"] <= 0.5)
    #     ]

    final_result_df = final_result_df[[period_column, "product", "product_type", *reg_seg, "percentile", "percentile_price",
                                       *metric_columns, "outlet_count", "total_outlet_count"]]
    print(final_result_df["product"].unique())

    outlet_totals = final_result_df[[period_column, "product", "product_type", *reg_seg, "total_outlet_count"]].drop_duplicates()

    return final_result_df, outlet_totals



@time_taken
def avg_price_function(metrics: list[str],
                           data: pd.DataFrame,
                           date_column: str = 'date',
                           outlet_column: str = 'outlet',
                           region_column: str = 'region',
                           segment_column: str = 'segment',
                           data_partner_column: str = 'data_partner'):

    data["price"] = data["value"] / data["quantity"]

    print("data is", "\n", data.head())

    reg_seg = [col for col in (region_column, segment_column) if col in data.columns]

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
    cols_to_add_global = reg_seg
    if reg_seg:
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
        if segment_column in reg_seg:
            ros_df[segment_column] = ros_df[segment_column].cat.add_categories('All Pubs')
            ros_df[segment_column] = ros_df[segment_column].fillna("All Pubs")
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

    if reg_seg:
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
        if segment_column in reg_seg:
            dp_df[segment_column] = dp_df[segment_column].cat.add_categories('All Pubs')
            dp_df[segment_column] = dp_df[segment_column].fillna("All Pubs")
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

    results = results[[date_column, "product", *reg_seg, "avg_price", *metric_columns, "outlet_count", "data_partner_count",
    "nda_data_partner_with_max_share", "nda_max_data_partner_share"]]

    # results = results.loc[
    #     (results["outlet_count"] >= 50) &
    #     (results["data_partner_count"] >= 3) &
    #     (results["nda_max_data_partner_share"] <= 0.5)
    #     ]

    results = results[[date_column, "product", *reg_seg, "avg_price", *metric_columns, "outlet_count"]]
    print(results["product"].unique())

    return results


def main():

    group_columns = [
        'date',
        'region',
        'segment',
        'quality'
    ]

    metrics = [
        'volume',
        'value'
    ]

    products = [
        "product"
    ]

    data = pull_data(file_name='data.parquet', pull=read_in_data)
    print(data.head())

    from pathlib import Path

    path = Path("H:/Fullers_Pricing_Analysis/Code")
    data.to_csv(path / "data.csv", index=False)


    print("before removing outliers", data.head(), data.shape)
    data = remove_outliers(data=data, group_columns=group_columns + ['product'], metrics=metrics, units=9000)
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
                                    period_filter=[1,5],
                       region_column="region",
                       segment_column="segment")

    regions = prices[["region"]].drop_duplicates().reset_index(drop=True)
    segments = prices[["segment"]].drop_duplicates().reset_index(drop=True)
    products = prices[["product"]].drop_duplicates().reset_index(drop=True)
    lists = pd.concat([regions, segments, products], axis=1)

    file_path = "H:/Fullers_Pricing_Analysis/Results/price_results.xlsx"

    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        prices.to_excel(writer, sheet_name='PC_Data', index=False)
        lists.to_excel(writer, sheet_name='lists', index=False)
        outlet_totals.to_excel(writer, sheet_name='OT_Data', index=False)
    print(prices.head())



    avg = avg_price_function(metrics=metrics,
                           data=data)
    print(avg.head())


    regions = avg[["region"]].drop_duplicates().reset_index(drop=True)
    segments = avg[["segment"]].drop_duplicates().reset_index(drop=True)
    products = avg[["product"]].drop_duplicates().reset_index(drop=True)
    dates = avg[["date"]].drop_duplicates().reset_index(drop=True)

    lists = pd.concat([regions, segments, products, dates], axis=1)

    file_path = "H:/Fullers_Pricing_Analysis/Results/avg_results.xlsx"

    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        avg.to_excel(writer, sheet_name='avg', index=False)
        lists.to_excel(writer, sheet_name='lists', index=False)

    from pathlib import Path

    path = Path("H:/Fullers_Pricing_Analysis/Results")
    avg.to_csv(path / "avg.csv", index=False)


if __name__ == '__main__':
    main()