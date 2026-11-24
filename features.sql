-- features.sql
-- Builds the DuckDB tables used by Late_delivery_prediction_annotated.ipynb.
-- Every statement below is copied from the notebook, with two mechanical changes:
--   * CSV paths point to data/ (the notebook's load cell must point to the same place);
--   * the Python variable K_PRIOR is replaced by its value, 20 (shrinkage strength).
--
-- Order: 0. load the CSVs, 1. create the haversine_km macro, 2. features_base, 3. order_features_v2.
-- Example (DuckDB CLI):  duckdb olist.duckdb < features.sql
--
-- Not in this file (done in the notebook with pandas, after reading order_features_v2):
--   dropping purchases before 2017-01, the distance_unknown and new_seller flags, the log1p transform
--   and the time-based train/test split. The notebook reads the table with:
--   SELECT * FROM order_features_v2 ORDER BY order_purchase_timestamp, order_id

-- 0. Load the raw CSVs (save the Olist CSVs from Kaggle in data/)
CREATE OR REPLACE TABLE orders AS
SELECT * FROM read_csv_auto('data/olist_orders_dataset.csv');

CREATE OR REPLACE TABLE order_items AS
SELECT * FROM read_csv_auto('data/olist_order_items_dataset.csv');

CREATE OR REPLACE TABLE customers AS
SELECT * FROM read_csv_auto('data/olist_customers_dataset.csv');

CREATE OR REPLACE TABLE sellers AS
SELECT * FROM read_csv_auto('data/olist_sellers_dataset.csv');

CREATE OR REPLACE TABLE products AS
SELECT * FROM read_csv_auto('data/olist_products_dataset.csv');

CREATE OR REPLACE TABLE geolocation AS
SELECT * FROM read_csv_auto('data/olist_geolocation_dataset.csv');

CREATE OR REPLACE TABLE category_translation AS
SELECT * FROM read_csv_auto('data/product_category_name_translation.csv');

-- 1. Haversine distance in km (NULL if any coordinate is missing).
-- The notebook does not create this macro; it only exists in the .duckdb file it was run against.
CREATE OR REPLACE MACRO haversine_km(lat1, lng1, lat2, lng2) AS (
    CASE WHEN lat1 IS NULL OR lng1 IS NULL OR lat2 IS NULL OR lng2 IS NULL THEN NULL
    ELSE 2 * 6371.0 * ASIN(LEAST(1.0, SQRT(
            POWER(SIN(RADIANS(lat2 - lat1) / 2), 2) +
            COS(RADIANS(lat1)) * COS(RADIANS(lat2)) *
            POWER(SIN(RADIANS(lng2 - lng1) / 2), 2)))) END
);

-- 2. features_base: one row per order item (delivered orders only)
create or replace table features_base as 
with geolocationv2 as 
(
select geolocation_zip_code_prefix, MEDIAN(geolocation_lat) as geolocation_lat, MEDIAN(geolocation_lng) as geolocation_lng from geolocation
group by 1 
)

select a.order_id,a.customer_id, a.order_purchase_timestamp, date(a.order_delivered_customer_date) as delivered_date, a.order_estimated_delivery_date,
b.order_item_id, b.product_id, b.seller_id,b.price, b.freight_value,c.customer_unique_id,customer_zip_code_prefix, c.customer_city, c.customer_state,
s.seller_zip_code_prefix, s.seller_city, s.seller_state, p.product_category_name,
p.product_weight_g, p.product_length_cm, p.product_height_cm, p.product_width_cm,
g1.geolocation_lat as seller_lat, g1.geolocation_lng as seller_lng,
g2.geolocation_lat as buyer_lat, g2.geolocation_lng as buyer_lng
 
from orders a 
left join order_items b 
on a.order_id = b.order_id

left join customers c 
on a.customer_id = c.customer_id

left join sellers s 
on b.seller_id = s.seller_id

left join products p 
on p.product_id = b.product_id

left join geolocationv2 g1 
on s.seller_zip_code_prefix = g1.geolocation_zip_code_prefix

left join geolocationv2 g2 
on c.customer_zip_code_prefix = g2.geolocation_zip_code_prefix

where a.order_status = 'delivered' and a.order_delivered_customer_date is not null and a.order_estimated_delivery_date is not null;

-- 3. order_features_v2: one row per delivered order (label + purchase-time features), K_PRIOR = 20
CREATE OR REPLACE TABLE order_features_v2 AS
-- Step 1 (order_agg): collapse item rows to one row per order (grain = order_id and month).
WITH order_agg AS (
    SELECT
        order_id,
        YEAR(order_purchase_timestamp) * 100 + MONTH(order_purchase_timestamp) as month,
        -- primary seller, its location and state = those of the highest-priced item; MAX() picks one value for order-level attributes
        ARG_MAX(seller_id, price) AS primary_seller_id,
        MAX(customer_zip_code_prefix) AS customer_zip_code_prefix,
        MAX(order_purchase_timestamp) AS order_purchase_timestamp,
        CAST(MAX(order_purchase_timestamp) AS DATE) AS order_date,
        MAX(delivered_date) AS delivered_date,
        MAX(order_estimated_delivery_date) AS order_estimated_delivery_date,
        ARG_MAX(seller_state, price) AS primary_seller_state,
        MAX(customer_state) AS customer_state,
        ARG_MAX(seller_lat, price) AS seller_lat,
        ARG_MAX(seller_lng, price) AS seller_lng,
        MAX(buyer_lat) AS buyer_lat,
        MAX(buyer_lng) AS buyer_lng,
        -- basket size: items, sellers and distinct products
        1 AS order_count,
        COUNT(DISTINCT order_item_id) AS item_count,
        COUNT(DISTINCT seller_id) AS seller_count,
        COUNT(DISTINCT product_id) AS product_count,
        SUM(price) AS total_price,
        SUM(freight_value) AS total_freight,
        -- freight as a share of price (0 when the price is 0)
        CASE WHEN SUM(price) > 0 THEN SUM(freight_value) / SUM(price) ELSE 0 END AS freight_ratio,
        -- product size features: totals, maxima and averages over the items
        SUM(product_weight_g) AS total_weight,
        MAX(product_weight_g) AS max_weight,
        MAX(product_length_cm) AS max_length,
        MAX(product_height_cm) AS max_height,
        MAX(product_width_cm) AS max_width,
        AVG(product_length_cm) AS avg_length,
        AVG(product_height_cm) AS avg_height,
        AVG(product_width_cm) AS avg_width,
        SUM(product_length_cm * product_height_cm * product_width_cm) AS total_volume,
        MAX(product_length_cm * product_height_cm * product_width_cm) AS max_volume,
        COUNT(DISTINCT product_category_name) AS category_count,
        -- category of the highest-priced item, unknown when missing
        COALESCE(ARG_MAX(product_category_name, price), 'unknown') AS top_category,
        -- LABEL: 1 if delivered after the estimated date (an outcome, never a feature)
        CASE WHEN MAX(delivered_date) > MAX(order_estimated_delivery_date) THEN 1 ELSE 0 END AS late_delivery_flag,
        -- promised window in days (known at purchase time)
        DATEDIFF('day', CAST(MAX(order_purchase_timestamp) AS DATE), MAX(order_estimated_delivery_date)) AS eta_days,
        CASE WHEN ARG_MAX(seller_state, price) = MAX(customer_state) THEN 1 ELSE 0 END AS same_state_flag,
        HOUR(MAX(order_purchase_timestamp)) AS order_hour,
        DAYOFWEEK(MAX(order_purchase_timestamp)) AS order_day_of_week,
        -- seller-to-buyer distance in km (NULL if any coordinate is missing)
        haversine_km(
            ARG_MAX(seller_lat, price), ARG_MAX(seller_lng, price),
            MAX(buyer_lat), MAX(buyer_lng)
        ) AS distance_km,
        -- distance divided by the promised days
        case when DATEDIFF('day', CAST(MAX(order_purchase_timestamp) AS DATE), MAX(order_estimated_delivery_date)) > 0 then
        haversine_km(
            ARG_MAX(seller_lat, price), ARG_MAX(seller_lng, price),
            MAX(buyer_lat), MAX(buyer_lng)
        ) / DATEDIFF('day', CAST(MAX(order_purchase_timestamp) AS DATE), MAX(order_estimated_delivery_date))

        else haversine_km(
            ARG_MAX(seller_lat, price), ARG_MAX(seller_lng, price),
            MAX(buyer_lat), MAX(buyer_lng)
        ) end as distance_per_day,
        -- 1 if any of the four coordinates is missing
        max(case when seller_lat is null or seller_lng is null or buyer_lat is null or buyer_lng is null then 1 else 0 end) as seller_or_buyer_loc_missing

    FROM features_base
    GROUP BY 1,2
),
-- Step 2 (order_with_history): history from EARLIER orders only. 1 PRECEDING excludes the current row,
-- and the 7-day windows end yesterday.
order_with_history AS (
    SELECT
        *,
        COALESCE(SUM(order_count) OVER (
            PARTITION BY primary_seller_id
            ORDER BY order_purchase_timestamp, order_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS seller_past_orders,
        AVG(eta_days) OVER (
            PARTITION BY primary_seller_id
            ORDER BY order_purchase_timestamp, order_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS seller_avg_eta,
        COALESCE(SUM(order_count) OVER (
            PARTITION BY primary_seller_id
            ORDER BY order_date
            RANGE BETWEEN INTERVAL '7 days' PRECEDING AND INTERVAL '1 day' PRECEDING), 0) AS seller_7d_rolling_orders,
        COALESCE(SUM(order_count) OVER (
            ORDER BY order_date
            RANGE BETWEEN INTERVAL '7 days' PRECEDING AND INTERVAL '1 day' PRECEDING), 0) AS platform_7d_rolling_orders,
        COALESCE(SUM(order_count) OVER (
            PARTITION BY customer_zip_code_prefix
            ORDER BY order_purchase_timestamp, order_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS cust_code_past_orders,
        COALESCE(SUM(order_count) OVER (
            PARTITION BY top_category
            ORDER BY order_purchase_timestamp, order_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS category_past_orders
    FROM order_agg
),
-- Step 3 (*_cum): cumulative delivered and late counts per entity, indexed by DELIVERY date,
-- because an outcome is only known on the day the order is delivered.
seller_cum AS (
    SELECT k, delivered_date,
           SUM(n)      OVER (PARTITION BY k ORDER BY delivered_date) AS cum_n,
           SUM(n_late) OVER (PARTITION BY k ORDER BY delivered_date) AS cum_late
    FROM (SELECT primary_seller_id AS k, delivered_date,
                 COUNT(*) AS n, SUM(late_delivery_flag) AS n_late
          FROM order_agg GROUP BY 1, 2)
),
zip_cum AS (
    SELECT k, delivered_date,
           SUM(n)      OVER (PARTITION BY k ORDER BY delivered_date) AS cum_n,
           SUM(n_late) OVER (PARTITION BY k ORDER BY delivered_date) AS cum_late
    FROM (SELECT customer_zip_code_prefix AS k, delivered_date,
                 COUNT(*) AS n, SUM(late_delivery_flag) AS n_late
          FROM order_agg GROUP BY 1, 2)
),
cat_cum AS (
    SELECT k, delivered_date,
           SUM(n)      OVER (PARTITION BY k ORDER BY delivered_date) AS cum_n,
           SUM(n_late) OVER (PARTITION BY k ORDER BY delivered_date) AS cum_late
    FROM (SELECT top_category AS k, delivered_date,
                 COUNT(*) AS n, SUM(late_delivery_flag) AS n_late
          FROM order_agg GROUP BY 1, 2)
),
state_cum AS (
    SELECT k, delivered_date,
           SUM(n)      OVER (PARTITION BY k ORDER BY delivered_date) AS cum_n,
           SUM(n_late) OVER (PARTITION BY k ORDER BY delivered_date) AS cum_late
    FROM (SELECT customer_state AS k, delivered_date,
                 COUNT(*) AS n, SUM(late_delivery_flag) AS n_late
          FROM order_agg GROUP BY 1, 2)
),
-- platform-wide version: constant key 1 so all orders share one series
platform_cum AS (
    SELECT 1 AS k, delivered_date,
           SUM(n)      OVER (ORDER BY delivered_date) AS cum_n,
           SUM(n_late) OVER (ORDER BY delivered_date) AS cum_late
    FROM (SELECT delivered_date, COUNT(*) AS n, SUM(late_delivery_flag) AS n_late
          FROM order_agg GROUP BY 1)
),
-- Step 4 (daily): delivered and late orders per delivery day, used for the trailing 7/14-day platform features.
daily AS (   -- deliveries and late deliveries per delivery day, for the recent platform late rate
    SELECT delivered_date AS d, COUNT(*) AS n, SUM(late_delivery_flag) AS n_late
    FROM order_agg
    GROUP BY 1
),
-- Step 5 (joined): ASOF join = latest cumulative row strictly BEFORE the order date,
-- so no same-day or future outcome leaks in. Missing history becomes 0.
joined AS (
    SELECT
        b.*,
        COALESCE(s.cum_n, 0)     AS seller_hist_delivered,
        COALESCE(s.cum_late, 0)  AS s_late,
        COALESCE(z.cum_n, 0)     AS zip_hist_delivered,
        COALESCE(z.cum_late, 0)  AS z_late,
        COALESCE(c.cum_n, 0)     AS cat_hist_delivered,
        COALESCE(c.cum_late, 0)  AS c_late,
        COALESCE(st.cum_n, 0)    AS st_n,
        COALESCE(st.cum_late, 0) AS st_late,
        COALESCE(p.cum_n, 0)     AS p_n,
        COALESCE(p.cum_late, 0)  AS p_late
    FROM (SELECT *, 1 AS plat_k FROM order_with_history) b
    ASOF LEFT JOIN seller_cum   s  ON b.primary_seller_id = s.k        AND b.order_date > s.delivered_date
    ASOF LEFT JOIN zip_cum      z  ON b.customer_zip_code_prefix = z.k AND b.order_date > z.delivered_date
    ASOF LEFT JOIN cat_cum      c  ON b.top_category = c.k             AND b.order_date > c.delivered_date
    ASOF LEFT JOIN state_cum    st ON b.customer_state = st.k          AND b.order_date > st.delivered_date
    ASOF LEFT JOIN platform_cum p  ON b.plat_k = p.k                   AND b.order_date > p.delivered_date
),
-- Step 6 (rates): platform rate, and the state rate shrunk toward the platform rate.
rates AS (
    SELECT *,
        p_late * 1.0 / NULLIF(p_n, 0) AS platform_rate,
        (st_late + 20 * (p_late * 1.0 / NULLIF(p_n, 0))) / (st_n + 20) AS state_hist_late_rate
    FROM joined
)
SELECT
    -- Step 7 (final select): drop helper columns and shrink each rate: (late + K * parent rate) / (n + K), with K = K_PRIOR.
    -- Parents: seller and category -> platform, zip -> state, state -> platform.
    * EXCLUDE (plat_k, s_late, z_late, c_late, st_n, st_late, p_n, p_late, platform_rate),
    (s_late + 20 * platform_rate)         / (seller_hist_delivered + 20) AS seller_hist_late_rate,
    (z_late + 20 * state_hist_late_rate)  / (zip_hist_delivered + 20)    AS zip_hist_late_rate,
    (c_late + 20 * platform_rate)         / (cat_hist_delivered + 20)    AS cat_hist_late_rate,

    -- recent platform late rate: only orders delivered on days strictly before this order's date
    (SELECT COALESCE(SUM(n), 0) FROM daily
      WHERE d BETWEEN r.order_date - 14 AND r.order_date - 1) AS platform_delivered_14d,
    (SELECT SUM(n_late) * 1.0 / NULLIF(SUM(n), 0) FROM daily
      WHERE d BETWEEN r.order_date - 14 AND r.order_date - 1) AS platform_late_rate_14d,
    (SELECT COALESCE(SUM(n), 0) FROM daily
      WHERE d BETWEEN r.order_date - 7 AND r.order_date - 1)  AS platform_delivered_7d,
    (SELECT SUM(n_late) * 1.0 / NULLIF(SUM(n), 0) FROM daily
      WHERE d BETWEEN r.order_date - 7 AND r.order_date - 1)  AS platform_late_rate_7d
FROM rates r;
