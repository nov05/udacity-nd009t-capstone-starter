
### 👉 **Drop table**

```SQL
DROP TABLE IF EXISTS table_metadata;
DROP TABLE IF EXISTS table_metadata_flat;
DROP TABLE IF EXISTS table_metadata_consolidated;
```

### 👉 **Created flattened table**  

* Some records are missing in this case, for one JSON file might contain multiple records.  

```SQL
CREATE EXTERNAL TABLE table_metadata_flat (
  asin STRING,
  name STRING,
  normalized_name STRING,
  quantity INT,
  height_value DECIMAL(20,18),
  height_unit STRING,
  length_value DECIMAL(20,18),
  length_unit STRING,
  width_value DECIMAL(20,18),
  width_unit STRING,
  weight_value DECIMAL(20,18),
  weight_unit STRING
)
ROW FORMAT SERDE 'com.amazon.ionhiveserde.IonHiveSerDe'
WITH SERDEPROPERTIES (
  "ion.asin.path_extractor" = "(BIN_FCSKU_DATA * asin)",
  "ion.name.path_extractor" = "(BIN_FCSKU_DATA * name)",
  "ion.normalized_name.path_extractor" = "(BIN_FCSKU_DATA * normalizedName)",
  "ion.quantity.path_extractor" = "(BIN_FCSKU_DATA * quantity)",
  "ion.height_value.path_extractor" = "(BIN_FCSKU_DATA * height value)",
  "ion.height_unit.path_extractor" = "(BIN_FCSKU_DATA * height unit)",
  "ion.length_value.path_extractor" = "(BIN_FCSKU_DATA * length value)",
  "ion.length_unit.path_extractor" = "(BIN_FCSKU_DATA * length unit)",
  "ion.width_value.path_extractor" = "(BIN_FCSKU_DATA * width value)",
  "ion.width_unit.path_extractor" = "(BIN_FCSKU_DATA * width unit)",
  "ion.weight_value.path_extractor" = "(BIN_FCSKU_DATA * weight value)",
  "ion.weight_unit.path_extractor" = "(BIN_FCSKU_DATA * weight unit)"
)
STORED AS
  INPUTFORMAT 'com.amazon.ionhiveserde.formats.IonInputFormat'
  OUTPUTFORMAT 'com.amazon.ionhiveserde.formats.IonOutputFormat'
LOCATION 's3://dataset-aft-vbi-pds/metadata';
```

### 👉 **Created nested table with MAP function**

```SQL
CREATE EXTERNAL TABLE table_metadata (
  bin_fcsku_data MAP<STRING, STRUCT<
    asin: STRING,
    name: STRING,
    normalizedname: STRING,
    quantity: INT,
    height: STRUCT<unit: STRING, value: DECIMAL(20,18)>,
    length: STRUCT<unit: STRING, value: DECIMAL(20,18)>,
    width: STRUCT<unit: STRING, value: DECIMAL(20,18)>,
    weight: STRUCT<unit: STRING, value: DECIMAL(20,18)>
  >>,
  expected_quantity INT
)
ROW FORMAT SERDE 'com.amazon.ionhiveserde.IonHiveSerDe'
STORED AS
  INPUTFORMAT 'com.amazon.ionhiveserde.formats.IonInputFormat'
  OUTPUTFORMAT 'com.amazon.ionhiveserde.formats.IonOutputFormat'
LOCATION 's3://dataset-aft-vbi-pds/metadata';
```

### 👉 **Select from table**

```SQL
SELECT COUNT(*) FROM "database-aft-vbi-pds"."table_metadata";
SELECT * FROM "database-aft-vbi-pds"."table_metadata" limit 10;
```

### 👉 **Query table to retrieve data for a specific item**  

```SQL  
WITH key AS (
  SELECT 'B00CLCIQDI' AS asin
)
SELECT
  element_at(bin_fcsku_data, key.asin).asin AS asin,
  element_at(bin_fcsku_data, key.asin).name AS product_name,
  element_at(bin_fcsku_data, key.asin).normalizedname AS normalized_name,
  element_at(bin_fcsku_data, key.asin).quantity AS quantity,
  element_at(bin_fcsku_data, key.asin).height.value AS height_value,
  element_at(bin_fcsku_data, key.asin).height.unit AS height_unit,
  element_at(bin_fcsku_data, key.asin).length.value AS length_value,
  element_at(bin_fcsku_data, key.asin).length.unit AS length_unit,
  element_at(bin_fcsku_data, key.asin).width.value AS width_value,
  element_at(bin_fcsku_data, key.asin).width.unit AS width_unit,
  element_at(bin_fcsku_data, key.asin).weight.value AS weight_value,
  element_at(bin_fcsku_data, key.asin).weight.unit AS weight_unit
FROM "database-aft-vbi-pds"."table_metadata", key
WHERE element_at(bin_fcsku_data, key.asin).asin IS NOT NULL;
```   

### 👉 **Consolidate JSON files with AWS Athena CTAS**  

* In this example, 17.9MB, **10,441 JSON files** are saved as 3.9MB, **21 new files**, which will significantly improve PySpark performance, as Spark incurs overhead when opening and processing each file individually.    

```SQL
CREATE TABLE "database-aft-vbi-pds"."table_metadata_consolidated"
WITH (
  format = 'PARQUET',
  external_location = 's3://database-aft-vbi-pds/table_metadata_consolidated/',
  write_compression = 'SNAPPY' -- Optional
) AS
SELECT *
FROM "database-aft-vbi-pds"."table_metadata";
```

