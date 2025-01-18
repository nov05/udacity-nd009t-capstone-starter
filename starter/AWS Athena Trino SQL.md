
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

* e.g. the JSON structure is as follows. Use `Amazon Ion Hive SerDe` since the JSON is in a "pretty print" format.   

```JSON
{
    "BIN_FCSKU_DATA": {
        "B003E72M1G": {
            "asin": "B003E72M1G",
            "height": {
                "unit": "IN",
                "value": 1.199999998776
            },
            "length": {
                "unit": "IN",
                "value": 5.099999994798
            },
            "name": "Buxton Heiress Double Cardex Wallet, Mahogany, One Size",
            "normalizedName": "Buxton Heiress Double Cardex Wallet, Mahogany, One Size",
            "quantity": 3,
            "weight": {
                "unit": "pounds",
                "value": 0.4
            },
            "width": {
                "unit": "IN",
                "value": 4.399999995512001
            }
        },
        ...
    },
    "EXPECTED_QUANTITY": 3
}
```

### 👉 **Query table** 

```SQL
SELECT COUNT(*) FROM "database-aft-vbi-pds"."table_metadata_flat";
SELECT * FROM "database-aft-vbi-pds"."table_metadata_flat" limit 2;
```

* e.g. query the flattened table result

| #  | asin        | name                                                                            | normalized_name                                                                | quantity | height_value | height_unit | length_value | length_unit | width_value | width_unit | weight_value | weight_unit |
|----|-------------|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------|----------|--------------|-------------|--------------|-------------|-------------|------------|--------------|-------------|
| 1  | B00KTFGIKM  | Lifefactory 4-Cup Glass Food Storage with Silicone Sleeve, Huckleberry           | Lifefactory 4-Cup Glass Food Storage with Silicone Sleeve, Huckleberry           | 1        | 3.49999999643 | IN          | 7.39999999245 | IN          | 6.59999999327 | IN         | 1.89999999999 | pounds     |
| 2  | B0000AY9W6  | Camco 44412 Wheel Chock - Single                                                | Camco 44412 Wheel Chock - Single                                                | 1        | 4.5669291292  | IN          | 7.9527558974  | IN          | 4.8818897588  | IN         | 0.44092002372 | pounds     |


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

### 👉 **Query table**

```SQL
SELECT COUNT(*) FROM "database-aft-vbi-pds"."table_metadata";
SELECT * FROM "database-aft-vbi-pds"."table_metadata" limit 2;
```

* e.g. Query the nested table result  

| #  | bin_fcsku_data                                                                                                                                                                                                                                     | expected_quantity |
|----|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| 1  | {B01DBHZLSO={asin=B01DBHZLSO, name=Madhu's COLLECTION MG786169C Photo Frame, 4" by 6", Multicolor, normalizedname=Madhu's COLLECTION MG786169C Photo Frame, 4" by 6", Multicolor, quantity=4, height={unit=IN, value=1.500000000000000000}, length={unit=IN, value=8.000000000000000000}, width={unit=IN, value=7.000000000000000000}, weight={unit=pounds, value=3.000000000000000000}}, 0553373056={asin=0553373056, name=Creating Love: The Next Great Stage of Growth, normalizedname=Creating Love: The Next Great Stage of Growth, quantity=1, height={unit=IN, value=1.259842518400000000}, length={unit=IN, value=8.976377943600000000}, width={unit=IN, value=5.511811018000000000}, weight={unit=pounds, value=0.970000000000000000}}} | 5                 |
| 2  | {B000CCW2V6={asin=B000CCW2V6, name=Da Vinci and the Code He Lived By (History Channel), normalizedname=Da Vinci and the Code He Lived By (History Channel), quantity=1, height={unit=IN, value=0.582677164760000000}, length={unit=IN, value=7.098425189610000000}, width={unit=IN, value=5.417322829120000000}, weight={unit=pounds, value=0.183336417079200000}}, B01AL1UO3U={asin=B01AL1UO3U, name=Galaxy S7 Edge Case, Terrapin [GENUINE LEATHER] Samsung S7 Edge Case Executive [Black] Premium Wallet Case with Card Slots & Bill Compartment Case for Samsung Galaxy S7 Edge - Black, normalizedname=Galaxy S7 Edge Case, Terrapin [GENUINE LEATHER] Samsung S7 Edge Case Executive [Black] Premium Wallet Case with Card Slots & Bill Compartment Case for Samsung Galaxy S7 Edge - Black, quantity=1, height={unit=IN, value=1.499999998470000000}, length={unit=IN, value=5.999999993880000000}, width={unit=IN, value=3.999999995920000000}, weight={unit=pounds, value=0.449999999622634200}}} | 2                 |


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

* Query result

| #  | asin        | product_name                                            | normalized_name                                           | quantity | height_value | height_unit | length_value | length_unit | width_value | width_unit | weight_value         | weight_unit |
|----|-------------|---------------------------------------------------------|-----------------------------------------------------------|----------|--------------|-------------|--------------|-------------|-------------|------------|----------------------|-------------|
| 1  | B00CLCIQDI  | Renew Life Daily Liver Support, 60 Veggie Capsules     | Renew Life Daily Liver Support, 60 Veggie Capsules        | 1        | 2.299999997654 | IN          | 4.799999995104 | IN          | 2.49999999745  | IN         | 0.022046001186074866 | pounds      |


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

