fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --8<-- [start:read]
    use polars::prelude::*;

    // --8<-- [start:write]
    let mut df = df!(
        "foo" => &[1, 2, 3],
        "bar" => &[None, Some("bak"), Some("baz")],
    )
    .unwrap();

    let mut file = std::fs::File::create("docs/assets/data/path.csv").unwrap();
    CsvWriter::new(&mut file).finish(&mut df).unwrap();
    // --8<-- [end:write]

    let df = CsvReadOptions::default()
        .try_into_reader_with_file_path(Some("docs/assets/data/path.csv".into()))
        .unwrap()
        .finish()
        .unwrap();
    // --8<-- [end:read]
    println!("{df}");

    // --8<-- [start:schema_overwrite]
    let csv = "id,value\n1,2\n2,text\n";

    let mut schema_overwrite = Schema::with_capacity(1);
    schema_overwrite.insert("value".into(), DataType::String);

    let df = CsvReadOptions::default()
        .with_schema_overwrite(Some(std::sync::Arc::new(schema_overwrite)))
        .into_reader_with_file_handle(std::io::Cursor::new(csv))
        .finish()
        .unwrap();

    assert_eq!(df.column("value")?.dtype(), &DataType::String);
    // --8<-- [end:schema_overwrite]
    
    // --8<-- [start:scan]
    let lf = LazyCsvReader::new(PlRefPath::new("docs/assets/data/path.csv"))
        .finish()
        .unwrap();
    // --8<-- [end:scan]
    println!("{}", lf.collect()?);

    Ok(())
}
