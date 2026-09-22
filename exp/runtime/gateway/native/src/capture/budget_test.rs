use super::*;
use serde_json::json;

#[test]
fn sizes_match_serde_without_an_encoded_buffer() {
    let controls: String = (0..=127).map(char::from).collect();
    for value in [
        json!(null),
        json!(false),
        json!(true),
        json!(1),
        json!(-1),
        json!(1.5),
        json!(1e30),
        json!(u64::MAX),
        json!("雪😀 café "),
        json!(controls),
        json!([]),
        json!({}),
        json!({"nested\0": [1, true, "a\n\"b", {"x":"\\"}]}),
    ] {
        let encoded = serde_json::to_string(&value).unwrap();
        assert_eq!(json_bytes(&value), encoded.len());
        assert_eq!(
            encode(&value, encoded.len()).as_deref(),
            Some(encoded.as_str())
        );
        assert!(encode(&value, encoded.len() - 1).is_none());
    }
}

#[test]
fn node_heavy_trees_and_spare_vectors_are_charged() {
    let mut values = Vec::with_capacity(4096);
    values.push(Value::Null);
    let value = Value::Array(values);
    assert!(heap_bytes(&value) >= 4096 * std::mem::size_of::<Value>());
    let nodes = Value::Array(vec![Value::Null; 1000]);
    assert!(heap_bytes(&nodes) > json_bytes(&nodes));
}
