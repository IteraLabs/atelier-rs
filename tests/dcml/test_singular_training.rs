
#[cfg(test)]

// -- ----------------------------------------------------------------- TESTS UTILS -- //
// -- ----------------------------------------------------------------- ----------- -- //

mod test_utils {

    use atelier_data::orderbooks::Orderbook;

    // ------------------------------------------------------------- TEST ORDERBOOK -- //

    pub fn test_orderbook(update_ts: u64) -> Orderbook {
        Orderbook::random(
            Some(update_ts),
            100_000.0,
            Some((5, 10)),
            Some((20, 30)),
            Some((0.1, 1.0)),
            100_001.0,
            Some((5, 10)),
            Some((20, 30)),
        )
    }
}

mod tests {

    // ----------------------------------------------------------- COMPUTE_FEATURES -- //

    #[test]
    fn test_compute_features() {

    }

}

