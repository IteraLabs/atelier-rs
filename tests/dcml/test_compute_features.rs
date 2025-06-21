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
        use crate::test_utils::test_orderbook;
        use atelier_dcml::features;

        // Get a random Orderbook from test_orderbook
        let ob_data_0 = test_orderbook(50);
        let ob_data_1 = test_orderbook(150);
        let ob_data_2 = test_orderbook(5000);

        let i_spread = features::compute_spread(&ob_data_0);
        let i_midprice = features::compute_midprice(&ob_data_0);
        let i_w_midprice = features::compute_w_midprice(&ob_data_0);
        let i_imb = features::compute_imb(&ob_data_0);
        let i_vwap = features::compute_vwap(&ob_data_0, 5);
        let i_tav = features::compute_tav(&ob_data_0, 0.0001);

        let i_obts = features::compute_obts(&vec![
            ob_data_0.clone(),
            ob_data_1.clone(),
            ob_data_2.clone(),
        ]);
        println!("\n");
        println!("i_spread: {:?}", i_spread);
        println!("i_midprice: {:?}", i_midprice);
        println!("i_w_midprice: {:?}", i_w_midprice);
        println!("i_imb: {:?}", i_imb);
        println!("i_vwap: {:?}", i_vwap);
        println!("i_tav: {:?}", i_tav);
        println!("i_obts: {:?}", i_obts);
    }
}
