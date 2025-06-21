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

    // ------------------------------------------------------------- FIT PDF PARAMS -- //

    #[test]
    fn test_probabilistic_fit() {
        use crate::test_utils::test_orderbook;
        use atelier_dcml::features;
        use atelier_generators::probabilistic;
        use atelier_generators::probabilistic::{Sampling, PDF};

        // Get a random Orderbook from test_orderbook
        let ob_data_0 = test_orderbook(0);
        let ob_data_1 = test_orderbook(0);
        let ob_data_2 = test_orderbook(0);
        let ob_data_3 = test_orderbook(0);

        let v_orderbooks = vec![
            ob_data_0.clone(),
            ob_data_1.clone(),
            ob_data_2.clone(),
            ob_data_3.clone(),
        ];

        println!("\n: timestamps in microseconds, arbitrary values");

        let l_obts = features::compute_obts(&v_orderbooks);
        println!("orderbook_ts updates: {:?}", &l_obts);

        let f_lambda = 250.0;
        let mut lmbd = probabilistic::Poisson { lambda: f_lambda };
        println!("Before fit: {:?} - Samples: {:?}", lmbd, lmbd.sample(4));

        lmbd.fit(l_obts.clone());
        let n_lambda = lmbd.lambda.clone();
        println!("After fit: {:?} - Samples: {:?}", lmbd, lmbd.sample(4));

        // Verify lambda was updated
        let lambdas_diff = (f_lambda - n_lambda).abs();
        println!("lambdas_diff: {:?}", lambdas_diff);
        assert!(lambdas_diff < 500.0);
    }
}
