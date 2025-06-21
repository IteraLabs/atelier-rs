#[cfg(test)]

// -- ----------------------------------------------------------------- TESTS UTILS -- //
// -- ----------------------------------------------------------------- ----------- -- //

mod test_orderbook_utils {

    use atelier_data::{levels::Level, orderbooks::Orderbook, orders::OrderSide};
    use rand::{distr::Uniform, rng, Rng};

    // ------------------------------------------------------------- TEST ORDERBOOK -- //

    pub fn test_orderbook() -> Orderbook {

        let ini_bid_price = 100_000.00;
        let ini_bid_levels = Some((2, 5));
        let ini_bid_orders = Some((10, 20));

        let ini_ticksize = Some((0.1, 1.1));

        let ini_ask_price = 100_001.00;
        let ini_ask_levels = Some((2, 5));
        let ini_ask_orders = Some((10, 20));

        Orderbook::random(
            None,
            ini_bid_price,
            ini_bid_levels,
            ini_bid_orders,
            ini_ticksize,
            ini_ask_price,
            ini_ask_levels,
            ini_ask_orders,
        )

    }

    // ----------------------------------------------------------------- TEST LEVEL -- //

    pub fn test_level(testable_ob: &Orderbook) -> Result<Level, ()> {
        // Random samples
        let mut uni_rand = rng();
        let rand_level_b = uni_rand.sample(
            Uniform::new(0, testable_ob.bids.len() - 1)
                .expect("Failed to create Uniform for Bids"),
        );
        let rand_level_a = uni_rand.sample(
            Uniform::new(0, testable_ob.asks.len() - 1)
                .expect("Failed to create Uniform for Asks"),
        );

        // Get a cloned random Level from the Test Orderbook
        let random_level = match OrderSide::random() {
            OrderSide::Bids => testable_ob.bids[rand_level_b].clone(),
            OrderSide::Asks => testable_ob.asks[rand_level_a].clone(),
        };

        Ok(random_level)
    }
}

mod tests {

    // ----------------------------------------------------------------- FIND_LEVEL -- //
    // ----------------------------------------------------------------- ---------- -- //

    // --------------------------------------------------- FIND_LEVEL: OUTPUT VALUE -- //

    #[test]
    fn find_level_output_value() {
        use crate::test_orderbook_utils::{test_level, test_orderbook};

        // Get a random Orderbook from test_orderbook
        let testable_ob = test_orderbook();
        // Get a random Level from the test_level
        let random_level = test_level(&testable_ob).unwrap();

        let r_side = random_level.side.clone();
        let r_price = random_level.price.clone();
        let r_level_id = random_level.level_id.clone();

        println!("random level for the test");
        println!(
            "side: {:?}, price: {:?}, level_id: {:?}",
            r_side, r_price, r_level_id
        );

        // Find the same Level using the function
        let find_level_ob = testable_ob.find_level(&r_price);

        println!("level found result: {:?}", find_level_ob);

        match find_level_ob {
            Ok(n) if n < 0 => {
                let bid_found = find_level_ob.unwrap().abs() as usize - 1;
                println!(
                    "level found price: {:?}",
                    &testable_ob.bids[bid_found].price
                );
                let bid_level_found = testable_ob.bids[bid_found].clone();
                assert_eq!(bid_level_found, random_level);
            }

            Ok(n) if n > 0 => {
                let ask_found = find_level_ob.unwrap() as usize - 1;
                println!("level found price: {:?}", testable_ob.asks[ask_found].price);
                let ask_level_found = testable_ob.asks[ask_found].clone();
                assert_eq!(ask_level_found, random_level);
            }

            Ok(_) => {
                println!("error");
            }
            Err(_) => {}
        }
    }

    // ------------------------------------------------------------- RETRIEVE_LEVEL -- //
    // ------------------------------------------------------------- -------------- -- //

    // ----------------------------------------------- RETRIEVE_LEVEL: OUTPUT VALUE -- //

    #[test]
    fn retrieve_level_output_value() {
        use crate::test_orderbook_utils::{test_level, test_orderbook};

        // Get a random Orderbook from test_orderbook
        let testable_ob = test_orderbook();
        // Get a random Level from the test_level
        let random_level = test_level(&testable_ob).unwrap();

        let r_side = random_level.side.clone();
        let r_price = random_level.price.clone();
        let r_level_id = random_level.level_id.clone();

        println!("\nrandom_level");
        println!(
            "\nside: {:?}, price: {:?}, level_id: {:?}",
            r_side, r_price, r_level_id
        );

        let retrieved_level = testable_ob.retrieve_level(&r_price).unwrap();
        println!("\nretrieved_level: {:?}", retrieved_level.price);
        println!("assert_eq! {:?} == {:?}", r_price, retrieved_level.price);
        // assert_eq!(r_price, retrieved_level.price)
    }

    // --------------------------------------------------------------- DELETE_LEVEL -- //
    // --------------------------------------------------------------- ------------ -- //

    // ------------------------------------------------- DELETE_LEVEL: OUTPUT VALUE -- //

    //#[test]
    //fn delete_level_output_value() {
    //}

    // --------------------------------------------------------------- INSERT_LEVEL -- //
    // --------------------------------------------------------------- ------------ -- //

    // ------------------------------------------------- INSERT_LEVEL: OUTPUT VALUE -- //

    // #[test]
    // fn insert_level_output_value() {
    // }

    // ----------------------------------------------------------------- FIND_ORDER -- //
    // ----------------------------------------------------------------- ---------- -- //

    // --------------------------------------------------- FIND_ORDER: OUTPUT VALUE -- //

    // #[test]
    // fn find_order_output_value() {
    // }

    // ------------------------------------------------------------- RETRIEVE_ORDER -- //
    // ------------------------------------------------------------- -------------- -- //

    // ----------------------------------------------- RETRIEVE_ORDER: OUTPUT VALUE -- //

    // #[test]
    // fn retrieve_order_output_value() {
    // }

    // --------------------------------------------------------------- DELETE_ORDER -- //
    // --------------------------------------------------------------- ------------ -- //

    // ------------------------------------------------- DELETE_ORDER: OUTPUT VALUE -- //

    // #[test]
    // fn delete_order_output_value() {
    // }

    // --------------------------------------------------------------- INSERT_ORDER -- //
    // --------------------------------------------------------------- ------------ -- //

    // ------------------------------------------------- INSERT_ORDER: OUTPUT VALUE -- //

    // #[test]
    // fn insert_order_output_value() {
    // }

    // --------------------------------------------------------------- MODIFY_ORDER -- //
    // --------------------------------------------------------------- ------------ -- //

    // ------------------------------------------------- MODIFY_ORDER: OUTPUT VALUE -- //

    // #[test]
    // fn modify_order_output_value() {
    // }
}
