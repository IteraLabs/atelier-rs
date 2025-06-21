use crate::{
    levels::Level,
    orders::{Order, OrderSide, OrderType},
};

use atelier_results::errors::{LevelError, OrderError};
use rand::{distr::Uniform, Rng};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Orderbook {
    pub orderbook_id: u32,
    pub orderbook_ts: u64,
    pub symbol: String,
    pub bids: Vec<Level>,
    pub asks: Vec<Level>,
}

impl Orderbook {
    // -------------------------------------------------------------- New Orderbook -- //
    // -------------------------------------------------------------- ------------- -- //

    /// Creates a new instance of `Orderbook`.
    ///
    /// # Parameters
    ///
    /// - `orderbook_id`: The unique identifier for the order book.
    /// - `orderbook_ts`: The timestamp for the order book.
    /// - `symbol`: The trading symbol for the order book.
    /// - `bids`: A vector of `Level` representing the buy orders.
    /// - `asks`: A vector of `Level` representing the sell orders.
    ///
    /// # Returns
    ///
    /// Returns a new `Orderbook` instance.
    pub fn new(
        orderbook_id: u32,
        orderbook_ts: u64,
        symbol: String,
        bids: Vec<Level>,
        asks: Vec<Level>,
    ) -> Self {
        Orderbook {
            orderbook_id,
            orderbook_ts,
            symbol,
            bids,
            asks,
        }
    }

    // --------------------------------------------------------------- Find a Level -- //
    // --------------------------------------------------------------- ------------ -- //

    /// If a level exists, either within the Bids, or, the Asks,
    /// it will return the index of it, positive for asks, negative for bids.
    ///
    /// ## Parameters
    /// level_price: f64 = The Level's price to be found.
    ///
    /// ## Returns
    /// Ok(i32): Index of the Level found (if it exists),
    /// the sign encodes the side, negative (bids) and positive (asks)
    /// Err(LevelError): with LevelNotFound
    ///
    pub fn find_level(&self, level_price: &f64) -> Result<i32, LevelError> {
        let mut i_level: i32 = 0;

        if level_price <= &self.bids[0].price {
            for i_bid in &self.bids {
                i_level -= 1;
                if level_price == &i_bid.price {
                    return Ok(i_level);
                }
            }
        }

        if level_price >= &self.asks[0].price {
            for i_ask in &self.asks {
                i_level += 1;
                if level_price == &i_ask.price {
                    return Ok(i_level);
                }
            }
        }

        Err(LevelError::LevelNotFound)
    }

    // ------------------------------------------------- Retrieve an Existing Level -- //
    // ------------------------------------------------- -------------------------- -- //

    /// If a Level exists, either within the Bids, or, the Asks, it will
    /// return a _cloned()_ version of it.
    ///
    /// ## Parameters
    /// level_price: f64 : The level's price to be used as index.
    ///
    /// ## Returns
    /// Ok(Level) : A cloned version of the founded Level. \
    /// Err(LevelError): A custom error type as LevelError:LevelNotFound
    pub fn retrieve_level(&self, level_price: &f64) -> Result<Level, LevelError> {
        // return the level_price if it exists, or, LevelError::LevelNotFound
        if let Ok(i_level) = self.find_level(level_price) {
            println!("i_level: {:?}, self.bids: {:?}", i_level, self.bids.len());
            println!("i_level: {:?}, self.asks: {:?}", i_level, self.asks.len());

            // Level is on the Bid side
            if i_level < 0 {
                let i_level = i_level.abs();
                return Ok(self.bids[i_level as usize].clone());

            // Level is on the Ask side
            } else if i_level > 0 {
                let i_level = i_level - 1;
                return Ok(self.asks[i_level as usize].clone());

            // level is not present
            } else {
                return Err(LevelError::LevelInfoNotAvailable);
            }

        // find_level returned err in the first place
        } else {
            return Err(LevelError::LevelNotFound);
        }
    }

    // --------------------------------------------------- Delete an Existing Level -- //
    // --------------------------------------------------- ------------------------ -- //

    /// Deletes an existing level
    ///
    /// ## Parameters
    /// level_price: &f64

    pub fn delete_level(&mut self, level_price: &f64) -> Result<(), LevelError> {
        // see if level exists
        let find_level_ob = self.find_level(level_price);

        match find_level_ob {
            Ok(n) if n < 0 => {
                let bid_found = find_level_ob.unwrap().abs() as usize - 1;
                self.bids.remove(bid_found);
                Ok(())
            }

            Ok(n) if n > 0 => {
                let ask_found = find_level_ob.unwrap() as usize - 1;
                self.asks.remove(ask_found);
                Ok(())
            }

            Err(_e) => Err(LevelError::LevelDeletionFailed),

            Ok(_) => Err(LevelError::LevelInfoNotAvailable),
        }
    }

    // --------------------------------------------------------- Insert a New Level -- //
    // --------------------------------------------------------- ------------------ -- //

    /// Inserts a new level. If the level already exists, the new level over
    /// rides the existing one, if it does not exists, the new level is inserted
    /// in its corresponding slot within the `Vec<Level>` for the corresponding
    /// side.
    ///
    /// ## Parameters
    /// level: With a Level::new()
    ///
    /// ## Returns
    /// Ok(Level)
    /// Err(LevelError): Custom Error Type of LevelInsertionFailed.
    pub fn insert_level(&mut self, level: Level) -> Result<(), LevelError> {
        // return the level_price if it exists, or, LevelError::LevelNotFound
        if let Ok(i_level) = self.find_level(&level.price) {
            println!("i_level found: {:?}", &i_level);

            // -- Level exist on the Bid side (to be replaced)
            if i_level < 0 {
                // updated counter to this side
                let i_level = i_level.abs() - 1;
                // use the same id for the level
                let same_level_id = &self.bids[i_level as usize].level_id;
                // override existing level with the new one
                self.bids[i_level as usize] = Level::new(
                    *same_level_id,
                    OrderSide::Bids,
                    level.price,
                    level.volume,
                    level.orders,
                );

                return Ok(());

            // -- Level exist on the Ask side (to be replaced)
            } else if i_level > 0 {
                // update counter to this side
                let i_level = i_level - 1;
                // use the same id for the level
                let same_level_id = &self.asks[i_level as usize].level_id;
                // override existing level with new one
                self.asks[i_level as usize] = Level::new(
                    *same_level_id,
                    level.side,
                    level.price,
                    level.volume,
                    level.orders,
                );

                return Ok(());

            // A response was produced but with an error on level index
            } else {
                // find the right position within the vector
                // and insert new Level there

                println!("i_level found: {:?}", &i_level);

                return Err(LevelError::LevelNotFound);
            }
        // Level not found.
        // Localize index, insert into vector.
        } else {
            match level.side {
                OrderSide::Bids => {
                    // Get the level_price for all levels in the vector
                    let mut v_bids = self.bids.clone().into_iter();

                    // Find position : Start with the upper most position
                    // (given the ordering in orders)
                    let index_level = v_bids
                        .position(|existing_level| level.price > existing_level.price)
                        .unwrap_or(v_bids.len());

                    // Insert new Level into the existing vector of levels.
                    self.bids.insert(
                        index_level,
                        Level::new(
                            index_level as u32,
                            level.side,
                            level.price,
                            level.volume,
                            level.orders,
                        ),
                    );
                    return Ok(());
                }

                OrderSide::Asks => {
                    // get the level_price for all levels in the vector
                    let mut v_asks = self.asks.clone().into_iter();

                    // Find position : Start with the lower and top most position
                    // (given the ordering in orders)
                    let index_level = v_asks
                        .position(|existing_level| level.price < existing_level.price)
                        .unwrap_or(v_asks.len());

                    // Insert the new level into the existing vector of levels.
                    self.asks.insert(
                        index_level,
                        Level::new(
                            index_level as u32,
                            level.side,
                            level.price,
                            level.volume,
                            level.orders,
                        ),
                    );
                    return Ok(());
                }
            }
        }
    }

    // -------------------------------------------------------------- Find an Order -- //
    // -------------------------------------------------------------- ------------- -- //

    /// To find if a given `Order` exists within the current Level.
    ///
    /// ## Parameters
    /// side: Side = {Side::Bids, Side::Asks}
    /// price: f64 = the order's price, which is the same as the Level's price
    /// order_ts: u64 = Order's timestamp
    ///
    /// ## Results
    /// Ok: (level_index: usize, order_index: usize)
    /// Err: OrderError

    pub fn find_order(
        &self,
        price: f64,
        order_ts: u64,
    ) -> Result<(i32, usize), OrderError> {
        // see if level exists
        let find_level_ob = self.find_level(&price);

        match find_level_ob {
            Ok(n) if n < 0 => {
                let level_found = find_level_ob.unwrap().abs() as usize - 1;
                let level_orders = &self.bids[level_found].orders;

                // Level has orders
                if level_orders.len() > 0 {
                    let r_level = level_orders
                        .binary_search_by(|order| order.order_ts.cmp(&order_ts))
                        .unwrap();

                    Ok((n, r_level))

                // Level is empty
                } else {
                    Err(OrderError::OrderNotFound)
                }
            }

            Ok(n) if n > 0 => {
                let level_found = find_level_ob.unwrap().abs() as usize - 1;
                let level_orders = &self.asks[level_found].orders;

                // Level has orders
                if level_orders.len() > 0 {
                    let r_level = level_orders
                        .binary_search_by(|order| order.order_ts.cmp(&order_ts))
                        .unwrap();

                    Ok((n, r_level))

                // Level is empty
                } else {
                    Err(OrderError::OrderNotFound)
                }
            }

            Err(_e) => Err(OrderError::OrderNotFound),
            Ok(_) => Err(OrderError::OrderInfoNotAvailable),
        }
    }

    // ------------------------------------------------- Retrieve an Existing Order -- //
    // ------------------------------------------------- -------------------------- -- //

    /// To retrieve info about an existing `Order`.
    ///
    /// ## Parameters
    /// Order
    ///
    /// ## Results
    ///  

    pub fn retrieve_order(&self, price: f64, order_ts: u64) -> Result<Order, OrderError> {
        if let Ok((found_level, found_order)) = self.find_order(price, order_ts) {
            if found_level > 0 {
                Ok(self.asks[found_level.abs() as usize].orders[found_order])
            } else {
                Ok(self.bids[found_level.abs() as usize].orders[found_order])
            }
        } else {
            Err(OrderError::OrderNotFound)
        }
    }

    // --------------------------------------------------- Delete an Existing Order -- //
    // --------------------------------------------------- ------------------------ -- //

    /// To delete an existing `Order`.
    ///
    /// ## Parameters
    ///
    /// ## Results

    pub fn delete_order(&mut self, price: f64, order_ts: u64) -> Result<(), OrderError> {
        if let Ok((found_level, found_order)) = self.find_order(price, order_ts) {
            if found_level > 0 {
                self.asks[found_level.abs() as usize - 1]
                    .orders
                    .remove(found_order);

                Ok(())
            } else {
                self.bids[found_level.abs() as usize - 1]
                    .orders
                    .remove(found_order);
                Ok(())
            }
        } else {
            Err(OrderError::OrderNotFound)
        }
    }

    // --------------------------------------------------------- Insert a New Order -- //
    // --------------------------------------------------------- ------------------ -- //

    /// To insert a new `Order`.
    ///
    /// The first process is to find whether the necessary Level in the
    /// Orderbook exists. Then, depending on which side is it (or has to be
    /// created) all the parameters for Order::new() are created and parsed.
    ///
    /// ## Parameters
    ///
    ///
    /// ## Results

    pub fn insert_order(&mut self, price: f64, amount: f64) -> Result<(), OrderError> {
        // see if level exists
        let find_level_ob = self.find_level(&price);

        match find_level_ob {
            Ok(n) if n < 0 => {
                // Get the curren timestamp
                let bid_ts = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_micros() as u64;

                let i_order = Order::builder()
                    .order_ts(bid_ts)
                    .order_type(OrderType::Limit)
                    .side(OrderSide::Bids)
                    .price(self.bids[n as usize].price)
                    .amount(amount)
                    .build()
                    .expect("Order Failed");

                self.bids[find_level_ob.unwrap() as usize]
                    .orders
                    .push(i_order);

                Ok(())
            }

            Ok(n) if n > 0 => {
                // Get the curren timestamp
                let ask_ts = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_micros() as u64;

                let i_order = Order::builder()
                    .order_ts(ask_ts)
                    .order_type(OrderType::Limit)
                    .side(OrderSide::Asks)
                    .price(self.asks[n as usize].price)
                    .amount(amount)
                    .build()
                    .expect("Order Failed");

                self.asks[find_level_ob.unwrap() as usize]
                    .orders
                    .push(i_order);

                Ok(())
            }

            Err(_e) => Err(OrderError::OrderNotFound),
            Ok(_) => Err(OrderError::OrderInfoNotAvailable),
        }
    }

    // ------------------------------------------------------------ Modify an Order -- //
    // ------------------------------------------------------------ --------------- -- //

    /// To modify an existing `Order`.
    ///
    /// ## Parameters
    ///
    /// ## Results

    pub fn modify_order(
        &mut self,
        order_ts: u64,
        price: f64,
        amount: f64,
    ) -> Result<Order, OrderError> {
        match self.find_order(price, order_ts) {
            Ok((found_level, found_order)) => {
                if found_level < 0 {
                    println!("\nfounded level: {:?}", found_level.abs() as usize - 1);

                    let founded_order =
                        self.bids[found_level.abs() as usize].orders[found_order].clone();

                    println!("\nfounded order: {:?}", founded_order);

                    let _moded_ts = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_micros();

                    let founded_ts = founded_order.order_ts;

                    //  TODO: validate same order_id when a modification of
                    // an order is done by creating a new one with all the
                    // same except the amount

                    let to_moded_order = Order::builder()
                        .side(founded_order.side)
                        .order_type(founded_order.order_type)
                        .order_ts(founded_ts)
                        .price(founded_order.price.unwrap())
                        .amount(amount)
                        .build()
                        .expect("Modification of order failed");

                    let moded_order = to_moded_order;

                    Ok(moded_order.clone())
                } else if found_level > 0 {
                    println!("\nfounded_level: {:?}", found_level.abs() as usize - 1);

                    let founded_order =
                        self.asks[found_level.abs() as usize].orders[found_order].clone();

                    println!("founded order: {:?}", founded_order);

                    let _moded_ts = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_micros();

                    let founded_ts = founded_order.order_ts;

                    //  TODO: validate same order_id when a modification of
                    // an order is done by creating a new one with all the same
                    // except the amount

                    let to_moded_order = Order::builder()
                        .side(founded_order.side)
                        .order_type(founded_order.order_type)
                        .order_ts(founded_ts)
                        .price(founded_order.price.unwrap())
                        .amount(amount)
                        .build()
                        .expect("Modification of order failed");

                    let moded_order = to_moded_order;

                    Ok(moded_order.clone())
                } else {
                    println!("else");

                    Err(OrderError::OrderNotFound)
                }
            }
            Err(_e) => Err(OrderError::OrderNotFound),
        }
    }

    // ----------------------------------------------------------- Random Orderbook -- //
    // ----------------------------------------------------------- ---------------- -- //

    /// Generates a synthetic order book with specified parameters.
    ///
    /// This method is useful for benchmarking and simulation purposes.
    ///
    /// # Parameters
    ///
    /// - `bids_price`: The Best Bid (Top Of the Book).
    /// - `bids_levels`: The amount of levels to create in the Buy (bids) side.
    /// - `bids_orders`: Parameters of the distribution to sample values from.
    ///   Uniform ~ (u32, u32).
    /// - `tick_size`: Parameters of the distribution to sample from. Uniform ~
    ///   (f64, f64).
    /// - `asks_price`: The Best Ask (Top Of the Book).
    /// - `asks_levels`: The amount of levels to create in the Sell (asks) side.
    /// - `asks_orders`: Parameters of the distribution to sample from. Uniform
    ///   ~ (f64, f64).
    ///
    /// # Returns
    ///
    /// Returns a new `Orderbook` instance populated with synthetic bid and
    /// ask levels.
    ///
    /// TODO: update this to be done with builder method.

    pub fn random(
        update_ts: Option<u64>,
        bids_price: f64,
        bids_levels: Option<(u32, u32)>,
        bids_orders: Option<(u32, u32)>,

        tick_size: Option<(f64, f64)>,

        asks_price: f64,
        asks_levels: Option<(u32, u32)>,
        asks_orders: Option<(u32, u32)>,
    ) -> Self {
        let mut rng = rand::rng();

        // -- Default values -- //
        let mut i_bids = Vec::new();
        let mut i_asks = Vec::new();

        // Handle update_ts with default value
        let current_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        let r_orderbook_ts = match update_ts {
            Some(ts) => current_ts + ts,
            _ => current_ts
        };

        //  TODO: Change this to a hashed formation of the Orderbook ID
        let r_orderbook_id = 1234;

        // -- Generate random value for bids_levels & ask_levels

        let n_bids_levels = rng.sample(
            Uniform::new(bids_levels.unwrap().0, bids_levels.unwrap().1)
                .expect("Failed to create random n_bids_levels"),
        );

        let n_asks_levels = rng.sample(
            Uniform::new(asks_levels.unwrap().0, asks_levels.unwrap().1)
                .expect("Failed to create random n_asks_levels"),
        );

        // -- Generate all the ticks ahead -- //

        // -- Bids
        let mut v_bids_ticks: Vec<f64> = if let Some(bids_range) = tick_size {
            let uni_rand =
                Uniform::new(bids_range.0, bids_range.1).expect("Failed to create distr");
            (0..n_bids_levels).map(|_| rng.sample(uni_rand)).collect()
        } else {
            let uni_rand =
                Uniform::new(0.0, 1.0).expect("Failed to create Standard Uniform");
            (0..n_bids_levels).map(|_| rng.sample(uni_rand)).collect()
        };

        v_bids_ticks.insert(0, 0.0);
        let mut v_bids_prices: Vec<f64> = vec![bids_price];

        // -- Asks
        let mut v_asks_ticks: Vec<f64> = if let Some(asks_range) = tick_size {
            let uni_rand =
                Uniform::new(asks_range.0, asks_range.1).expect("Failed to create distr");
            (0..n_asks_levels).map(|_| rng.sample(uni_rand)).collect()
        } else {
            let uni_rand =
                Uniform::new(0.0, 1.0).expect("Failed to create Standard Uniform");
            (0..n_asks_levels).map(|_| rng.sample(uni_rand)).collect()
        };

        v_asks_ticks.insert(0, 0.0);
        let mut v_asks_prices: Vec<f64> = vec![asks_price];

        // ----------------------------------------------------- Bid Side Formation -- //

        for i in 1..=n_bids_levels {
            // -- Id formation

            //  TODO: Change this to a hashed formation of the Level ID
            let i_bids_id = 4321;

            // -- Side formation

            let i_bids_side = OrderSide::Bids;

            // -- Price formation

            let i_bids_price =
                v_bids_prices[(i - 1) as usize] - v_bids_ticks[(i - 1) as usize];
            v_bids_prices.push(i_bids_price);

            // -- Orders formation

            let i_bids_orders = if let Some(bid_orders_range) = bids_orders {
                rng.random_range(bid_orders_range.0..bid_orders_range.1)
            } else {
                rng.sample(Uniform::new(1, 5).unwrap())
            };

            let mut v_bids_orders: Vec<Order> = (0..i_bids_orders)
                .map(|_| {
                    Order::random(
                        OrderType::Limit,
                        OrderSide::Bids,
                        (10_000.01, 11_000.01),
                        (0.001, 0.100),
                    )
                    .unwrap()
                })
                .collect();

            v_bids_orders.sort_by_key(|order| order.order_ts);

            // -- Volume formation

            let i_bids_volume: f64 = v_bids_orders
                .iter()
                .map(|order| order.amount.unwrap_or(0.0))
                .sum();

            // -- Result formation

            i_bids.push(Level {
                level_id: i_bids_id,
                side: i_bids_side,
                price: i_bids_price,
                volume: i_bids_volume,
                orders: v_bids_orders,
            });
        }

        // ----------------------------------------------------- Ask Side Formation -- //

        for i in 1..=n_asks_levels {
            let i_asks_id = 7654;

            let i_asks_side = OrderSide::Asks;

            let i_asks_price =
                v_asks_prices[(i - 1) as usize] - v_asks_ticks[(i - 1) as usize];
            v_asks_prices.push(i_asks_price);

            let i_asks_orders = if let Some(asks_orders_range) = asks_orders {
                rng.random_range(asks_orders_range.0..asks_orders_range.1)
            } else {
                rng.sample(Uniform::new(1, 5).unwrap())
            };

            let mut v_asks_orders: Vec<Order> = (0..i_asks_orders)
                .map(|_| {
                    Order::random(
                        OrderType::Limit,
                        OrderSide::Asks,
                        (10_000.01, 11_000.01),
                        (0.001, 0.100),
                    )
                    .unwrap()
                })
                .collect();

            v_asks_orders.sort_by_key(|order| order.order_ts);

            // -- Volume formation

            let i_asks_volume: f64 = v_asks_orders
                .iter()
                .map(|order| order.amount.unwrap_or(0.0))
                .sum();

            // -- Result formation

            i_asks.push(Level {
                level_id: i_asks_id,
                side: i_asks_side,
                price: i_asks_price,
                volume: i_asks_volume,
                orders: v_asks_orders,
            });
        }

        Orderbook {
            orderbook_id: r_orderbook_id,
            orderbook_ts: r_orderbook_ts,
            symbol: String::from("BTCUSDT"),
            bids: i_bids,
            asks: i_asks,
        }
    }
}
