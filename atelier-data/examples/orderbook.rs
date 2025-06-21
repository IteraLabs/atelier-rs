use atelier_data::orderbooks::Orderbook;

fn main() {
    let ini_bid_price = 100_000.00;
    let ini_bid_levels = Some((2, 5));
    let ini_bid_orders = Some((10, 20));

    let ini_ticksize = Some((0.1, 1.1));

    let ini_ask_price = 100_001.00;
    let ini_ask_levels = Some((2, 5));
    let ini_ask_orders = Some((10, 20));

    let r_ob = Orderbook::random(
        None,
        ini_bid_price,
        ini_bid_levels,
        ini_bid_orders,
        ini_ticksize,
        ini_ask_price,
        ini_ask_levels,
        ini_ask_orders,
    );

    // number of levels per side
    let n_bids = r_ob.bids.len();
    let n_asks = r_ob.asks.len();
    // mid price and total volume calculations
    let mid_price = (r_ob.asks[0].price + r_ob.bids[0].price) / 2.0;
    let volume_bids: f64 = r_ob.bids.clone().into_iter().map(|x| x.volume).sum();
    let volume_asks: f64 = r_ob.asks.clone().into_iter().map(|x| x.volume).sum();
    // random orders created at particular levels
    let n_orders_b0 = r_ob.bids[0].orders.len();
    let n_orders_b1 = r_ob.bids[1].orders.len();
    let n_orders_a0 = r_ob.asks[0].orders.len();
    let n_orders_a1 = r_ob.asks[1].orders.len();

    println!("\n-- Random Prices/Amounts generated --\n");
    println!("- Best bid price: {:?}", ini_bid_price);
    println!("- Best ask price: {:?}", ini_ask_price);
    println!("- Midprice: {:?}", mid_price);
    println!("- total volume in bids: {:?}", volume_bids);
    println!("- total volume in asks: {:?}", volume_asks);

    println!("\n-- Random Levels generated --\n");
    println!("- No. of levels in bids: {:?}", n_bids);
    println!("- No. of levels in asks: {:?}", n_asks);

    println!("\n-- Random Orders generated --\n");
    println!("- No. of Orders at 1st level, Bids side: {:?}", n_orders_b0);
    println!("- No. of Orders at 1st level, Bids side: {:?}", n_orders_b1);
    println!("- No. of Orders at 2nd level, Asks side: {:?}", n_orders_a0);
    println!("- No. of Orders at 2nd level, Asks side: {:?}", n_orders_a1);
    println!("");
}
