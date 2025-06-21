use atelier_data::orderbooks::Orderbook;
use rand::Rng;
use rand_distr::{Bernoulli, Distribution, Uniform};

fn main() {
    // -- Orderbook parameters -- //
    let ini_bid_price = 100_000.00;
    let ini_bid_levels = Some((1, 2));
    let ini_bid_orders = Some((1, 10));

    let ini_ask_price = 100_001.00;
    let ini_ask_levels = Some((1, 2));
    let ini_ask_orders = Some((1, 10));

    let ini_ticksize = Some((0.1, 1.1));

    let mut v_orderbook = vec![];
    let n_progressions = 10;

    // -- Probabilistic parameters -- //

    let uni_params = vec![0.001, 0.005];
    let ber_params = vec![0.5];

    let mut rng = rand::rng();

    // ------------------------------------------------------------------------------- //
    for _ in 0..n_progressions {
        let uni_rand = Uniform::new(uni_params[0], uni_params[1])
            .expect("Failed to create Uniform distribution sampler");
        let r_amount_ret = rng.sample(uni_rand);

        let bernoulli = Bernoulli::new(ber_params[0]).unwrap();
        let r_sign_ret = if bernoulli.sample(&mut rng) {
            1.0
        } else {
            -1.0
        };

        let v_bid_price = ini_bid_price + ini_bid_price * r_amount_ret * r_sign_ret;
        let v_ask_price = ini_ask_price + ini_ask_price * r_amount_ret * r_sign_ret;

        let r_ob = Orderbook::random(
            None,
            v_bid_price,
            ini_bid_levels,
            ini_bid_orders,
            ini_ticksize,
            v_ask_price,
            ini_ask_levels,
            ini_ask_orders,
        );

        v_orderbook.push(r_ob);
    }

    println!("\nNumber of progressions: {:?}\n", n_progressions);

    println!(
        "\nfirst 4 bid prices: {:?}, {:?}, {:?}, {:?}",
        v_orderbook[0].bids[0].price,
        v_orderbook[1].bids[0].price,
        v_orderbook[2].bids[0].price,
        v_orderbook[3].bids[0].price
    );

    println!(
        "\nfirst 4 ask prices: {:?}, {:?}, {:?}, {:?}",
        v_orderbook[0].asks[0].price,
        v_orderbook[1].asks[0].price,
        v_orderbook[2].asks[0].price,
        v_orderbook[3].asks[0].price
    );
}
